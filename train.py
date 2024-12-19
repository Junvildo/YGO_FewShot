'''
Training process:
1. get args from input
2. setup device, output directory, logging
3. get model, set output dim
4. setup train & eval transformation rules.
5. get data to torch dataset
6. use ClassBalanceBatchSampler to form train & eval dataloader
7. init NormSoftmaxLoss
8. move model and loss function to device, set model to train mode.
9. Training:
    9a: 
        1. Pretrained new layer of the model and the loss function
        2. Extract feature and evaluate_float_binary_embedding_faiss
    9b:
        1. Finetune whole model
        2. Adjust learning rate starting from epoch 1 (exclude epoch 0)
        3. Extract feature and evaluate_float_binary_embedding_faiss every set epoch
'''
from argparse import ArgumentParser
import torch
import os
from util import log_and_print, plot_metrics
from mobileone import mobileone
from models import EmbeddedFeatureWrapper
from torchvision import transforms
from torch.utils.data import DataLoader
from sampler import ClassBalancedBatchSampler
from data import CustomDataset
from torch.utils.data.dataloader import default_collate
import losses
import time
from extract_features import extract_feature
from retrieval import evaluate_float_binary_embedding_faiss
from itertools import chain
import random
import numpy as np


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def adjust_learning_rate(optimizer, epoch, epochs_per_step, gamma=0.1):
    """Sets the learning rate to the initial LR decayed by 10 every epochs"""
    # Skip gamma update on first epoch.
    if epoch != 0 and epoch % epochs_per_step == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= gamma
            print("learning rate adjusted: {}".format(param_group['lr']))

def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    output_directory = os.path.join(args.output, str(args.dim),
                                    '_'.join([args.model_variant, str(args.batch_size)]))
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    # Open log file for writing
    out_log = os.path.join(output_directory, "train_log.txt")
    log_file = open(out_log, "w")

    # Select model
    baseline = mobileone(variant=args.model_variant)
    if args.pretrain_path != "":
        log_and_print('Load pretrain model checkpoint', log_file)
        checkpoint = torch.load(args.pretrain_path, map_location=device, weights_only=True)
        baseline.load_state_dict(checkpoint)
    model = EmbeddedFeatureWrapper(feature=baseline, input_dim=2048, output_dim=args.dim)


    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet mean and std


    # Setup train and eval transformations
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomResizedCrop((args.img_size, args.img_size)),
        transforms.ColorJitter(brightness=(0.5,1.5),contrast=(0.3,2.0),hue=.05, saturation=(.0,.15)),
        transforms.RandomAffine(degrees=(1, 359), translate=(0,0.3), scale=(0.6,1.8), shear=(0.0,0.4), fill=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    eval_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Setup dataset
    train_dataset = CustomDataset(root=args.dataset_root, train=True, transform=train_transform)
    eval_dataset = CustomDataset(root=args.dataset_root, train=False, transform=eval_transform)

    # Setup dataset loader
    if args.class_balancing:
        print("Class Balancing")
        train_sampler = ClassBalancedBatchSampler(train_dataset.class_labels_list, args.batch_size, args.images_per_class)
        train_loader = DataLoader(train_dataset,
                                  batch_sampler=train_sampler, num_workers=4,
                                  pin_memory=True, drop_last=False, collate_fn=default_collate)
    else:
        print("No class balancing")
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  drop_last=False,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=4)
        
    eval_loader = DataLoader(eval_dataset,
                            batch_size=args.batch_size,
                            drop_last=False,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=4)

    # Setup loss function
    loss_fn = losses.NormSoftmaxLoss(args.dim, train_dataset.num_instance)

    model = torch.nn.DataParallel(model)
    model.to(device=device)

    loss_fn = torch.nn.DataParallel(loss_fn)
    loss_fn.to(device=device)

    # Training mode
    model.train()

    opt = torch.optim.AdamW(list(loss_fn.parameters()) + list(set(model.module.parameters()) -
                                                            set(model.module.feature.parameters())),
                          lr=args.lr * args.lr_mult, betas=(0.9, 0.999), weight_decay=1e-4)

    # Lists to store max_f and max_b for pretraining and finetuning
    pretrain_max_r_f, pretrain_max_r_b = [], []
    pretrain_max_p_f, pretrain_max_p_b = [], []
    finetune_max_r_f, finetune_max_r_b = [], []
    finetune_max_p_f, finetune_max_p_b = [], []
    pretrain_losses, finetune_losses = [], []

    log_every_n_step = args.log_per_n_steps
    print("Start pretraining for {} epochs".format(args.pretrain_epochs))
    print("="*80)
    for epoch in range(args.pretrain_epochs):
        begin = time.time()
        epoch_loss = 0.0
        for i, (im, instance_label, _) in enumerate(train_loader):
            opt.zero_grad()

            im = im.to(device=device, non_blocking=True)
            instance_label = instance_label.to(device=device, non_blocking=True)

            embedding = model(im)
            loss = loss_fn(embedding, instance_label)

            loss.mean().backward()
            opt.step()


            epoch_loss += loss.mean().item()
            if (i + 1) % log_every_n_step == 0:
                log_and_print(f'Epoch {epoch}, LR {opt.param_groups[0]["lr"]}, Iteration {i} / {len(train_loader)} loss:\t{loss.mean().item()}', log_file)
        
        average_loss = epoch_loss / max(1, len(train_loader))
        pretrain_losses.append(average_loss)
        log_and_print(f'Epoch {epoch} average loss: {average_loss}', log_file)

        finish = time.time()
        remaining_epochs = args.pretrain_epochs - epoch - 1
        estimate_finish_time = round((finish - begin) * remaining_epochs, 5)
        print(f'Pretrain: Epoch {epoch} finished in {finish - begin} seconds, estimated finish time: {estimate_finish_time}s\n')
        log_and_print(f'Pretrain: Epoch {epoch} finished in {finish - begin} seconds, estimated finish time: {estimate_finish_time}s\n', log_file)
        if epoch == 0 or epoch == args.pretrain_epochs - 1:
            eval_file = os.path.join(output_directory, 'epoch_{}'.format(args.pretrain_epochs - epoch))
            embeddings, labels = extract_feature(model, eval_loader, device, step=log_every_n_step)
            max_r_f, max_r_b, max_p_f, max_p_b = evaluate_float_binary_embedding_faiss(embeddings, embeddings, labels, labels, eval_file, k=1000)
            model.train()

            # Store max_f and max_b
            finetune_max_r_f.append(max_r_f)
            finetune_max_r_b.append(max_r_b)
            finetune_max_p_f.append(max_p_f)
            finetune_max_p_b.append(max_p_b)

    print("="*80)
    print("Pretraining finished")

    # Full end-to-end finetune of all parameters
    model.train()
    opt = torch.optim.AdamW(chain(model.module.parameters(), loss_fn.module.parameters()), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    print("Start finetuning for {} epochs".format(args.epochs_per_step * args.num_steps))
    print("="*80)
    for epoch in range(args.epochs_per_step * args.num_steps):
        begin = time.time()
        log_and_print(f'Output Directory: {output_directory}', log_file)
        adjust_learning_rate(opt, epoch, args.epochs_per_step, gamma=args.gamma)
        
        epoch_loss = 0.0
        for i, (im, instance_label, _) in enumerate(train_loader):

            opt.zero_grad()

            im = im.to(device=device, non_blocking=True)
            instance_label = instance_label.to(device=device, non_blocking=True)

            embedding = model(im)
            loss = loss_fn(embedding, instance_label)

            loss.mean().backward()
            opt.step()


            epoch_loss += loss.mean().item()
            if (i + 1) % log_every_n_step == 0:
                log_and_print(f'Epoch {epoch}, LR {opt.param_groups[0]["lr"]}, Iteration {i} / {len(train_loader)} loss:\t{loss.mean().item()}', log_file)

        average_loss = epoch_loss / max(1, len(train_loader))
        finetune_losses.append(average_loss)
        log_and_print(f'Epoch {epoch} average loss: {average_loss}', log_file)

        finish = time.time()
        remaining_epochs = (args.epochs_per_step * args.num_steps) - epoch - 1
        estimate_finish_time = round((finish - begin) * remaining_epochs, 5)
        log_and_print(f'Finetune: Epoch {epoch} finished in {finish - begin} seconds, estimated finish time: {estimate_finish_time}s\n', log_file)
        snapshot_path = os.path.join(output_directory, 'epoch_{}.pth'.format(epoch + 1))
        torch.save(model.state_dict(), snapshot_path)

        if (epoch + 1) % args.test_every_n_epochs == 0:
            eval_file = os.path.join(output_directory, 'epoch_{}'.format(epoch + 1))
            embeddings, labels = extract_feature(model, eval_loader, device, step=log_every_n_step)
            max_r_f, max_r_b, max_p_f, max_p_b = evaluate_float_binary_embedding_faiss(embeddings, embeddings, labels, labels, eval_file, k=1000)
            model.train()

            # Store max_f and max_b
            finetune_max_r_f.append(max_r_f)
            finetune_max_r_b.append(max_r_b)
            finetune_max_p_f.append(max_p_f)
            finetune_max_p_b.append(max_p_b)
    print("="*80)
    print("Finetuning finished")
    log_file.close()

    # Save plots
    plot_metrics(pretrain_max_r_f, "Max Recall F over Pretraining Epochs", "Max F", os.path.join(output_directory, "pretrain_max_r_f.png"))
    plot_metrics(pretrain_max_r_b, "Max Racall B over Pretraining Epochs", "Max B", os.path.join(output_directory, "pretrain_max_r_b.png"))
    plot_metrics(pretrain_max_p_f, "Max Precision F over Pretraining Epochs", "Max F", os.path.join(output_directory, "pretrain_max_p_f.png"))
    plot_metrics(pretrain_max_p_b, "Max Precision B over Pretraining Epochs", "Max B", os.path.join(output_directory, "pretrain_max_p_b.png"))
    plot_metrics(pretrain_losses, "Loss over Pretraining Epochs", "Loss", os.path.join(output_directory, "pretrain_loss.png"))
    plot_metrics(finetune_max_r_f, "Max Recall F over Finetuning Epochs", "Max F", os.path.join(output_directory, "finetune_max_r_f.png"))
    plot_metrics(finetune_max_r_b, "Max Recall B over Finetuning Epochs", "Max B", os.path.join(output_directory, "finetune_max_r_b.png"))
    plot_metrics(finetune_max_p_f, "Max Precision F over Finetuning Epochs", "Max F", os.path.join(output_directory, "finetune_max_p_f.png"))
    plot_metrics(finetune_max_p_b, "Max Precision B over Finetuning Epochs", "Max B", os.path.join(output_directory, "finetune_max_p_b.png"))
    plot_metrics(finetune_losses, "Loss over Finetuning Epochs", "Loss", os.path.join(output_directory, "finetune_loss.png"))


if __name__ == '__main__':
    parser = ArgumentParser(description="PyTorch metric learning training script")
    # Optional arguments for the launch helper
    parser.add_argument("--dataset_root", type=str, default="./main_dataset",
                        help="The root directory to the dataset")
    parser.add_argument("--batch_size", type=int, default=75, help="Batch size for training")
    parser.add_argument("--log_per_n_steps", type=int, default=1000, help="Log every N steps")
    parser.add_argument("--img_size", type=int, default=224, help="Image size for training")
    parser.add_argument("--model_variant", type=str, default="s2", help="MobileOne variant (s0, s1, s2, s3, s4)")
    parser.add_argument("--lr", type=float, default=0.01, help="The base lr")
    parser.add_argument("--gamma", type=float, default=0.1, help="Gamma applied to learning rate")
    parser.add_argument("--class_balancing", default=True, action='store_true', help="Use class balancing")
    parser.add_argument("--images_per_class", type=int, default=5, help="Images per class")
    parser.add_argument("--lr_mult", type=float, default=1, help="lr_mult for new params")
    parser.add_argument("--dim", type=int, default=2048, help="The dimension of the embedding")
    parser.add_argument("--test_every_n_epochs", type=int, default=2, help="Tests every N epochs")
    parser.add_argument("--epochs_per_step", type=int, default=4, help="Epochs for learning rate step")
    parser.add_argument("--pretrain_epochs", type=int, default=5, help="Epochs for pretraining")
    parser.add_argument("--num_steps", type=int, default=3, help="Num steps to take")
    parser.add_argument("--output", type=str, default="./output", help="The output folder for training")
    parser.add_argument("--pretrain_path", type=str, default="", help="Pretrain mobileone path, end with .tar")

    # Reduce randomness
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    args = parser.parse_args()
    print(args)

    main(args)