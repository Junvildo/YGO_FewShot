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
import sys
from util import SimpleLogger, calculate_mean_std
from mobileone import mobileone, reparameterize_model
from models import EmbeddedFeatureWrapper
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sampler import ClassBalancedBatchSampler
from data import CustomDataset
from torch.utils.data.dataloader import default_collate
import losses
import time
from extract_features import extract_feature
from retrieval import evaluate_float_binary_embedding_faiss


def parse_args():
    """
    Helper function parsing the command line options
    """
    parser = ArgumentParser(description="PyTorch metric learning training script")
    # Optional arguments for the launch helper
    parser.add_argument("--dataset_root", type=str, default="./main_dataset",
                        help="The root directory to the dataset")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--img_size", type=int, default=7, help="Image size for training")
    parser.add_argument("--model_variant", type=str, default="s2", help="MobileOne variant (s0, s1, s2, s3, s4)")
    parser.add_argument("--lr", type=float, default=0.01, help="The base lr")
    parser.add_argument("--gamma", type=float, default=0.1, help="Gamma applied to learning rate")
    parser.add_argument("--class_balancing", default=True, action='store_true', help="Use class balancing")
    parser.add_argument("--images_per_class", type=int, default=2, help="Images per class")
    parser.add_argument("--lr_mult", type=float, default=1, help="lr_mult for new params")
    parser.add_argument("--dim", type=int, default=2048, help="The dimension of the embedding")

    parser.add_argument("--test_every_n_epochs", type=int, default=1, help="Tests every N epochs")
    parser.add_argument("--epochs_per_step", type=int, default=1, help="Epochs for learning rate step")
    parser.add_argument("--pretrain_epochs", type=int, default=5, help="Epochs for pretraining")
    parser.add_argument("--num_steps", type=int, default=1, help="Num steps to take")
    parser.add_argument("--output", type=str, default="./output", help="The output folder for training")

    return parser.parse_args()

def adjust_learning_rate(optimizer, epoch, epochs_per_step, gamma=0.1):
    """Sets the learning rate to the initial LR decayed by 10 every epochs"""
    # Skip gamma update on first epoch.
    if epoch != 0 and epoch % epochs_per_step == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= gamma
            print("learning rate adjusted: {}".format(param_group['lr']))

def main():
    args = parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    output_directory = os.path.join(args.output, str(args.dim),
                                    '_'.join([args.model_variant, str(args.batch_size)]))
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    out_log = os.path.join(output_directory, "train.log")
    sys.stdout = SimpleLogger(out_log, sys.stdout)

    # Select model
    baseline = mobileone(variant=args.model_variant)
    model = EmbeddedFeatureWrapper(feature=baseline, input_dim=2048, output_dim=args.dim)

    # Calculate mean and std of data
    mean, std = calculate_mean_std(data_path=args.dataset_root)

    # Setup train and eval transformations
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ColorJitter(brightness=(0.5,1.5),contrast=(0.3,2.0),hue=.05, saturation=(.0,.15)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-270,270)),
        transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
        transforms.RandomAffine(0, translate=(0,0.3), scale=(0.6,1.8), shear=(0.0,0.4), fill=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    eval_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Setup dataset
    train_dataset = CustomDataset(root=args.dataset_root, train=True, transform=train_transform)
    eval_dataset = CustomDataset(root=args.dataset_root, train=False, transform=eval_transform)

    # Setup dataset loader
    if args.class_balancing:
        print("Class Balancing")
        train_sampler = ClassBalancedBatchSampler(train_dataset.instance_labels, args.batch_size, args.images_per_class)
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

    model.to(device=device)
    loss_fn.to(device=device)

    # Training mode
    model.train()

    # Start with pretraining where we finetune only new parameters to warm up
    opt = torch.optim.SGD(list(loss_fn.parameters()) + list(set(model.parameters()) -
                                                            set(model.feature.parameters())),
                          lr=args.lr * args.lr_mult, momentum=0.9, weight_decay=1e-4)

    log_every_n_step = 10
    for epoch in range(args.pretrain_epochs):
        for i, (im, _, instance_label, index) in enumerate(train_loader):
            data = time.time()
            opt.zero_grad()

            im = im.to(device=device, non_blocking=True)
            instance_label = instance_label.to(device=device, non_blocking=True)

            forward = time.time()
            embedding = model(im)
            loss = loss_fn(embedding, instance_label)

            back = time.time()
            loss.backward()
            opt.step()

            end = time.time()

            if (i + 1) % log_every_n_step == 0:
                print('Epoch {}, LR {}, Iteration {} / {}:\t{}'.format(
                    args.pretrain_epochs - epoch, opt.param_groups[0]['lr'], i, len(train_loader), loss.item()))

                print('Data: {}\tForward: {}\tBackward: {}\tBatch: {}'.format(
                    forward - data, back - forward, end - back, end - forward))

        eval_file = os.path.join(output_directory, 'epoch_{}'.format(args.pretrain_epochs - epoch))
        embeddings, labels = extract_feature(model, eval_loader, device)
        evaluate_float_binary_embedding_faiss(embeddings, embeddings, labels, labels, eval_file, k=1000)

if __name__ == '__main__':
    main()