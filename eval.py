from argparse import ArgumentParser
import torch
from mobileone import mobileone
from models import EmbeddedFeatureWrapper, GeM
from data import CustomDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from extract_features import extract_feature
from retrieval import evaluate_float_binary_embedding_faiss



if __name__ == '__main__':
    parser = ArgumentParser(description="PyTorch metric learning training script")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="The number of workers for eval")
    parser.add_argument("--snap", type=str,default="./finetuned_models/s2_56_epoch_45.pth",
                        help="The snapshot to compute")
    parser.add_argument("--dataset_root", type=str, default="./main_dataset",
                        help="The dataset to compute")
    parser.add_argument("--img_size", type=int, default=56,
                        help="The number of workers for eval")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="The number of workers for eval")
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size)),
        transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(0.3, 2.0), hue=.01, saturation=(0.5, 2.0)),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))], p=0.5),
        transforms.RandomAffine((1, 359), translate=(0, 0.3), scale=(1, 1.8), shear=(0.0, 0.8), fill=0),
        transforms.RandomPerspective(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.RandomErasing(scale=(0.02, 0.1), ratio=(0.3, 3.3), p=0.2, value=1),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])

    mean, std = [0.4935736358165741, 0.46013686060905457, 0.4618111848831177], [0.2947998642921448, 0.28370970487594604, 0.2891422510147095]

    train_transform.transforms.append(transforms.Normalize(mean=mean, std=std))
    eval_transform.transforms.append(transforms.Normalize(mean=mean, std=std))

    
    base_dataset = CustomDataset(root=args.dataset_root, train=True, transform=train_transform)
    eval_dataset = CustomDataset(root=args.dataset_root, train=True, transform=eval_transform)

    base_loader = DataLoader(base_dataset,
                            batch_size=args.batch_size,
                            drop_last=False,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=args.num_workers)
    
    eval_loader = DataLoader(eval_dataset,
                            batch_size=args.batch_size,
                            drop_last=False,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=args.num_workers)

    model = EmbeddedFeatureWrapper(feature=mobileone(variant="s2"), input_dim=2048, output_dim=2048)
    model.feature.gap = GeM()
    state_dict = torch.load(args.snap, map_location=device, weights_only=True)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    db_embeddings, db_labels = extract_feature(loader=base_loader, model=model, device=device)
    query_embeddings, query_labels = extract_feature(loader=eval_loader, model=model, device=device)
    evaluate_float_binary_embedding_faiss(query_embeddings=query_embeddings, query_labels=query_labels, db_embeddings=db_embeddings, db_labels=db_labels, output=args.snap, k=4)