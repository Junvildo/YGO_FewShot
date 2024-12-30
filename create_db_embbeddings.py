import argparse

parser = argparse.ArgumentParser(description='create db embeddings')
parser.add_argument('--model_path', type=str, required=True, help='path to model checkpoint')
parser.add_argument('--use_gem', action='store_true', default=False, help='whether to use GeM')
parser.add_argument('--is_train', action='store_true', help='whether to use the training set to create db embeddings')
parser.add_argument('--image_dir', type=str, required=True, help='path to image directory')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for data loader')
parser.add_argument('--image_size', type=int, default=56, help='image size for data loader')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loader')

args = parser.parse_args()

import torch
import faiss
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from models import EmbeddedFeatureWrapper, GeM
from mobileone import mobileone, reparameterize_model
from data import CustomDataset
from extract_features import extract_feature
import antialiased_cnns


# Model setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = EmbeddedFeatureWrapper(feature=mobileone(variant="s2"), input_dim=2048, output_dim=2048)
if args.use_gem:
    model.feature.gap = torch.nn.Sequential(antialiased_cnns.BlurPool(2048, stride=2, filt_size=1), GeM())
state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
model_eval = reparameterize_model(model)

# Transform setup
mean, std = [0.49362021684646606, 0.4601792097091675, 0.4618436098098755], [0.27437326312065125, 0.2629182040691376, 0.270280659198761]
trans = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Data setup
base_dataset = CustomDataset(root=args.image_dir, train=args.is_train, transform=trans)

base_loader = DataLoader(base_dataset,
                        batch_size=args.batch_size,
                        drop_last=False,
                        shuffle=False,
                        pin_memory=True,
                        num_workers=args.num_workers)
embeddings, labels = extract_feature(loader=base_loader, model=model_eval, device=device)
binary_db_embeddings = np.require(embeddings > 0, dtype='float32')

# Index setup
index = faiss.IndexFlatL2(binary_db_embeddings.shape[1])
index.add(binary_db_embeddings)
index_f = faiss.IndexFlatIP(embeddings.shape[1])
index_f.add(embeddings)

# Write index to disk
source = "full" if args.is_train else "org"
faiss.write_index(index, "{}_class_embeddings_{}.faiss".format(source, args.image_size))
faiss.write_index(index_f, "{}_class_embeddings_f_{}.faiss".format(source, args.image_size))