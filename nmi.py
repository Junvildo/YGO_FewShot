import faiss
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from argparse import ArgumentParser
from models import EmbeddedFeatureWrapper
from torch.utils.data import DataLoader
from torchvision import transforms
from data import CustomDataset
from torch.utils.data.dataloader import default_collate
from extract_features import extract_feature
from mobileone import mobileone
import torch
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(description="CPU-compatible metric learning NMI script")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="The number of workers for eval")
    parser.add_argument("--snap", type=str,default="./finetuned_models/s2_56_epoch_45.pth",
                        help="The snapshot to compute NMI")
    parser.add_argument("--dataset", type=str, default="./main_dataset",
                        help="The dataset to compute NMI")
    return parser.parse_args()


def test_nmi(embeddings, labels):
    unique_labels = np.unique(labels)
    kmeans = KMeans(n_clusters=unique_labels.size, random_state=0, n_init=10).fit(embeddings)

    nmi = normalized_mutual_info_score(kmeans.labels_, labels)

    print("NMI: {}".format(nmi))
    return nmi


def test_nmi_faiss(embeddings, labels):
    # Setup FAISS k-means for CPU
    unique_labels = np.unique(labels)
    d = embeddings.shape[1]  # Dimension of embeddings

    # Initialize k-means with FAISS
    kmeans = faiss.Clustering(d, unique_labels.size)
    kmeans.verbose = True
    kmeans.niter = 300
    kmeans.nredo = 10
    kmeans.seed = 0

    # Use FAISS CPU index
    index = faiss.IndexFlatL2(d)

    # Train FAISS k-means
    kmeans.train(embeddings, index)

    # Assign each embedding to its nearest cluster
    dists, pred_labels = index.search(embeddings, 1)
    pred_labels = pred_labels.squeeze()

    # Calculate NMI between true labels and predicted labels
    nmi = normalized_mutual_info_score(labels, pred_labels)

    print("NMI: {}".format(nmi))
    return nmi


if __name__ == '__main__':
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trans = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((56, 56)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = CustomDataset(args.dataset, transform=trans, train=False)
    dataloader = DataLoader(dataset, batch_size=8, num_workers=args.num_workers,pin_memory=True, collate_fn=default_collate)
    model = EmbeddedFeatureWrapper(feature=mobileone(variant="s2"), input_dim=2048, output_dim=2048)
    state_dict = torch.load(args.snap, map_location=device, weights_only=True)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    embeddings, labels = extract_feature(loader=dataloader, model=model, device=device)
    test_nmi(embeddings, labels)
    test_nmi_faiss(embeddings, labels)

