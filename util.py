# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch



def log_and_print(message, log_file):
    print(message)
    log_file.write(message + "\n")
    log_file.flush()

def plot_metrics(metric_values, title, ylabel, output_path):
    plt.figure()
    plt.plot(metric_values, marker='o')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid()
    plt.savefig(output_path)
    plt.close()

# Function to calculate mean and std
def calculate_mean_std(dataset_path, transform, device, batch_size=32):
    """
    Calculate mean and standard deviation of a dataset on 3 color channels.

    Args:
        dataset_path (str): Path to the dataset directory.
        transform (torchvision.transforms.Compose): Transformations to apply.
        device (torch.device): Device to perform calculations on.
        batch_size (int): Batch size for DataLoader.

    Returns:
        tuple: Mean and standard deviation for each channel (R, G, B).
    """
    # Load dataset
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Accumulate pixel sums and squared sums
    total_sum = torch.zeros(3, device=device)
    total_squared_sum = torch.zeros(3, device=device)
    total_pixels = 0

    for images, _ in tqdm(dataloader, desc="Calculating mean and std"):
        # Move images to the specified device
        images = images.to(device)

        # Flatten image batch to calculate per-channel stats
        images = images.view(images.size(0), images.size(1), -1)  # (batch_size, channels, H * W)
        total_sum += images.sum(dim=(0, 2))
        total_squared_sum += (images ** 2).sum(dim=(0, 2))
        total_pixels += images.size(0) * images.size(2)

    # Calculate mean and std
    mean = total_sum / total_pixels
    std = torch.sqrt((total_squared_sum / total_pixels) - (mean ** 2))

    return mean.tolist(), std.tolist()
