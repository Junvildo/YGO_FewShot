# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import os
import matplotlib.pyplot as plt


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
def calculate_mean_std(data_path: str):
    data = ImageFolder(os.path.join(data_path, 'train'), transform=transforms.Compose([transforms.Resize((7, 7)), transforms.ToTensor()]))
    loader = DataLoader(data, batch_size=2, shuffle=False, num_workers=4)
    mean = 0.0
    std = 0.0
    total_images_count = 0
    
    for images, _ in loader:
        images = images.view(images.size(0), images.size(1), -1)  # Flatten each image to [C, H*W]
        mean += images.mean(2).sum(0)  # Sum mean of each channel across all images
        std += images.std(2).sum(0)    # Sum std of each channel across all images
        total_images_count += images.size(0)

    mean /= total_images_count
    std /= total_images_count
    return mean, std
