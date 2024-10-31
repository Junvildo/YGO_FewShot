# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import os


class SimpleLogger(object):
    def __init__(self, logfile, terminal):
        self.log = open(logfile, 'a', buffering=1)  # Line buffered mode for text
        self.terminal = terminal

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()  # Flush terminal output
        self.log.write(message)
        self.log.flush()  # Immediately flush log to file

    def flush(self):
        # Explicitly flushing both the terminal and log file
        self.terminal.flush()
        self.log.flush()


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
