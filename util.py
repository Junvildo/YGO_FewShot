# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.utils.data import DataLoader


class SimpleLogger(object):
    def __init__(self, logfile, terminal):
        ZERO_BUFFER_SIZE = 0  # immediately flush logs

        self.log = open(logfile, 'a', ZERO_BUFFER_SIZE)
        self.terminal = terminal

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# Function to calculate mean and std
def calculate_mean_std(data_path: str):

    loader = DataLoader(data_path, batch_size=64, shuffle=False, num_workers=4)
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
