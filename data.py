import os
import random
from PIL import Image
from abc import ABCMeta, abstractmethod
from torchvision import transforms
from torch.utils.data import Dataset as BaseDataset
import numpy as np
import torch

class Dataset(object):
    """
    This abstract class defines the interface needed for dataset loading.
    All concrete subclasses need to implement the following method/property:

    def name: returns the name of the dataset
    def image_root_dir: the root directory of the images
    def _load: the actual logic to load the dataset,
        It needs to populate these three lists:
        self.image_paths = []
        self.class_labels = []
        self.instance_labels = []
    """
    __metaclass__ = ABCMeta

    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.image_paths = []
        self.class_labels = []
        self.instance_labels = []

        self._load()

        # Check dataset is loaded properly
        assert len(self.image_paths) > 0, "No images loaded."
        assert len(self.class_labels) == len(self.image_paths), "Mismatch between image paths and class labels."

    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError("Subclasses should implement this!")

    @property
    @abstractmethod
    def image_root_dir(self):
        raise NotImplementedError("Subclasses should implement this!")

    @property
    @abstractmethod
    def _load(self):
        raise NotImplementedError("Subclasses should implement this!")

    def __getitem__(self, index):
        im_path = self.image_paths[index]
        im = Image.open(im_path).convert('RGB')
        if self.transform is not None:
            im = self.transform(im)
        class_target = self.class_labels[index]
        return im, class_target, index

    def __len__(self):
        return len(self.image_paths)


class CustomDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.train = train
        self.transform = transform
        super(CustomDataset, self).__init__(root, train, transform)
        print("Loaded {} samples for dataset {}, {} classes, {} instances".format(
            len(self), self.name, self.num_cls, self.num_instance))

    @property
    def name(self):
        return 'custom_dataset_{}'.format('train' if self.train else 'test')

    @property
    def image_root_dir(self):
        return os.path.join(self.root, 'train' if self.train else 'test')

    @property
    def num_cls(self):
        return len(self.class_map)

    @property
    def num_instance(self):
        return len(self.image_paths)

    @property
    def class_labels_list(self):
        return self.class_labels

    def _load(self):
        self.class_map = {}
        self.image_paths = []
        self.class_labels = []

        # List all class folders in the root/train or root/test directory
        for class_name in sorted(os.listdir(self.image_root_dir)):
            class_dir = os.path.join(self.image_root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            # Assign a unique id to each class if not already in the map
            if class_name not in self.class_map:
                self.class_map[class_name] = len(self.class_map)

            # Load all images for the class
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if not img_path.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue
                self.image_paths.append(img_path)
                self.class_labels.append(self.class_map[class_name])

    def __getitem__(self, index):
        im_path = self.image_paths[index]
        im = Image.open(im_path).convert('RGB')
        if self.transform is not None:
            im = self.transform(im)
        class_target = self.class_labels[index]
        return im, class_target, index

    # def __getitem__(self, idx):
    #     Open the image
    #     img_path = self.image_paths[idx]
    #     image = Image.open(img_path).convert('RGB')
        
    #     Convert the image to a NumPy array
    #     image = np.asarray(image)
        
    #     Extract the folder name as the label
    #     label = self.class_labels[idx]
        
    #     Apply transformation if provided
    #     if self.transform:
    #         image = self.transform(image=image)["image"]
        
    #     return image, label


    def __len__(self):
        return len(self.image_paths)


class InferenceDataset(BaseDataset):
    def __init__(self, numpy_arrays, transform=None):
        """
        Args:
            numpy_arrays: List of numpy arrays representing images.
            transform: Transformations to be applied to each image.
        """
        self.numpy_arrays = numpy_arrays
        self.transform = transform

    def __len__(self):
        return len(self.numpy_arrays)

    def __getitem__(self, idx):
        image = self.numpy_arrays[idx]
        if self.transform:
            # Apply transformations (convert numpy to PIL image first)
            image = Image.fromarray(image)
            image = self.transform(image)
        return image
    
class ImageDataset(BaseDataset):
    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, idx):
        # Open the image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Extract the folder name as the label
        label = os.path.basename(os.path.dirname(image_path))
        
        # Apply transformation if provided
        if self.transform:
            image = self.transform(image)
        
        return image, label



if __name__ == '__main__':
    # Load the training set
    data_path = 'dataset_artworks'
    custom_train_set = CustomDataset(data_path, train=True)
    print("Loaded {} samples for dataset {}".format(len(custom_train_set), custom_train_set.name))
    for i in random.sample(range(0, len(custom_train_set)), 5):
        image, label = custom_train_set[i]
        print("Image: {} | Label: {} | Class Name: {}".format(
            custom_train_set.image_paths[i], label, list(custom_train_set.class_map.keys())[label]))

    # Load the test set
    custom_test_set = CustomDataset(data_path, train=False)
    print("Loaded {} samples for dataset {}".format(len(custom_test_set), custom_test_set.name))
    for i in random.sample(range(0, len(custom_test_set)), 5):
        image, label = custom_test_set[i]
        print("Image: {} | Label: {} | Class Name: {}".format(
            custom_test_set.image_paths[i], label, list(custom_test_set.class_map.keys())[label]))
