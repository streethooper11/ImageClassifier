# Contains functions that creates a data structure that contains multiple DataLoaders that can be
# retrieved depending on the mode
#
# Author: Sehwa Kim
# Date Created: 06/20/2023
# Date Modified: 06/20/2023
##
# Imports python modules
import torch
from torchvision import datasets, transforms
import os


def create_dataloaders(data_dir):
    """
    Creates a data structure that contains multiple DataLoaders, retrievable by their corresponding keys.
    The function assumes that each dataset used in each DataLoader is separated by the following subfolders;
    train, valid, and test.

    Parameters:
        1. data_dir: A directory that contains train, valid, and test subfolders, each of which has images used
            for training, validation, and testing, respectively.

    Returns:
        A data structure that contains train, valid, test DataLoaders
        A mapping from classes to indices
    """

    # Assumption: The data directory already has 3 subfolders; train, valid, test
    # Each of them represents the corresponding dataset
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    # Define transforms for the training, validation, and testing sets; transforms from the lesson videos
    # Implement augmentation for training sets; augmentations from the lesson videos
    # Validation and testing sets share the same set of transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]),
    }

    # Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(train_dir,
                                      transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir,
                                      transform=data_transforms['test']),
        'test': datasets.ImageFolder(test_dir,
                                     transform=data_transforms['test']),
    }

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'],
                                             batch_size=32, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32),
    }

    return dataloaders, image_datasets['train'].class_to_idx
