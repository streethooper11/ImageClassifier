# Contains functions related to machine learning models
# Such functions include initializing a model, saving a checkpoint and loading a checkpoint
#
# Author: Sehwa Kim
# Date Created: 06/20/2023
# Date Modified: 06/20/2023
##
# Imports python modules
import torch
from torchvision import models
import os


def get_model_optimizer(device, arch, hidden_units, learning_rate):
    """
    Initializes a machine learning model and its optimizer given its architecture, hidden layers and learning rate.
    alexnet, vgg11, vgg13, vgg16 will have their classifier attribute replaced
    resnet18 and resnet50 will have their fc attribute replaced

    Parameters:
        1. device: Device to send the model to (GPU/CPU)
        2. arch: The model architecture
        3. hidden_units: The number of neurons for the hidden layer
        4. learning_rate: Learning rate of the optimizer

    Return:
        Initialized model, optimizer, and the classifier defined
    """

    # Load a pre-trained model depending on its architecture
    # Input units are from the source code of each model
    if arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        replace_fc = False
        input_units = 4096
    elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
        replace_fc = True
        input_units = 512
    elif arch == 'resnet50':
        model = models.resnet50(pretrained=True)
        replace_fc = True
        input_units = 2048
    elif arch == 'vgg11':
        model = models.vgg11(pretrained=True)
        replace_fc = False
        input_units = 4096
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
        replace_fc = False
        input_units = 4096
    else:
        # The remaining option is vgg16
        model = models.vgg16(pretrained=True)
        replace_fc = False
        input_units = 4096

    # Turn off gradients
    for param in model.parameters():
        param.requires_grad = False

    num_classes = 102  # Number of flower categories

    # Replace the model's classifier with a new one
    classifier = torch.nn.Sequential(
        torch.nn.Linear(input_units, hidden_units),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.4),
        torch.nn.Linear(hidden_units, num_classes),
        torch.nn.LogSoftmax(dim=1)
    )

    if replace_fc:
        model.fc = classifier  # It only has 1 layer; replace with the new classifier
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate)
    else:
        # The classifier is dense; replace only the last classifier layer
        # with the new one so some pretrained layers remain
        model.classifier[-1] = classifier
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Send model to the set device
    model.to(device)

    return model, optimizer, classifier


def save_checkpoint(checkpoint_path, model, optimizer, epoch, class_to_idx, classifier, arch, lr):
    """
    Saves a data structure that contains current state to a file

    Parameters:
        1. checkpoint_path: The file path to save the checkpoint to
        2. model: The trained model
        3. optimizer: The optimizer used
        4. curr_epoch: Current epoch number
        5. class_to_idx: Mapping from classes to indices
        6. classifier: The classifier used
        7. arch: The model archetype
        8. lr: Learning rate of the optimizer

    Return:
        None; the function simply saves the file
    """

    # Include the states, epoch, class to indices mapping, classifier and the model archetype
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch,
        'class_to_idx': class_to_idx,
        'classifier': classifier,
        'arch': arch,
        'lr': lr,
    }

    torch.save(checkpoint, checkpoint_path)


# Define a function that loads a checkpoint and rebuilds the model and the optimizer
# Returns the rebuilt model, the rebuilt optimizer, and the number of epochs the model already trained for
def load_checkpoint(checkpoint_path, device):
    """
    Loads the checkpoint from the given path and rebuilds the model and optimizer

    Parameters:
        1. checkpoint_path: File path of the checkpoint
        2. device: Device to use (GPU/CPU)

    Returns:
        Created model, optimizer, and epoch number
    """

    # Load the checkpoint dictionary
    checkpoint_load = torch.load(checkpoint_path)

    # Load a pre-trained model depending on its architecture
    # Input units are from the source code of each model
    if checkpoint_load['arch'] == 'alexnet':
        model = models.alexnet(pretrained=True)
        replace_fc = False
    elif checkpoint_load['arch'] == 'resnet18':
        model = models.resnet18(pretrained=True)
        replace_fc = True
    elif checkpoint_load['arch'] == 'resnet50':
        model = models.resnet50(pretrained=True)
        replace_fc = True
    elif checkpoint_load['arch'] == 'vgg11':
        model = models.vgg11(pretrained=True)
        replace_fc = False
    elif checkpoint_load['arch'] == 'vgg13':
        model = models.vgg13(pretrained=True)
        replace_fc = False
    else:
        # The remaining option is vgg16
        model = models.vgg16(pretrained=True)
        replace_fc = False

    # Turn off gradients
    for param in model.parameters():
        param.requires_grad = False

    if replace_fc:
        model.fc = checkpoint_load['classifier']  # It only has 1 layer; replace with the new classifier
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=checkpoint_load['lr'])
    else:
        # The classifier is dense; replace only the last classifier layer
        # with the new one so some pretrained layers remain
        model.classifier[-1] = checkpoint_load['classifier']
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=checkpoint_load['lr'])

    # Load model's state
    model.load_state_dict(checkpoint_load['model_state'])

    # Load the optimizer's state
    optimizer.load_state_dict(checkpoint_load['optimizer_state'])

    # Load indices mapping
    model.class_to_idx = checkpoint_load['class_to_idx']

    # Send the model to the set device
    model.to(device)

    return model, optimizer, checkpoint_load['epoch']
