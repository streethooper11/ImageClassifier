# Based on get_input_args.py from pet classifier project
# Contains functions that parses and returns arguments.
#
# Author: Sehwa Kim
# Date Created: 06/19/2023
# Date Modified: 06/20/2023
##
# Imports python modules
import argparse
import os

def get_input_args_train():
    """
    Retrieves and parses command line arguments provided for training the model
    It throws an error if the first argument is missing; a default value is given for each of the remaining arguments.
    Command Line Arguments:
        1. Directory of images used for training/validation/testing
        2. Save folder as --save_dir; 'save' used as the default value
        3. Model architecture as --arch; 'resnet18' used as the default value
        4. Learning rate as --learning_rate; 0.001 used as the default value
        5. Number of neurons for the hidden layer as --hidden_units;
            512 used as the default value
        6. Number of epochs as --epochs; 15 used as the default value

    Parameters:
        None; the command line arguments are parsed

    Returns:
        A data structure that contains all parsed arguments
    """

    # Create a parser using ArgumentParser
    parser = argparse.ArgumentParser()
        
    # Create command line arguments
    parser.add_argument('data_dir', type=str,
                        help='Path to the folder of images')
    parser.add_argument('--save_dir', type=str, default='save',
                        help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='resnet18',
                        choices=['alexnet', 'resnet18', 'resnet50', 'vgg11', 'vgg13', 'vgg16'],
                        help='Name of the CNN model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate of the optimizer')
    parser.add_argument('--hidden_units', type=int, default=512,
                        help='Number of neurons in the hidden layer')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of epochs for training')
    parser.add_argument('--gpu', action='store_true',
                        help='Set to use GPU when training')

    # Parse arguments and return the result
    return parser.parse_args()


def get_input_args_predict():
    """
    Retrieves and parses command line arguments provided for making predictions
    It throws an error if any of the first two arguments is missing.
    A default value is given for each of the remaining arguments.
    Command Line Arguments:
        1. Image path used for prediction
        2. Checkpoint path to load states
        3. Return Top K as --top_k; 1 used as the default value
        4. Path of file used for mapping of categories as --category_names;
                        'cat_to_name.json' used as the default value
        5. GPU flag as --gpu; flag is off by default

    Parameters:
        None; the arguments are simply parsed

    Returns:
        A data structure that contains all parsed arguments
    """

    # Create a parser using ArgumentParser
    parser = argparse.ArgumentParser()

    # Create command line arguments
    # An Orange Dahlia from the testing folder will be given as the default value
    parser.add_argument('image_path', type=str, default=os.path.join('flowers', 'train', '59', 'image_05031.jpg'),
                        help='Path to the image to ake predictions')
    parser.add_argument('checkpoint', type=str, default='checkpoint.pth',
                        help='Path to the checkpoint to load')
    parser.add_argument('--top_k', type=int, default=1,
                        help='Number of top ranks to return')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='Path of the file that maps categories to real names')
    parser.add_argument('--gpu', action='store_true',
                        help='Set to use GPU when training')

    # Parse arguments and return the result
    return parser.parse_args()
