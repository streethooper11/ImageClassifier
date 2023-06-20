# Contains a series of functions that loads a saved machine learning model and
# given arguments, such as data directory, save directory, model architecture, etc.
#
# Author: Sehwa Kim
# Date Created: 06/19/2023
# Date Modified: 06/20/2023
##
# import modules
import json
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from get_input_args import get_input_args_predict
from model import load_checkpoint


def resize_image(image, size):
    """
    Resizes the PIL image.
    It resizes to the given size; the shorter side is converted to the given size, and the longer side
    is scaled appropriately to keep the aspect ratio.

    Parameters:
        1. image: The PIL image
        2. size: The size to resize to
    """
    width, height = image.size

    if width < height:
        new_width = size
        new_height = int(height * (new_width / width))
    else:
        new_height = size
        new_width = int(width * (new_height / height))

    return image.resize((new_width, new_height))


def process_image(image):
    """
    Scales, crops, and normalizes a PIL image to be used for input in inference.
    Specifically, it is resized to 256 pixels then center cropped to 224x224.
    It is then normalized and transposed to follow the torch specifications

    Parameters:
        1. image: The PIL image

    Returns:
        A torch tensor with transforms applied
    """
    # Resize the PIL image
    image = resize_image(image, 256)
    # Crop the center 224*224; use pytorch CenterCrop method
    image = transforms.CenterCrop(224)(image)

    # Get a numpy array from the result
    np_image = np.array(image, dtype=float)

    # Convert the array to have values 0-1 instead of 0-255
    np_image /= 255.0

    # Normalize by subtracting the means then dividing by std
    np_image -= np.array([0.485, 0.456, 0.406])
    np_image /= np.array([0.229, 0.224, 0.225])

    # Reorder dimensions by reversing the order from the imshow function
    np_image = np_image.transpose(2, 0, 1)

    # Convert numpy array to tensor in FloatTensor type to match the weights when making predictions
    torch_image = torch.from_numpy(np_image).type(torch.FloatTensor)

    # Return the tensor
    return torch_image


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


def predict(image_path, model, topk=5):
    """
    Predicts the class(es) of an image using a trained model.

    Parameters:
        1. image_path: Path of the image used for inference
        2. model: Model used to make predictions
        3. topk: The Top K categories that the model predicts
    """
    # Ensure the model is in the evaluation mode
    model.eval()

    # Open image with PIL and process the image
    image_processed = process_image(Image.open(image_path))

    # Unsqueeze to match dimensions
    # Source: https://discuss.pytorch.org/t/expected-stride-to-be-a-single-integer-value-or-a-list/17612
    image_input = torch.unsqueeze(image_processed, 0)

    # Send the resulting tensor to the device being used
    image_input = image_input.to(device)

    # Convert to tensor and get predictions
    # Converting to tensor source: https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html
    outputs = model(image_input)

    # Identical to batch_accuracy function code, except topk argument is now given
    ps = torch.exp(outputs)  # LogSoftmax was used, so perform e^result to get probabilities
    top_ps, top_class = ps.topk(topk, dim=1)  # Top topk probabilities and classes along the columns

    return image_processed, top_ps, top_class


def plot_predictions(image_processed, top_class, top_ps):
    """
    Plots the predictions made by the model with its image

    Parameters:
        1. image_processed: The image that went through transformations
        2. top_class: Top K class names
        3. top_ps: Top K probabilities for corresponding class names

    Returns:
        None; the result plots the result and shows it as a figure
    """

    # Add the result to the matplotlib figure
    # Add the image first
    # Put the tensor in CPU so it can be converted to numpy array
    image_processed = image_processed.cpu()

    # Using subplots: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
    fig, (ax1, ax2) = plt.subplots(2, 1)

    # Use imshow to show image
    ax1 = imshow(image_processed, ax1)
    ax1.set_title('Orange Dahlia')
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Add the horizontal bar chart
    # Modified from https://matplotlib.org/stable/gallery/lines_bars_and_markers/barh.html
    ax2.barh(top_class, top_ps)
    ax2.invert_yaxis()
    ax2.set_xlabel('Probability')

    plt.show()


def print_prediction_result(top_class, top_ps, top_k):
    """
    Prints the prediction result on the console

    Parameters:
        1. top_class: Top K class names
        2. top_ps: Top K probabilities for corresponding class names
        3. top_k: Number of K

    Return:
        None; the function simply prints out the results
    """

    print('Top {} predictions:'.format(top_k))

    for i in range(top_k):
        print('{}. {}: {:.2f}'.format(i + 1, top_class[i], top_ps[i]))

if __name__ == "__main__":
    in_arg = get_input_args_predict()

    # Define device; if gpu is selected, use gpu unless it is not available
    if in_arg.gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    # Load the checkpoint and get the resulting model and optimizer
    model, optimizer, curr_epoch = load_checkpoint(in_arg.checkpoint, device)

    # Predict the probabilities and classes from the given image path
    image_processed, top_ps, top_idx = predict(in_arg.image_path, model, topk=in_arg.top_k)

    # The results are in idx; make an reverse dictionary of class_to_idx
    # and retrieve by the key to get the numbered labels
    # Dictionary comprehension source: https://www.programiz.com/python-programming/dictionary-comprehension
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}

    # Load the data structure that maps class number to names
    with open(in_arg.category_names, 'r') as f:
        cat_to_name = json.load(f)

    # Flatten and convert the tensor outputs into a 1D list to be used for plotting
    top_ps = top_ps.reshape(-1).tolist()
    top_idx = top_idx.reshape(-1).tolist()

    # Change the IDs to list and find the values from the inverse dictionary, which are class numbers
    top_labels = [idx_to_class[x] for x in top_idx]

    # Get names from the class numbers
    top_class = [cat_to_name[x] for x in top_labels]

    print_prediction_result(top_class, top_ps, in_arg.top_k)
    plot_predictions(image_processed, top_class, top_ps)
