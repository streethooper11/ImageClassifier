# Contains a series of functions that trains and saves a machine learning model based on
# given arguments, such as data directory, save directory, model architecture, etc.
#
# Author: Sehwa Kim
# Date Created: 06/19/2023
# Date Modified: 06/20/2023
##
# import modules
import torch
import os

from get_dataloaders import create_dataloaders
from get_input_args import get_input_args_train
from model import get_model_optimizer, save_checkpoint


# Source: Transfer Learning lesson from Deep Learning with PyTorch section
def batch_accuracy(outputs, labels):
    """
    Calculates accuracy of a batch from the dataloader.

    Parameters:
        1. outputs: Labels classified by the classifier model in a batch of data
        2. labels: True labels given for the same batch of data

    Returns:
        Accuracy of the batch in decimals
    """

    ps = torch.exp(outputs)  # LogSoftmax was used, so perform e^result to get probabilities
    top_ps, top_class = ps.topk(1, dim=1)  # Get top 1 along the columns
    # Count the number of matches between the classifier label (top one) and the actual label
    equality = top_class == labels.view(*top_class.shape)
    # Accuracy is the average of the matches along the batch
    accuracy = torch.mean(equality.type(torch.FloatTensor)).item()

    return accuracy


def test(device, model, criterion, dataloaders, mode='valid'):
    """
    Validates/tests the classifier model with a set of data

    Parameters:
        1. device: The device to use; cuda or cpu
        2. model: The classifier model to test with
        3. criterion: The loss function used
        4. dataloaders: A data structure that contains multiple DataLoader objects
        5. mode: Mode to use; 'valid' and 'test' are possible options; it is used to retrieve a
            DataLoader object from dataloaders

    Returns:
        Resulting loss and accuracy
    """

    model.eval()  # Put the model in the evaluation mode
    loss = 0
    accuracy = 0

    for images, labels in dataloaders[mode]:
        # Move data to device to use
        images, labels = images.to(device), labels.to(device)

        # Turn off gradients for speed
        with torch.no_grad():
            logps = model(images)  # Obtain output

        loss = criterion(logps, labels)  # Calculate loss from classifier label and true label
        loss += loss.item()  # Track total loss
        accuracy += batch_accuracy(logps, labels)  # Track accuracy

    return loss, accuracy


def print_results(dataloaders, test_loss, test_accuracy, train_loss=0, mode='valid', epoch=-1, epochs=-1):
    """
    Prints the results. The function is called after validation or test.

    Parameters:
        1. dataloaders: A data structure that contains multiple DataLoader objects
        1. test_loss: Loss from validation/testing
        2. test_accuracy: Accuracy from validation/testing
        3. train_loss: Loss from training; only used during training and validation
        4. mode: Mode to use; 'valid' and 'test' are possible options; it is used to retrieve a
            DataLoader object from dataloaders
        5. epoch: Current epoch number; only used during training and validation
        6. epochs: The total number of epochs for training

    Returns:
        None; the function prints data
    """

    # print epoch and train information if it is in validation mode
    if mode == 'valid':
        print('Epoch {}/{}..'.format(epoch + 1, epochs))
        print('Train loss: {:.3f}..'.format(train_loss / len(dataloaders['train'])))

    print('{} loss: {:.3f}'.format(mode.title(), test_loss / len(dataloaders[mode])))
    print('{} accuracy: {:.3f}'.format(mode.title(), test_accuracy / len(dataloaders[mode])))


def train(device, model, criterion, dataloaders, epochs):
    """
    Trains the classifier model with a set of data

    Parameters:
        1. device: The device to use; cuda or cpu
        2. model: The classifier model to test with
        3. criterion: The loss function used
        4. dataloaders: A data structure that contains multiple DataLoader objects
        5. epochs: The total number of epochs to train

    Returns:
        None; the function simply trains the model
    """

    # Train the network; based on the Transfer Learning lesson
    # Train print validation results for each epoch
    # Start with training mode
    model.train()
    curr_epoch = 0
    while curr_epoch < epochs:
        train_loss = 0

        for images, labels in dataloaders['train']:
            # Move data to device to use
            images, labels = images.to(device), labels.to(device)

            logps = model(images)  # Obtain output
            loss = criterion(logps, labels)  # Calculate loss from classifier label and true label

            train_loss += loss.item()  # Track running loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        valid_loss, valid_accuracy = test(device, model, criterion, dataloaders, mode='valid')
        print_results(dataloaders, valid_loss, valid_accuracy, train_loss, mode='valid', epoch=curr_epoch)

        model.train()  # Put the model in the training mode
        curr_epoch += 1


if __name__ == "__main__":
    # Parse command line arguments
    in_arg = get_input_args_train()

    # Define device; if gpu is selected, use gpu unless it is not available
    if in_arg.gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    # Get dataloaders from the given data directory with the indices mapping
    dataloaders, class_to_idx = create_dataloaders(in_arg.data_dir)

    model, optimizer, classifier = get_model_optimizer(device, in_arg.arch, in_arg.hidden_units, in_arg.learning_rate)

    # Define loss function
    criterion = torch.nn.NLLLoss()  # Negative log likelihood loss

    # Define variables used
    epochs = in_arg.epochs

    # Train the model
    train(device, model, criterion, dataloaders, in_arg.epochs)

    # Do validation on the test set
    test_loss, test_accuracy = test(device, model, criterion, dataloaders, mode='test')
    print_results(dataloaders, test_loss, test_accuracy, mode='test')

    # Create the save directory if it does not exist
    os.makedirs(in_arg.save_dir, exist_ok=True)

    # Save the state after training the model
    checkpoint_path = os.path.join(in_arg.save_dir, 'checkpoint.pth')
    save_checkpoint(checkpoint_path, model, optimizer, epochs,
                    class_to_idx, classifier, in_arg.arch, in_arg.learning_rate)
