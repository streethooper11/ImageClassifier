# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

## What the project does

There are two files, train.py and predict.py.\
train.py is used for training the model, and predict.py is used for inference.

train.py will train and save checkpoint with its state, and predict.py will
make a prediction based on the model's checkpoint and print the results out on
the console while plotting the predictions on a figure.

## Arguments

### Arguments for train.py

1. (Required) Directory of images to be used for training/validation/testing
2. (Optional) --save_dir: Save directory
3. (Optional) --arch: Model architecture to use
    - The following options are available:
      - alexnet, resnet18, resnet50, vgg11, vgg13, vgg16
4. (Optional) --learning_rate: Learning rate of the optimizer
5. (Optional) --hidden_units: Number of neurons in the hidden layer
6. (Optional) --epochs: Number of epochs for training
7. (Optional) --gpu: GPU on/off flag

### Arguments for predict.py

1. (Required) Path of image used for prediction
2. (Required) Path of checkpoint file
3. (Optional) --top_k: Number of top classes to retrieve from predictions
4. (Optional) --category_name: Path of the file that maps category ids to names
5. (Optional) --gpu: GPU on/off flag

### Examples

python train.py flowers --save_dir save --arch alexnet --learning_rate 0.001

- Uses flowers as the root data folder
- Uses save as the save folder
- Uses AlexNet as its model architecture
- Uses 0.001 as its learning rate
- Uses 512 as its hidden neurons by default
- Uses 15 as its number of epochs by default
- Uses CPU by default

python predict.py image.jpg checkpoint.pth --top_k 5 --category_names name.json --gpu

- Uses image.jpg as inference file
- Uses checkpoint.pth to load states for model and optimizer
- Makes top 5 predictions
- Uses name.json as its mapping file
- Uses GPU

### Notes
1. train.py assumes the image directory includes 3 subfolders; train, valid, test.
The subfolders will be used for training, validation, and testing, respectively.
2. predict.py assumes the inference file, the checkpoint file, and the mapping file all exist
and the arguments are correctly defined.