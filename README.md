# SSD300-VGG16-Pytorch
SSD300-VGG16-Pytorch-Implement

source code based on https://github.com/amdegroot/ssd.pytorch/tree/master/data

# Training SSD

First download the fc-reduced VGG-16 PyTorch base network weights at: https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
By default, we assume you have downloaded the file in the ssd-pytorch/weights dir:
mkdir weights
cd weights
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
To train SSD using the train script simply specify the parameters listed in train.py as a flag or manually change them.
python train.py
Note:
For training, an NVIDIA GPU is strongly recommended for speed.
For instructions on Visdom usage/installation, see the Installation section.
You can pick-up training from a checkpoint by specifying the path as one of the training parameters (again, see train.py for options)

