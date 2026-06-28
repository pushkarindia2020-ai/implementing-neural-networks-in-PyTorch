# implementing-neural-networks-in-PyTorch
Traning NN on MNIST dataset

These are like "loading tools before you start work." Each library adds specific superpowers.
1
import torch
The heart of everything. torch is PyTorch itself. It gives you tensors (multi-dimensional number arrays), automatic differentiation, and GPU support. Think of it as NumPy but for deep learning.
2
import torch.nn as nn
Neural network building blocks. nn stands for "neural network." It has pre-built layers (Linear, Conv2d), loss functions (CrossEntropyLoss), and the base Module class all models inherit from.
3
import torch.nn.functional as F
Activation functions and more. F gives you functions like F.relu(), F.sigmoid(), F.softmax(). Unlike nn layers, these are stateless functions — they take an input, transform it, and return output with no stored weights.
4
import torch.optim as optim
Optimizers — the "learning" mechanism. optim contains algorithms like SGD, Adam, RMSProp that decide how to update model weights after each backward pass. Without this, the model sees errors but can't fix them.
5
from torchvision import datasets, transforms
Image dataset helpers. torchvision is a separate package (install with pip). datasets gives you MNIST, CIFAR10 etc. already packaged. transforms gives you image preprocessing tools like ToTensor() and Normalize().
6
import matplotlib.pyplot as plt
For drawing images and graphs. plt.imshow() shows an image, plt.plot() draws a line chart. Used at the very end to visually display sample predictions. Not needed for training itself.
7
import numpy as np
Numerical arrays in Python. matplotlib works with NumPy arrays, not PyTorch tensors. So you convert tensors → numpy arrays before plotting. np is used minimally here — just for that conversion.
