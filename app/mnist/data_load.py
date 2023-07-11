import os

import torch
from torchvision import datasets, transforms

import numpy as np

dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# Concentrating on the first 100 samples
n_samples = 400
batch_size = 4

def get_train_loader():
    train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=True, num_workers=8)
    return train_loader

def get_test_loader():
    test_loader = torch.utils.data.DataLoader(X_test, batch_size=batch_size, shuffle=True, num_workers=8)
    return test_loader

X_train = datasets.MNIST(root=dir_path+'/data', train=True, download=True,
                         transform=transforms.Compose([transforms.ToTensor()]))
X_train.data = X_train.data[:n_samples]
X_train.targets = X_train.targets[:n_samples]
# filter for 4 classes
filtered = torch.logical_or(X_train.targets == 0, X_train.targets == 1)
filtered = torch.logical_or(filtered, X_train.targets == 2)
filtered = torch.logical_or(filtered, X_train.targets == 3)
X_train.data = X_train.data[filtered]
X_train.targets = X_train.targets[filtered]


n_samples = 150
X_test = datasets.MNIST(root=dir_path+'/data', train=False, download=True,
                        transform=transforms.Compose([transforms.ToTensor()]))
X_test.data = X_test.data[:n_samples]
X_test.targets = X_test.targets[:n_samples]

# filter for 4 classes
filtered = torch.logical_or(X_test.targets == 0, X_test.targets == 1)
filtered = torch.logical_or(filtered, X_test.targets == 2)
filtered = torch.logical_or(filtered, X_test.targets == 3)
X_test.data = X_test.data[filtered]
X_test.targets = X_test.targets[filtered]
