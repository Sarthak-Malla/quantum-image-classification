import os

import torch
from torchvision import datasets, transforms

import numpy as np

from config import Config

# declaring hyperparameters and constants
config = Config()

def get_train_loader():
    train_loader = torch.utils.data.DataLoader(X_train, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    return train_loader

def get_test_loader():
    test_loader = torch.utils.data.DataLoader(X_test, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    return test_loader

transform_train = transforms.Compose([transforms.Resize((224, 224)), 
                                      transforms.RandomHorizontalFlip(p=0.7),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.226, 0.225])])
transform_test = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.226, 0.225])])

if config.dataset == "mnist":
    X_train = datasets.MNIST(root=config.dir_path+'/data', train=True, download=True,
                            transform=transforms.Compose([transforms.ToTensor()]))

    X_test = datasets.MNIST(root=config.dir_path+'/data', train=False, download=True,
                            transform=transforms.Compose([transforms.ToTensor()]))
elif config.dataset == "cifar10":
    X_train = datasets.CIFAR10(root=config.dir_path+'/data', train=True, download=True,
                            transform=transform_train)

    X_test = datasets.CIFAR10(root=config.dir_path+'/data', train=False, download=True,
                            transform=transform_test)
