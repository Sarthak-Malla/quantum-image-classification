import os

import torch
from torchvision import datasets, transforms
import torch.optim as optim

import numpy as np

from hybrid_net import HybridNet
from device import device

dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# defining hyperparameters
batch_size = 32
epochs = 50
lr = 1e-4

# defining the training data
n_samples = 5000
transform_test = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.226, 0.225])])

# ----------------------------------------------
X_test = datasets.CIFAR10(root=dir_path + '/data', train=False, download=True,
                            transform=transform_test)

X_test.data = X_test.data[:n_samples]
X_test.targets = X_test.targets[:n_samples]

# ----------------------------------------------
test_loader = torch.utils.data.DataLoader(X_test, batch_size=batch_size, shuffle=True, num_workers=10)

# ----------------------------------------------

# defining the model
model = HybridNet().to(device)

# load model
model.load_state_dict(torch.load(dir_path + "/models/cifar10/ten_class/ten_class_200.pth"))

with torch.no_grad():
    correct = 0
    total = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    print(f"Test accuracy: {100 * correct / total}")