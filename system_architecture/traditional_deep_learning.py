import os
import sys

dirname = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(dirname))
sys.path.append(os.path.join(dirname, '../'))

import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms

from tasks.traininig.models.baseline import Baseline
from tasks.traininig.utils.train import train
from tasks.traininig.utils.test import test

# Dataset Transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# Dataset
batch_size = 16

trainset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True,
    download=True, 
    transform=transform
)

testset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=False,
    download=True, 
    transform=transform
)

# Dataloader
trainloader = torch.utils.data.DataLoader(
    trainset, 
    batch_size=batch_size,
    shuffle=True, 
    num_workers=2
)

testloader = torch.utils.data.DataLoader(
    testset, 
    batch_size=batch_size,
    shuffle=False, 
    num_workers=2
)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
model = Baseline()

# Criterion
criterion = nn.CrossEntropyLoss()

# Optimizer
learning_rate = 0.1

optimizer = torch.optim.SGD(
    params=model.parameters(),
    lr=learning_rate
)

# Training
num_rounds = 100
epoch_losses, epoch_accuracies = train(
    model=model,
    dataloader=trainloader,
    num_epochs=num_rounds,
    criterion=criterion,
    optimizer=optimizer
)

# Testing
batch_losses, accuracy = test(
    model=model,
    dataloader=testloader,
    criterion=criterion
)

print('Training accuracy:', epoch_accuracies[-1])
print('Testing accuracy:', accuracy)