import os
import sys

dirname = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(dirname))
sys.path.append(os.path.join(dirname, '../'))

import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms

from tasks.training.models.baseline import Baseline
from tasks.training.utils.train import train
from tasks.training.utils.test import test

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
)

testloader = torch.utils.data.DataLoader(
    testset, 
    batch_size=batch_size,
    shuffle=False, 
)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
model = Baseline()
model.to(device)

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
    optimizer=optimizer, 
    traditional=True,
    device=device
)

# Testing
batch_losses, accuracy = test(
    model=model,
    dataloader=testloader,
    criterion=criterion,
    traditional=True,
    device=device
)

print('Training accuracy:', epoch_accuracies[-1])
print('Testing accuracy:', accuracy)