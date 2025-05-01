import os
import sys

dirname = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(dirname, '../'))

import torch
from torch import nn
import torch.utils.data
import numpy as np

from metrics.accuracy import compute_accuracy

def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    traditional: bool = False,
    device: str = 'cpu' 
):
    model.train()

    batch_losses = []

    epoch_num_correct = 0
    epoch_count = 0

    # Iterate through batch
    for idx, batch in enumerate(dataloader):
        if traditional:
            image = batch[0]
            labels = batch[1]
        else:
            image = batch['img']
            labels = batch['label']

        image = image.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(image)

        #  Compte loss
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute metrics
        num_correct, accuracy = compute_accuracy(
            ground_truths=labels.detach().cpu().numpy(),
            predictions=torch.argmax(outputs, dim=-1).detach().cpu().numpy()
        )

        epoch_num_correct += num_correct
        epoch_count += labels.shape[0]        

        batch_losses.append(loss.detach().cpu().item())

    epoch_accuracy = epoch_num_correct/epoch_count 

    return batch_losses, np.mean(batch_losses), epoch_accuracy

def train(
    model,
    dataloader,
    num_epochs,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    traditional: bool = False,
    device: str = 'cpu' 
):
    all_batch_losses = []
    epoch_losses = []
    epoch_accuracies = []

    for i in range(num_epochs):
        batch_losses, epoch_loss, epoch_accuracy = train_epoch(model, dataloader, criterion, optimizer, traditional, device)

        print(f'{i+1}/{num_epochs} - Accuracy: {epoch_accuracy * 100}, Loss: {epoch_loss}')

        all_batch_losses.append(batch_losses)
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_accuracy)

    return epoch_losses, epoch_accuracies