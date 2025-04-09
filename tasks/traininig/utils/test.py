import os
import sys

dirname = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(dirname, '../'))

import torch
from torch import nn
import torch.utils.data

from metrics.accuracy import compute_accuracy
 
def test(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    traditional: bool = False,
    device: str = 'cpu' 
):
    model.eval()

    batch_losses = []

    total_num_correct = 0
    total_count = 0

    with torch.no_grad():
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

            num_correct, accuracy = compute_accuracy(
                ground_truths=labels.detach().cpu().numpy(),
                predictions=torch.argmax(outputs, dim=-1).detach().cpu().numpy()
            )

            batch_losses.append(loss)

            total_num_correct += num_correct
            total_count += labels.shape[0]

    return batch_losses, total_num_correct / total_count
