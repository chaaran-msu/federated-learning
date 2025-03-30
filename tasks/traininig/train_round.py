import torch
from torch import nn
from typing import List

from traininig.data.create_dataloaders import create_dataloaders_flower
from traininig.models.baseline import Baseline
from tasks.traininig.utils.parameters import load_serialized_parameters, get_serialized_parameters
from traininig.utils.train import train
from traininig.utils.test import test

def train_round(
    num_clients: int,
    partition_id: int,
    batch_size: int,
    parameters: List,
    learning_rate: float,
    num_epochs: int
):
    # Set up the dataloaders
    trainloader, valloader = create_dataloaders_flower(
        num_clients=num_clients,
        partition_id=partition_id,
        batch_size=batch_size
    )

    # Initialize model
    model = Baseline()

    # Load model parameters
    load_serialized_parameters(
        model=model,
        parameters=parameters
    )

    # Criterion
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=learning_rate
    )

    # Training
    epoch_losses, epoch_accuracies = train(
        model=model,
        dataloader=trainloader,
        num_epochs=num_epochs,
        criterion=criterion,
        optimizer=optimizer
    )

    # Testing
    batch_losses, accuracy = test(
        model=model,
        dataloader=valloader,
        criterion=criterion
    )

    # Set model parameters
    load_serialized_parameters(
        model=model,
        parameters=parameters
    )

    # Serialize model parameters
    serialized_params = get_serialized_parameters(model)

    round_data = {
        'parameters': serialized_params,
        'accuracy': accuracy,
        'num_samples': len(trainloader.dataset)
    }

    return round_data