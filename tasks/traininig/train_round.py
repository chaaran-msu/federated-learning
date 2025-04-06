import os
import sys

dirname = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(dirname, '../'))

import torch
from torch import nn
from typing import List
import requests

from traininig.data.create_dataloaders import create_dataloaders_flower
from traininig.models.baseline import Baseline
from traininig.utils.parameters import load_serialized_parameters, get_serialized_parameters
from traininig.utils.train import train
from traininig.utils.test import test
from aggregation.aggregate import aggregate

# Initialize model
model = Baseline()

def train_model(
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

def train_round_client(
    id: str,
    server_address: str,
    num_clients: int,
    partition_id: int,
    batch_size: int,
    parameters: List,
    learning_rate: float,
    num_epochs: int
):
    # Train the model
    round_data = train_model(
        num_clients,
        partition_id,
        batch_size,
        parameters,
        learning_rate,
        num_epochs
    )

    # Send model to server for aggregation
    round_data['id'] = id

    # Send data back to server
    requests.post(
        url=f'{server_address}/aggregate',
        json=round_data
    )

def train_round_edge(
    current_round_data,
    server_address,
    device_id
):
    # Aggregate parameters from all clients
    total_num_samples, aggregated_parameters = aggregate(current_round_data)

    data = {
        'id': device_id,
        'num_samples': total_num_samples,
        'parameters': aggregated_parameters
    }

    # Send parameters up the hierarchy        
    requests.post(f"{server_address}/aggregate", json=data)

def train_round_server(
    round_data,
    round_clients
):
    # Aggregate Parameters
    total_num_samples, aggregated_parameters = aggregate(round_data)

    # Start Training for next round
    for client in round_clients:
        client_address = client['address']

        data = {
            'batch_size': 16,
            'learning_rate': 0.1,
            'num_epochs': 1,
            'parameters': aggregated_parameters
        }

        # Ask the client to start training
        requests.post(f"{client_address}/start_training", json=data)