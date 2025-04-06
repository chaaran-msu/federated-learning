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
from traininig.start_training import start_training_server

# Initialize model
model = Baseline()

def train_model(
    num_clients: int,
    partition_id: int,
    batch_size: int,
    parameters: List,
    learning_rate: float,
    num_epochs: int,
    logger
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
        serialized_params=parameters
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

    logger.info(f'Test Accuracy: {accuracy}')

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
    num_epochs: int,
    logger
):
    # Train the model
    round_data = train_model(
        num_clients,
        partition_id,
        batch_size,
        parameters,
        learning_rate,
        num_epochs,
        logger
    )

    print('Completed training in client')

    # Send model to server for aggregation
    round_data['id'] = id

    # Send data back to server
    requests.post(
        url=f'http://{server_address}/aggregate',
        json=round_data
    )

    print('Sent to edge server for aggregation from client')

def train_round_edge(
    round_data,
    server_address,
    device_id,
    logger
):
    # Aggregate parameters from all clients
    total_num_samples, aggregated_parameters = aggregate(
        round_data=round_data,
        algorithm='fed_avg',
        logger=logger
    )

    print('Aggregated parameters in edge server')

    data = {
        'id': device_id,
        'num_samples': total_num_samples,
        'parameters': aggregated_parameters
    }

    # Send parameters up the hierarchy        
    requests.post(f"http://{server_address}/aggregate", json=data)

    print('Sent to aggregation from edge server to main server')

def train_round_server(
    round_data,
    round_clients,
    logger
):
    # Aggregate Parameters
    total_num_samples, aggregated_parameters = aggregate(
        round_data=round_data,
        algorithm='fed_avg',
        logger=logger
    )
    print('Aggregated parameters in server')

    # Start Training for next round
    start_training_server(
        clients=round_clients,
        parameters=aggregated_parameters
    )

    print('Started next round from server')