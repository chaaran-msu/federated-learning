import os
import sys
import psutil
import threading

dirname = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(dirname, '../'))
sys.path.append(os.path.join(dirname, '../../'))

import torch
from torch import nn
from typing import List
import requests
import time

from roles.utils import get_current_time, monitor_cpu_usage

from training.data.create_dataloaders import create_dataloaders_flower, create_dataloaders_flower_multiple_partitions, create_global_test_dataloader
from training.models.baseline import Baseline
from training.utils.parameters import load_serialized_parameters, get_serialized_parameters
from training.utils.train import train
from training.utils.test import test
from aggregation.aggregate import aggregate
from training.start_training import start_training_server, start_training_edge

from tasks.selection.topology_generation import topology_generation_random
from systems.super_client_fl.registration import registration

# Initialize model
model = Baseline()

def train_model(
    num_clients: int,
    partition_id: int,
    batch_size: int,
    parameters: List,
    learning_rate: float,
    num_epochs: int,
    logger,
    round_data,
    computational_latency
):
    train_start_time = time.time()

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

    train_end_time = time.time()

    test_start_time = time.time()

    # Testing
    batch_losses, accuracy = test(
        model=model,
        dataloader=valloader,
        criterion=criterion
    )

    logger.info(f'Test Accuracy: {accuracy}')

    test_end_time = time.time()

    # Serialize model parameters
    serialized_params = get_serialized_parameters(model)

    round_data['parameters'] = serialized_params
    round_data['accuracy'] = accuracy
    round_data['num_samples'] = len(trainloader.dataset)

    computational_latency['training_time'] = train_end_time - train_start_time
    computational_latency['testing_time'] = test_end_time - test_start_time

def test_model(
    num_clients,
    parameters,
    partition_ids,
    batch_size,
    mode,
    logger
):
    if mode == 'edge' or mode == 'server':
        # Set up the dataloaders
        trainloader, test_loader = create_dataloaders_flower_multiple_partitions(
            num_clients=num_clients,
            partition_ids=partition_ids,
            batch_size=batch_size
        )
    elif mode == 'server_global':
        test_loader = create_global_test_dataloader(
            num_clients=num_clients,
            batch_size=batch_size
        )

    # Load model parameters
    load_serialized_parameters(
        model=model,
        serialized_params=parameters
    )

    # Criterion
    criterion = nn.CrossEntropyLoss()

    # Testing
    batch_losses, accuracy = test(
        model=model,
        dataloader=test_loader,
        criterion=criterion
    )

    logger.info(f'Test Accuracy: {accuracy}')

    round_data = {
        'accuracy': accuracy,
    }

    return round_data

def get_complete_cpu_usage_history(resource_utilization, current_round_utilizations, training_data):
    # Round utilization
    resource_utilization['avg_cpu_percent_round'] = current_round_utilizations['avg_cpu_percent']
    resource_utilization['peak_cpu_percent_round'] = current_round_utilizations['peak_cpu_percent']
    resource_utilization['avg_mem_bytes_round'] = current_round_utilizations['avg_mem_bytes']
    resource_utilization['peak_mem_bytes_round'] = current_round_utilizations['peak_mem_bytes']

    # overall Utilization
    resource_utilization['avg_cpu_percent'] = (resource_utilization['avg_cpu_percent'] * training_data['round'] + current_round_utilizations['avg_cpu_percent']) / (training_data['round'] + 1)
    resource_utilization['avg_mem_bytes'] = (resource_utilization['avg_mem_bytes'] * training_data['round'] + current_round_utilizations['avg_mem_bytes']) / (training_data['round'] + 1)
    resource_utilization['peak_cpu_percent'] = max(resource_utilization['peak_cpu_percent'], current_round_utilizations['peak_cpu_percent'])
    resource_utilization['peak_mem_bytes'] = max(resource_utilization['peak_mem_bytes'], current_round_utilizations['peak_mem_bytes'])

def train_round_client(
    id: str,
    server_address: str,
    num_clients: int,
    partition_id: int,
    logger,
    training_data: dict,
    results_file_path: str,
    resource_utilization_store: dict,
    send_to_server = True
):
    round_data = {}
    computational_latency= {}
    current_round_resource_utilization = {}

    process = psutil.Process(os.getpid())

    training_thread = threading.Thread(
        target=train_model,
        args=(num_clients, partition_id, training_data['batch_size'], training_data['parameters'], training_data['learning_rate'], training_data['num_epochs'], logger, round_data, computational_latency)
    )

    stop_monitor_event = threading.Event()

    cpu_monitor_thread = threading.Thread(
        target=monitor_cpu_usage,
        args=(process, stop_monitor_event, current_round_resource_utilization)
    )

    training_thread.start()
    cpu_monitor_thread.start()

    training_thread.join()

    # Stop monitoring
    stop_monitor_event.set()
    cpu_monitor_thread.join()

    # # Train the model
    # round_data, computational_latency = train_model(
    #     num_clients,
    #     partition_id,
    #     batch_size,
    #     parameters,
    #     learning_rate,
    #     num_epochs,
    #     logger
    # )

    round = training_data['round']
    training_time = computational_latency["training_time"]

    get_complete_cpu_usage_history(resource_utilization_store, current_round_resource_utilization, training_data)

    resource_utilization_store['avg_training_time'] = (resource_utilization_store['avg_training_time'] * round + training_time) / (round + 1)

    # Update round number
    training_data['round'] += 1

    # Save results in text file for analysis
    with open(results_file_path, 'a') as file:
        file.write(f'{training_data["round"]},{round_data["accuracy"]},{computational_latency["training_time"]},{computational_latency["testing_time"]}' + '\n')

    logger.info('Completed training in client')

    # Send model to server for aggregation
    round_data['id'] = id
    round_data['round'] = training_data['round']
    round_data['timestamp'] = get_current_time()
    round_data['resource_utilization'] = resource_utilization_store

    if send_to_server:
        # Send data back to server
        requests.post(
            url=f'http://{server_address}/aggregate',
            json=round_data
        )

        logger.info('Sent to edge server for aggregation from client')
    else:
        return round_data

def train_round_edge(
    local_address,
    num_edge_client_rounds,
    round_clients,
    round_data,
    server_address,
    device_id,
    num_clients,
    partition_ids,
    batch_size,
    learning_rate,
    num_epochs,
    round_num,
    logger,
    results_file_path
):
    aggregation_start_time = time.time()

    # Aggregate parameters from all clients
    total_num_samples, aggregated_parameters = aggregate(
        round_data=round_data,
        algorithm='fed_avg',
        logger=logger
    )

    aggregation_end_time = time.time()

    logger.info(f'Aggregated parameters in edge server using {total_num_samples} samples')

    data = {
        'id': device_id,
        'num_samples': total_num_samples,
        'parameters': aggregated_parameters,
        'round': round_num,
    }

    if round_num == 30:
        test_start_time = time.time()

        test_data = test_model(
            num_clients,
            aggregated_parameters,
            partition_ids,
            batch_size,
            "edge",
            logger
        )

        test_end_time = time.time()

        computational_latency = {
            'aggregation_time': aggregation_end_time - aggregation_start_time,
            'testing_time': test_end_time - test_start_time
        }

        # Save results in text file for analysis
        with open(results_file_path, 'a') as file:
            file.write(f'{round_num},{test_data["accuracy"]},{computational_latency["aggregation_time"]},{computational_latency["testing_time"]}' + '\n')

    # Reset current training round data
    requests.post(f"http://{local_address}/reset")
    logger.info('Reset previous round data')

    if round_num % num_edge_client_rounds == 0:
        data['timestamp'] = get_current_time()

        # Send parameters up the hierarchy
        requests.post(f"http://{server_address}/aggregate", json=data)

        logger.info('Sent to aggregation from edge server to main server')
    else:
        # Start next round of training in clients
        training_data = {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'parameters': aggregated_parameters
        }

        start_training_edge(
            round_clients=round_clients,
            training_data=training_data,
            logger=logger
        )

        logger.info('Started next round of training')

def train_round_super_client(
    local_address,
    num_edge_client_rounds,
    round_clients,
    round_data,
    server_address,
    device_id,
    num_clients,
    batch_size,
    learning_rate,
    num_epochs,
    round_num,
    logger,
    results_file_path
):
    aggregation_start_time = time.time()

    # Aggregate parameters from all clients
    total_num_samples, aggregated_parameters = aggregate(
        round_data=round_data,
        algorithm='fed_avg',
        logger=logger
    )

    aggregation_end_time = time.time()

    logger.info(f'Aggregated parameters in edge server using {total_num_samples} samples')

    data = {
        'id': device_id,
        'num_samples': total_num_samples,
        'parameters': aggregated_parameters,
        'round': round_num,
    }

    if round_num == 30:
        computational_latency = {
            'aggregation_time': aggregation_end_time - aggregation_start_time
        }
        
        # Save results in text file for analysis
        with open(results_file_path, 'a') as file:
            file.write(f'{round_num},{computational_latency["aggregation_time"]},{computational_latency["testing_time"]}' + '\n')

    if round_num % num_edge_client_rounds == 0:
        # Reset both clients and current training round data to get ready for the next topology
        requests.post(f"http://{local_address}/reset_clients")
        requests.post(f"http://{local_address}/reset_clients_data")
        logger.info('Reset clients and previous round data')

        data['timestamp'] = get_current_time()

        # Send parameters up the hierarchy
        requests.post(f"http://{server_address}/aggregate", json=data)

        logger.info('Sent to aggregation from edge server to main server')
    else:
        # Reset only current training round data to get ready for the aggregation
        requests.post(f"http://{local_address}/reset_clients_data")
        logger.info('Reset previous round data')

        # Start next round of training in clients
        training_data = {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'parameters': aggregated_parameters
        }

        start_training_edge(
            round_clients=round_clients,
            training_data=training_data,
            logger=logger
        )

        logger.info('Started next round of training')

def train_round_server(
    data_store,
    main_server_id,
    main_server_address,
    num_rounds,
    round_data,
    all_clients,
    round_clients,
    num_clients,
    partition_ids,
    batch_size,
    round_num,
    logger,
    results_file_path,
    checkpoint_times,
    do_client_selection = False,
    registration_times = {}
):
    aggregation_start_time = time.time()

    # Aggregate Parameters
    total_num_samples, aggregated_parameters = aggregate(
        round_data=round_data,
        algorithm='fed_avg',
        logger=logger
    )

    aggregation_end_time = time.time()

    print(f'Aggregated parameters in server using {total_num_samples} samples')

    # Reset current training round data
    requests.post(f"http://{main_server_address}/reset")

    if round_num == num_rounds:
        test_start_time = time.time()

        test_data = test_model(
            num_clients,
            aggregated_parameters,
            partition_ids,
            batch_size,
            "server",
            logger
        )

        test_end_time = time.time()

        test_global_start_time = time.time()

        test_data_global = test_model(
            num_clients,
            aggregated_parameters,
            [0],
            batch_size,
            "server_global",
            logger
        )

        test_global_end_time = time.time()

        computational_latency = {
            'aggregation_time': aggregation_end_time - aggregation_start_time,
            'testing_time': test_end_time - test_start_time,
            'global_testing_time': test_global_end_time - test_global_start_time
        }

        # Save results in text file for analysis
        with open(results_file_path, 'a') as file:
            file.write(f'{round_num},{test_data["accuracy"]},{test_data_global["accuracy"]},{computational_latency["aggregation_time"]},{computational_latency["testing_time"]},{computational_latency["global_testing_time"]}' + '\n')

    # If current round is less than num_rounds, start next round
    if round_num < num_rounds:
        # Client and Topology Selection for next round
        if do_client_selection:
            client_topologies = topology_generation_random(
                main_server_address=main_server_address,
                main_server_id=main_server_id,
                clients=all_clients
            )

            round_clients = registration(
                main_server_id=main_server_id,
                client_topologies=client_topologies,
                round_num=round_num,
                registration_times=registration_times,
                checkpoint_times=checkpoint_times,
                data_store=data_store,
                first_round=False, 
            )
        
        # Start Training for next round
        start_training_server(
            clients=round_clients,
            parameters=aggregated_parameters
        )

        logger.info('Started next round from server')
    # Else, stop training
    else:
        checkpoint_times['training_end'] = get_current_time()
        
        requests.post(f"http://{main_server_address}/stop")

        logger.info('Finsihed training')
