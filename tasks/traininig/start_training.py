import os
import sys

dirname = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(dirname))
sys.path.append(os.path.join(dirname, '../../'))

import requests

from tasks.traininig.models.baseline import Baseline
from tasks.traininig.utils.parameters import get_serialized_parameters

def start_training_server(
    clients,
    first_round = False,
    parameters = None
):
    if first_round:
        # Initialize model with randomly selected parameters
        model = Baseline()

        # Get serialized parameters
        parameters = get_serialized_parameters(
            model=model
        )

    # Start first round of training
    for client in clients:
        client_address = client['address']

        data = {
            'batch_size': 16,
            'learning_rate': 0.1,
            'num_epochs': 1,
            'parameters': parameters
        }

        # Ask the client to start training
        requests.post(f"http://{client_address}/start_training", json=data)

    print('Sent signal to clients to start training from server')

def start_training_edge(
    round_clients,
    training_data
):
    for client in round_clients:
        client_address = client['address']

        # Ask the client to start training
        requests.post(f"http://{client_address}/start_training", json=training_data)

    print('Sent signal to clients to start training from edge server')