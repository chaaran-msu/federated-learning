import os
import sys

dirname = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(dirname))
sys.path.append(os.path.join(dirname, '../../'))

import requests
from tasks.training.start_training import start_training_server
from roles.utils import get_current_time

def registration(
    server_id,
    server_address,
    clients,
    current_round_clients,
    checkpoint_times
):
    for client_idx, client_id in enumerate(clients):
        client_address = clients[client_id]['address']
        client_server_id = clients[client_id]['server_id']

        client_url = f'http://{client_address}/register_server'

        # Send the POST request to the client to inform it about the main server
        try:
            client_res = requests.post(
                client_url,
                json={
                    'server_id': server_id,
                    'server_address': server_address,
                    'partition_id': client_idx,
                    'num_clients': len(clients)
                }
            )
            # Check if the request was successful
            if client_res.status_code == 200:
                print(f"Successfully informed client {client_address} about server")
            else:
                print(f"Failed to inform client {client_address} about server")
        except requests.exceptions.RequestException as e:
            print(f"Error while trying to inform client {client_address} about server: {e}")

    print("Connections have been made and ready for training")

    checkpoint_times['resource_allocation_end'] = get_current_time()
    checkpoint_times['training_start'] = get_current_time()

    # Start first round of training
    # Parameters will be initialized randomly
    start_training_server(
        clients=current_round_clients,
        first_round=True
    )

    print('Started first round of training')