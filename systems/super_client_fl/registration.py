import os
import sys

dirname = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(dirname))
sys.path.append(os.path.join(dirname, '../../'))

import requests
from collections import defaultdict
from typing import List, Dict

from tasks.training.start_training import start_training_server
from roles.utils import get_current_time

def inform_server(
    server_address,
    client_id,
    client_address
):
    # Construct the URL to inform the edge about its client
    edge_url = f'http://{server_address}/register_client'

    # Send the POST request to the edge to register the client
    try:
        edge_res = requests.post(
            edge_url,
            json={
                'id': client_id,
                'address': client_address
            }
        )

        # Check if the request was successful
        if edge_res.status_code == 200:
            print(f"Successfully bound client {client_address} to edge {server_address}")
        else:
            print(f"Failed to bind client {client_address} to edge {server_address}")
    except requests.exceptions.RequestException as e:
        print(f"Error while trying to bind client {client_address} to edge {server_address}: {e}")    


def inform_client(
    client_address,
    server_id,
    server_address,
    partition_id,
    num_clients,
    role
):
    client_url = f'http://{client_address}/register_server'

    # Send the POST request to the client to inform it of its assigned edge
    try:
        client_res = requests.post(
            client_url,
            json={
                'server_id': server_id,
                'server_address': server_address,
                'partition_id': partition_id,
                'num_clients': num_clients,
                'role': role
            }
        )
        # Check if the request was successful
        if client_res.status_code == 200:
            print(f"Successfully informed client {client_address} about edge {server_address}")
        else:
            print(f"Failed to inform client {client_address} about edge {server_address}")
    except requests.exceptions.RequestException as e:
        print(f"Error while trying to inform client {client_address} about edge {server_address}: {e}")

def registration(
    main_server_id: str,
    client_topologies: List[Dict],
    round_num: int,
    registration_times: Dict,
    checkpoint_times: Dict,
    data_store,
    first_round = False
):
    '''
        Arguments:
            main_server_id (str): Clients for the main server.
            client_topologies (List[Dict]): Each client will have it's address, server's address and a list of clients.
    
    '''
    registration_start_time = get_current_time()

    current_round_clients = []

    print(client_topologies)

    for idx, client in enumerate(client_topologies):
        if client['server_id'] != main_server_id:
            # Inform the server about this client
            inform_server(
                server_address=client['server_address'],
                client_id=client['id'],
                client_address=client['address'],
            )
        else:
            current_round_clients.append(client)

        # Inform the client about the server
        inform_client(
            client_address=client['address'],
            server_id=client['server_id'],
            server_address=client['server_address'],
            partition_id=idx,
            num_clients=len(client_topologies),
            role='super_client' if len(client['clients']) > 0 else 'client'
        )

    print("Connections have been made and ready for training")

    registration_times[round_num] = get_current_time() - registration_start_time

    data_store['current_round_clients'] = current_round_clients

    # Start first round of training
    # Parameters will be initialized randomly
    if first_round:
        checkpoint_times['training_start'] = get_current_time()
        
        start_training_server(
            clients=current_round_clients,
            first_round=True
        )

        print('Started first round of training')
    else:
        return current_round_clients