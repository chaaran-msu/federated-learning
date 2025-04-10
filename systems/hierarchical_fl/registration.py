import os
import sys

dirname = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(dirname))
sys.path.append(os.path.join(dirname, '../../'))

import requests
from tasks.traininig.start_training import start_training_server

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
    num_clients
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
                'num_clients': num_clients
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
    main_server_id,
    main_server_address,
    edges, 
    clients,
    current_round_clients
):
    # Inform the edge about the main server
    for edge_id in edges:
        edge_address = edges[edge_id]['address']
        server_id = edges[edge_id]['server_id']

        # Inform the edge server about their server
        if server_id == main_server_id:
            # Construct the URL to inform the edge about its server
            edge_url = f'http://{edge_address}/register_server'

            # Send the POST request to the edge to register the server
            try:
                edge_res = requests.post(
                    edge_url,
                    json={
                        'id': main_server_id,
                        'address': main_server_address
                    }
                )

                # Check if the request was successful
                if edge_res.status_code == 200:
                    print(f"Successfully bound edge {edge_address} to server {main_server_address}")
                else:
                    print(f"Failed to bind edge {edge_address} to server {main_server_address}")
            except requests.exceptions.RequestException as e:
                print(f"Error while trying to bind edge {edge_address} to server {main_server_address}: {e}")


    for client_idx, client_id in enumerate(clients):
        client_address = clients[client_id]['address']
        client_server_id = clients[client_id]['server_id']

        if client_server_id == main_server_id:
            server_address = main_server_address

            # No need to inform the server about the client

            # Inform the client about the server
            inform_client(
                client_address,
                client_server_id,
                server_address,
                client_idx,
                len(clients)
            )
        else:
            server_address = edges[client_server_id]['address']

            # Inform server about client
            inform_server(
                server_address,
                client_id,
                client_address
            )

            # Inform the client about the server
            inform_client(
                client_address,
                client_server_id,
                server_address,
                client_idx,
                len(clients)
            )

    print("Connections have been made and ready for training")

    # Start first round of training
    # Parameters will be initialized randomly
    start_training_server(
        clients=current_round_clients,
        first_round=True
    )

    print('Started first round of training')