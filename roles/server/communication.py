import os
import sys

dirname = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(dirname))
sys.path.append(os.path.join(dirname, '../../'))

import requests
from tasks.traininig.start_training import start_training_server

def bind_clients_edges(
    edges, 
    clients
):
    # Now, we send a request to each edge to inform it of its assigned clients
    for edge in edges:
        edge_id = edge['id']
        edge_address = edge['address']

        for client_idx, client in enumerate(clients):
            client_id = client['id']
            client_address = client['address']

            if edge['edge_index'] == client['edge_index']:  # If the edge is responsible for the client
                # Construct the URL to inform the edge about its client
                edge_url = f'http://{edge_address}/register_client'

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
                        print(f"Successfully bound client {client_address} to edge {edge_address}")
                    else:
                        print(f"Failed to bind client {client_address} to edge {edge_address}")
                except requests.exceptions.RequestException as e:
                    print(f"Error while trying to bind client {client_address} to edge {edge_address}: {e}")

                # Construct the URL to inform the client about its assigned edge
                client_url = f'http://{client_address}/bind'

                # Send the GET request to the client to inform it of its assigned edge
                try:
                    client_res = requests.post(
                        client_url,
                        json={
                            'server_id': edge_id,
                            'server_address': edge_address,
                            'partition_id': client_idx,
                            'num_clients': len(clients)
                        }
                    )
                    # Check if the request was successful
                    if client_res.status_code == 200:
                        print(f"Successfully informed client {client_address} about edge {edge_address}")
                    else:
                        print(f"Failed to inform client {client_address} about edge {edge_address}")
                except requests.exceptions.RequestException as e:
                    print(f"Error while trying to inform client {client_address} about edge {edge_address}: {e}")

    print("Connections have been made and ready for training")

    # Start first round of training
    # Parameters will be initialized randomly
    start_training_server(
        clients=edges,
        first_round=True
    )

    print('Started first round of training')