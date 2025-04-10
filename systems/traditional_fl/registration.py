import os
import sys

dirname = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(dirname))
sys.path.append(os.path.join(dirname, '../../'))

import requests
from tasks.traininig.start_training import start_training_server

def registration(
    server_id,
    server_address,
    clients
):
    for client_idx, client in enumerate(clients):
        client_id = client['id']
        client_address = client['address']

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

    # Start first round of training
    # Parameters will be initialized randomly
    start_training_server(
        clients=clients,
        first_round=True
    )

    print('Started first round of training')