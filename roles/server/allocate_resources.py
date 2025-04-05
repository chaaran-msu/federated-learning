import os
import sys

dirname = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(dirname))

import requests

import config
from roles.utils import submit

def allocate_resources(server_address):
    # Submit jobs to allocate resources
    jobs = set()

    num_edges = len(config.data['edges'])
    num_clients = sum(len(edge['clients']) for edge in config.data['edges'])
    t_time = config.data.get('defaults', {}).get('time', '00:05:00')

    for edge_index, edge in enumerate(config.data['edges']):
        mem = edge.get('mem', 1)
        cpu = edge.get('cpu', 1)
        cluster = edge.get('cluster', 'intel18')

        jobs.add(submit(t_time, mem, cpu, cluster, server_address, 'edge', edge_index))
        print(f"Edge {edge_index} - Mem: {mem}, CPU: {cpu}, Cluster: {cluster}")

        for client_index, client in enumerate(edge['clients']):
            mem = client.get('mem', 1)
            cpu = client.get('cpu', 1)
            cluster = client.get('cluster', 'intel18')

            jobs.add(submit(t_time, mem, cpu, cluster, server_address, 'client', edge_index, client_index))
            print(f"  Client {client_index} - Mem: {mem}, CPU: {cpu}, Cluster: {cluster}")

    return jobs, num_edges, num_clients

def on_ready(edges, clients):
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
