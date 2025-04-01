import time
import config
import socket
import requests
import threading
import subprocess
from utils import *
from flask import Flask, request

app = Flask(__name__)

port = 5001
local = local(port)

jobs = set()
edges = dict()
clients = dict()
num_of_edges = len(config.data['edges'])
num_of_clients = sum(len(edge['clients']) for edge in config.data['edges'])
t_time = config.data.get('defaults', {}).get('time', '00:05:00')

print(f"Time: {t_time}")
print(f"Number of edges: {num_of_edges}")
print(f"Total number of clients: {num_of_clients}\n")


# allocate resources
for edge_index, edge in enumerate(config.data['edges']):
    mem = edge.get('mem', 1)
    cpu = edge.get('cpu', 1)
    cluster = edge.get('cluster', 'intel18')

    jobs.add(submit(t_time, mem, cpu, cluster, local, 'edge', edge_index))
    print(f"Edge {edge_index} - Mem: {mem}, CPU: {cpu}, Cluster: {cluster}")

    for client_index, client in enumerate(edge['clients']):
        mem = client.get('mem', 1)
        cpu = client.get('cpu', 1)
        cluster = client.get('cluster', 'intel18')

        jobs.add(submit(t_time, mem, cpu, cluster, local, 'client', edge_index, client_index))
        print(f"  Client {client_index} - Mem: {mem}, CPU: {cpu}, Cluster: {cluster}")

def on_ready():
    time.sleep(5)
    # Now, we send a request to each edge to inform it of its assigned clients
    for edge_address, edge_id in edges.items():
        for client_address, client_id in clients.items():
            if edge_id == client_id:  # If the edge is responsible for the client
                # Construct the URL to inform the edge about its client
                edge_url = f'http://{edge_address}/bind?address={client_address}'

                # Send the GET request to the edge to bind the client
                try:
                    edge_res = requests.get(edge_url)
                    # Check if the request was successful
                    if edge_res.status_code == 200:
                        print(f"Successfully bound client {client_address} to edge {edge_address}")
                    else:
                        print(f"Failed to bind client {client_address} to edge {edge_address}")
                except requests.exceptions.RequestException as e:
                    print(f"Error while trying to bind client {client_address} to edge {edge_address}: {e}")

                # Construct the URL to inform the client about its assigned edge
                client_url = f'http://{client_address}/bind?address={edge_address}'

                # Send the GET request to the client to inform it of its assigned edge
                try:
                    client_res = requests.get(client_url)
                    # Check if the request was successful
                    if client_res.status_code == 200:
                        print(f"Successfully informed client {client_address} about edge {edge_address}")
                    else:
                        print(f"Failed to inform client {client_address} about edge {edge_address}")
                except requests.exceptions.RequestException as e:
                    print(f"Error while trying to inform client {client_address} about edge {edge_address}: {e}")


@app.route('/')
def home():
    return 'OK'

@app.route('/register')
def register():
    role = request.args.get('role', None)
    edge_index = request.args.get('edge_index', None)
    address = request.args.get('address', None)

    if role == 'edge':
        edges[address] = edge_index
    elif role == 'client':
        clients[address] = edge_index

    if len(edges) == num_of_edges and len(clients) == num_of_clients:
        threading.Thread(target=on_ready, daemon=True).start()
        print("Ready!")

    return 'OK'

@app.route('/kill')
def kill():
    for id in jobs:
        subprocess.run(['scancel', id])
    return f"Successfully canceled jobs"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=False)
