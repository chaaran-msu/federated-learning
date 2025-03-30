import config
import socket
import subprocess
from utils import *
from flask import Flask, request

app = Flask(__name__)

port = 5000
local = local(port)

jobs = set()
edges = dict()
clients = dict()
num_of_edges = len(config.data['edges'])
num_of_clients = sum(len(edge['clients']) for edge in config.data['edges'])
time = config.data.get('defaults', {}).get('time', '00:05:00')

print(f"Time: {time}")
print(f"Number of edges: {num_of_edges}")
print(f"Total number of clients: {num_of_clients}\n")


# allocate resources
for edge_index, edge in enumerate(config.data['edges']):
    mem = edge.get('mem', 1)
    cpu = edge.get('cpu', 1)
    cluster = edge.get('cluster', 'intel18')

    jobs.add(submit(time, mem, cpu, cluster, local, 'edge', edge_index))
    print(f"Edge {edge_index} - Mem: {mem}, CPU: {cpu}, Cluster: {cluster}")

    for client_index, client in enumerate(edge['clients']):
        mem = client.get('mem', 1)
        cpu = client.get('cpu', 1)
        cluster = client.get('cluster', 'intel18')

        jobs.add(submit(time, mem, cpu, cluster, local, 'client', edge_index, client_index))
        print(f"  Client {client_index} - Mem: {mem}, CPU: {cpu}, Cluster: {cluster}")

print(local)
print(jobs)

@app.route('/')
def home():
    return 'OK'

@app.route('/register')
def register():
    role = request.args.get('role', None)
    edge_index = request.args.get('edge_index', None)
    address = request.args.get('address', None)

    print(role)
    print(edge_index)
    print(address)
    if role == 'edge':
        edges[address] = edge_index
    else:
        clients[address] = edge_index

    if len(edges) == num_of_edges and len(clients) == num_of_clients:
        print("Ready!")

    return 'OK'

@app.route('/kill')
def kill():
    ids = ' '.join(map(str, jobs))
    subprocess.run(['sbatch', ids])
    return f"Successfully canceled jobs: {ids}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=False)
