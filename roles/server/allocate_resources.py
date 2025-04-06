import os
import sys

dirname = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(dirname))

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