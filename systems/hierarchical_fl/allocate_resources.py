import os
import sys

dirname = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(dirname))

import uuid

from roles.utils import submit

architecture = {
    'defaults': {'time': '02:00:00'},
    'edges': [
        {
            'mem': 1,
            'cpu': 1,
            'cluster': 'intel18',
            'clients': [
                {'mem': 1, 'cpu': 1, 'cluster': 'intel18'},
                {'mem': 1, 'cpu': 1, 'cluster': 'intel18'},
                {'mem': 1, 'cpu': 1, 'cluster': 'intel18'},
                {'mem': 1, 'cpu': 1, 'cluster': 'intel18'}
            ]
        },
        {
            'mem': 1,
            'cpu': 1,
            'cluster': 'intel18',
            'clients': [
                {'mem': 1, 'cpu': 1, 'cluster': 'intel18'},
                {'mem': 1, 'cpu': 1, 'cluster': 'intel18'},
                {'mem': 1, 'cpu': 1, 'cluster': 'intel18'},
                {'mem': 1, 'cpu': 1, 'cluster': 'intel18'}
            ]
        }
    ]
}

def allocate_resources(
    main_server_id,
    main_server_address,
    architecture_name,
    num_edge_client_rounds,
    resources_data
):
    # Submit jobs to allocate resources
    num_edges = len(architecture['edges'])
    num_clients = sum(len(edge['clients']) for edge in architecture['edges'])
    t_time = architecture.get('defaults', {}).get('time', '02:00:00')

    for edge_index, edge in enumerate(architecture['edges']):
        edge_id = uuid.uuid4()

        mem = edge.get('mem', 1)
        cpu = edge.get('cpu', 1)
        cluster = edge.get('cluster', 'intel18')

        resources_data['jobs'].add(
            submit(
                device_id=edge_id,
                server_id=main_server_id,
                main_server_address=main_server_address,
                role='edge',
                time=t_time, 
                mem=mem, 
                cpu=cpu, 
                cluster=cluster,
                architecture=architecture_name,
                num_edge_client_rounds=num_edge_client_rounds
            )
        )
        print(f"Job submitted for edge {edge_index+1}")

        for client_index, client in enumerate(edge['clients']):
            client_id = uuid.uuid4()

            mem = client.get('mem', 1)
            cpu = client.get('cpu', 1)
            cluster = client.get('cluster', 'intel18')

            resources_data['jobs'].add(
                submit(
                    device_id=client_id,
                    server_id=edge_id,
                    main_server_address=main_server_address,
                    role='client',
                    time=t_time, 
                    mem=mem, 
                    cpu=cpu, 
                    cluster=cluster,
                    architecture=architecture_name,
                    num_edge_client_rounds=num_edge_client_rounds
                )
            )

            print(f"Job submitted for client {client_index+1}")

    resources_data['num_devices']['edges'] = num_edges
    resources_data['num_devices']['clients'] = num_clients