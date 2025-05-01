import os
import sys

dirname = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(dirname))

import uuid

from roles.utils import submit

architecture = {
    'defaults': {'time': '00:05:00'},
    'edges': [
        {
            'mem': 1,
            'cpu': 1,
            'cluster': 'intel18',
            'edges': [
                {
                    'mem': 1,
                    'cpu': 1,
                    'cluster': 'intel18',
                    'clients': [
                        {'mem': 1, 'cpu': 1, 'cluster': 'intel18'},
                        {'mem': 1, 'cpu': 1, 'cluster': 'intel18'},
                        {'mem': 1, 'cpu': 1, 'cluster': 'intel18'},
                        {'mem': 1, 'cpu': 1, 'cluster': 'intel18'},
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
                        {'mem': 1, 'cpu': 1, 'cluster': 'intel18'},
                    ]
                },
            ]  
        },
        {
            'mem': 1,
            'cpu': 1,
            'cluster': 'intel18',
            'edges': [
                {
                    'mem': 1,
                    'cpu': 1,
                    'cluster': 'intel18',
                    'clients': [
                        {'mem': 1, 'cpu': 1, 'cluster': 'intel18'},
                        {'mem': 1, 'cpu': 1, 'cluster': 'intel18'},
                        {'mem': 1, 'cpu': 1, 'cluster': 'intel18'},
                        {'mem': 1, 'cpu': 1, 'cluster': 'intel18'},
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
                        {'mem': 1, 'cpu': 1, 'cluster': 'intel18'},
                    ]
                },
            ]
        }
    ]
}

def allocate_resources(
    main_server_id,
    main_server_address,
    architecture_name,
    num_edge_client_rounds,
    resources_data,
    num_devices
):
    # Submit jobs to allocate resources
    jobs = set()

    num_edges = 0
    num_clients = 0
    t_time = architecture.get('defaults', {}).get('time', '00:05:00')

    for edge_index_fl, edge_fl in enumerate(architecture['edges']):
        edge_id_fl = uuid.uuid4()

        mem = edge_fl.get('mem', 1)
        cpu = edge_fl.get('cpu', 1)
        cluster = edge_fl.get('cluster', 'intel18')

        resources_data['jobs'].add(
            submit(
                device_id=edge_id_fl,
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
        print(f"Job submitted for edge first level {edge_index_fl+1}")

        num_edges += 1

        for edge_index_sl, edge_sl in enumerate(edge_fl['edges']):
            edge_id_sl = uuid.uuid4()

            mem = edge_sl.get('mem', 1)
            cpu = edge_sl.get('cpu', 1)
            cluster = edge_sl.get('cluster', 'intel18')

            resources_data['jobs'].add(
                submit(
                    device_id=edge_id_sl,
                    server_id=edge_id_fl,
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
            print(f"Job submitted for edge second level {edge_index_sl+1}")

            num_edges += 1

            for client_index, client in enumerate(edge_sl['clients']):
                client_id = uuid.uuid4()

                mem = client.get('mem', 1)
                cpu = client.get('cpu', 1)
                cluster = client.get('cluster', 'intel18')

                resources_data['jobs'].add(
                    submit(
                        device_id=client_id,
                        server_id=edge_id_sl,
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

                num_clients += 1

    resources_data['num_devices']['edges'] = num_edges
    resources_data['num_devices']['clients'] = num_clients