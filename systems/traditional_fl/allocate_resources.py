import os
import sys

dirname = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(dirname))

import uuid

from roles.utils import submit

architecture = {
    'defaults': {'time': '00:05:00'},
    'clients': [
        {'mem': 1, 'cpu': 1, 'cluster': 'intel18'},
        {'mem': 1, 'cpu': 1, 'cluster': 'intel18'},
        {'mem': 1, 'cpu': 1, 'cluster': 'intel18'},
        {'mem': 1, 'cpu': 1, 'cluster': 'intel18'},
    ]
}

def allocate_resources(
    main_server_id,
    main_server_address,
    architecture_name
):
    # Submit jobs to allocate resources
    jobs = set()

    num_clients = len(architecture['clients'])
    t_time = architecture.get('defaults', {}).get('time', '00:05:00')

    for client_index, client in enumerate(architecture['clients']):
        device_id = uuid.uuid4()

        mem = client.get('mem', 1)
        cpu = client.get('cpu', 1)
        cluster = client.get('cluster', 'intel18')

        jobs.add(
            submit(
                device_id=device_id,
                server_id=main_server_id,
                main_server_address=main_server_address,
                role='client',
                time=t_time, 
                mem=mem, 
                cpu=cpu, 
                cluster=cluster,
                architecture=architecture_name
            )
        )

        print(f"Job submitted for client {client_index+1}")

    num_devices = {
        'clients': num_clients
    }

    return jobs, num_devices