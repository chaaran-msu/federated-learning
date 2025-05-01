import os
import sys

dirname = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(dirname))

import uuid

from roles.utils import submit, get_current_time
import time

num_devices = 1

architecture = {
    'defaults': {'time': '02:00:00'},
    'clients': [{'mem': 1, 'cpu': 1, 'cluster': 'intel18'} for i in range(num_devices)]
}

def allocate_resources(
    main_server_id,
    main_server_address,
    architecture_name,
    num_edge_client_rounds,
    resources_data,
    checkpoint_times
):
    checkpoint_times['resource_allocation_start'] = get_current_time()

    # Submit jobs to allocate resources
    num_clients = len(architecture['clients'])
    t_time = architecture.get('defaults', {}).get('time', '02:00:00')

    for client_index, client in enumerate(architecture['clients']):
        device_id = uuid.uuid4()

        mem = client.get('mem', 1)
        cpu = client.get('cpu', 1)
        cluster = client.get('cluster', 'intel18')

        resources_data['jobs'].add(
            submit(
                device_id=device_id,
                server_id=main_server_id,
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

        time.sleep(2)

    resources_data['num_devices']['clients'] = num_clients