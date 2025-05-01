import numpy as np
import math

def topology_generation_random(
    main_server_id,
    main_server_address,
    clients
):
    clients_dict = {}

    for client in clients:
        clients_dict[client['id']] = client

    client_ids = list(clients_dict.keys())

    num_clients = {
        'first_level': 1,
        'second_level': 1
    }

    clients_first_level = list(np.random.choice(client_ids, num_clients['first_level'], replace=False))
    clients_second_level = list(np.random.choice(list(set(client_ids).difference(set(clients_first_level))), num_clients['second_level'], replace=False))
    clients_third_level = list(set(client_ids).difference(set(clients_first_level).union(set(clients_second_level))))

    client_connections = {}

    num_clients_first_level = math.ceil(len(clients_second_level) / len(clients_first_level))
    num_clients_second_level = math.ceil(len(clients_third_level) / len(clients_second_level))

    for idx, client_id in enumerate(clients_first_level):
        client_connections[client_id] = clients_second_level[idx * num_clients_first_level: min(len(clients_second_level), (idx + 1) * num_clients_first_level)]

    for idx, client_id in enumerate(clients_second_level):
        client_connections[client_id] = clients_third_level[idx * num_clients_second_level: min(len(clients_third_level), (idx + 1) * num_clients_second_level)]

    for client in clients:
        # Add clients of the device
        if client['id'] in client_connections:
            # Make the current device their server
            for client_id in client_connections[client['id']]:
                clients_dict[client_id]['server_address'] = client['address']
                clients_dict[client_id]['server_id'] = client['id']

            # Add them to the clients of this device
            client['clients'] = [clients_dict[client_id] for client_id in client_connections[client['id']]]
        else:
            client['clients'] = []

        if client['id'] in clients_first_level:
            client['server_address'] = main_server_address
            client['server_id'] = main_server_id

    return clients