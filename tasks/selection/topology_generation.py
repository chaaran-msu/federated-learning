import numpy as np

def topology_generation_random(
    main_server_id,
    main_server_address,
    clients
):
    client_ids = [client['id'] for client in clients]

    num_clients = {
        'first_level': 2,
        'second_level': 4
    }

    clients_first_level = np.random.choice(client_ids, num_clients['first_level'], replace=False)
    clients_second_level = np.random.choice(set(client_ids).difference(set(clients_first_level)), num_clients['second_level'], replace=False)
    clients_third_level = set(client_ids).difference(set(clients_first_level).union(set(clients_second_level)))

    client_connections = {}

    num_clients_first_level = len(clients_second_level) // len(clients_first_level)
    num_clients_second_level = len(clients_third_level) // len(clients_second_level)

    for idx, client_id in enumerate(clients_first_level):
        client_connections[client_id] = clients_second_level[idx * num_clients_first_level: min(len(clients_second_level), (idx + 1) * num_clients_first_level)]

    for idx, client_id in enumerate(clients_second_level):
        client_connections[client_id] = clients_third_level[idx * num_clients_second_level: min(len(clients_third_level), (idx + 1) * num_clients_second_level)]

    for client in clients:
        # Add clients of the device
        if client['id'] in client_connections:
            client['clients'] = client_connections[client['id']]

            # Add servers for the devices
            for secondary_client in clients:
                if secondary_client['id'] in client['clients']:
                    secondary_client['server_address'] = client['address']
                    secondary_client['server_id'] = client['id']
        else:
            client['clients'] = []

        if client['id'] in clients_first_level:
            client['server_address'] = main_server_address
            client['server_id'] = main_server_id

    return clients