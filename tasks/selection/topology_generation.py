import numpy as np
import math

def topology_generation_random(
    main_server_id,
    main_server_address,
    clients
):
    client_ids = list(clients.keys())

    num_clients = {
        'first_level': 2,
        'second_level': 2
    }

    clients_first_level = list(np.random.choice(client_ids, num_clients['first_level'], replace=False))
    clients_second_level = list(np.random.choice(list(set(client_ids).difference(set(clients_first_level))), num_clients['second_level'], replace=False))
    clients_third_level = list(set(client_ids).difference(set(clients_first_level).union(set(clients_second_level))))

    servers = {}

    num_clients_first_level = math.ceil(len(clients_second_level) / len(clients_first_level))
    num_clients_second_level = math.ceil(len(clients_third_level) / len(clients_second_level))

    for client_id in clients_first_level:
        servers[client_id] = {
            'server_address': main_server_address,
            'server_id': main_server_id
        }

    client_connections = {}

    for idx, client_id in enumerate(clients_first_level):
        current_server_clients = clients_second_level[idx * num_clients_first_level: min(len(clients_second_level), (idx + 1) * num_clients_first_level)]
        client_connections[client_id] = [clients[client_id] for client_id in current_server_clients]

        for server_client_id in current_server_clients:
            servers[server_client_id] = {
                'server_address': clients[client_id]['address'],
                'server_id': clients[client_id]['id']
            }

    for idx, client_id in enumerate(clients_second_level):
        current_server_clients = clients_third_level[idx * num_clients_second_level: min(len(clients_third_level), (idx + 1) * num_clients_second_level)]
        client_connections[client_id] = [clients[client_id] for client_id in current_server_clients]

        for server_client_id in current_server_clients:
            servers[server_client_id] = {
                'server_address': clients[client_id]['address'],
                'server_id': clients[client_id]['id']
            }

    client_topologies = []

    for client_id in clients:
        client = clients[client_id]
        client_topologies.append(
            {
                'id': client['id'],
                'address': client['address'],
                'clients': client_connections[client_id] if client_id in client_connections else [],
                'server_address': servers[client_id]['server_address'],
                'server_id': servers[client_id]['server_id']
            }
        )

    return client_topologies