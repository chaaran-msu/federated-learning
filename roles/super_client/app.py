import os
import sys

dirname = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(dirname, '../../'))

from flask import Flask, request, jsonify
import requests
import threading
import logging

from roles.utils import get_local_port, get_local_address
from roles.logging import get_logger
from tasks.training.train_round import train_round_super_client, train_round_client
from tasks.training.start_training import start_training_edge

port = get_local_port()
local_address = get_local_address(port)

device_id = sys.argv[1]
server_id = sys.argv[2]
main_server_address = sys.argv[3]
architecture = sys.argv[4]
num_edge_client_rounds = int(sys.argv[5])

# Sample data storage (in-memory)
data_store = {
    'id': device_id,
    'server_id': '',
    'server_address': '',
    'partition_id': '',
    'round_num': 0,
    'clients': [], # Client Index in 'clients' list
    'clients_data': {}, # Parameters for current round,
}

def register_with_server():
    # Register with the main server
    data = {
        'id': device_id,
        'address': local_address,
        'role': 'super_client',
        'server_id': server_id
    }

    requests.post(
        f'http://{main_server_address}/register',
        json=data
    )

training_data = {}

def create_app():
    app = Flask(__name__)
    # CORS(app)  # Enable Cross-Origin Resource Sharing

    # Create a custom logger
    logger = get_logger(
        log_dir=os.path.join(dirname, f'../logs/{architecture}'),
        device_id=device_id,
        role='edge'
    )

    # Create results folder
    results_folder = os.path.join(dirname, f'../results/{architecture}')
    os.makedirs(results_folder, exist_ok=True)

    # Results file path
    results_file_path = os.path.join(results_folder, f'{device_id}_client.txt')

    # Register with main server
    with app.app_context():
        data = {
            'id': device_id,
            'address': local_address,
            'role': 'client',
            'server_id': server_id
        }

        requests.post(
            f'http://{main_server_address}/register',
            json=data
        )

    @app.route("/", methods=["GET"])
    def home():
        return jsonify({"message": "Super Client API!"})
    
    # Register Server
    @app.route("/register_server", methods=["POST"])
    def register_server():
        data = request.json

        # Add client information to data store
        data_store['server_id'] = data.get('id', None)
        data_store['server_address'] = data.get('address', None)
        data_store['partition_id'] = data.get('partition_id', None)
        data_store['num_clients'] = data.get('num_clients', 0)
        data_store['role'] = data.get('role', '')

        logger.info(f'Server address: {data_store["server_address"]}')

        return 'OK'
    
    # Register Clients
    @app.route("/register_client", methods=["POST"])
    def register_client():
        data = request.json

        id = data.get('id', None)
        address = data.get('address', None)

        # Add client information to data store
        data_store['clients'].append({
            'id': id,
            'address': address,
        })

        logger.info(data_store['clients'])

        return 'OK'
    
    # Start Training
    @app.route("/start_training", methods=["POST"])
    def start_training():
        data = request.json

        training_data['batch_size'] = data.get('batch_size', None)
        training_data['learning_rate'] = data.get('learning_rate', None)
        training_data['num_epochs'] = data.get('num_epochs', None)
        training_data['parameters'] = data.get('parameters', None)

        # If there are clients, start training in clients
        if data_store['role'] == 'super_client':
            # Start Training signal for clients
            logger.info('Starting training as super client')

            edge_thread = threading.Thread(
                target=start_training_edge,
                args=[
                    data_store['clients'],
                    training_data,
                    logger
                ]
            )

            edge_thread.start()
    
        # Else, start training as client
        elif data_store['role'] == 'client':
            logger.info('Starting training as client')

            thread = threading.Thread(
                target=train_round_client,
                args=[
                    data_store['id'],
                    data_store['server_address'],
                    data_store['num_clients'],
                    data_store['partition_id'],
                    logger,
                    training_data,
                    results_file_path
                ]
            )
    
            thread.start()

        return 'OK'

    # Aggregation
    @app.route("/aggregate", methods=["POST"])
    def aggregate_clients():
        data = request.json

        # Add model parameters to data store
        device_id = data.get('id', None)
        parameters = data.get('parameters', None)
        accuracy = data.get('accuracy', None)
        num_samples = data.get('num_samples', None)
        round_num = data.get('round', None)

        data_store['clients_data'][device_id] = {
            'parameters': parameters,
            'accuracy': accuracy,
            'num_samples': num_samples
        }

        # If all clients in this round have sent the data
        if data_store['role'] == 'super_client':
            if len(data_store['clients_data']) == len(data_store['clients']):
                logger.info('Received parameters from all clients in edge server')
                
                thread = threading.Thread(
                    target=train_round_super_client,
                    args=[
                        local_address,
                        num_edge_client_rounds,
                        data_store['clients'],
                        data_store['clients_data'],
                        data_store['server_address'],
                        data_store['id'],
                        data_store['num_clients'],
                        data_store['partition_ids'],
                        data_store['batch_size'],
                        data_store['learning_rate'],
                        data_store['num_epochs'],
                        round_num,
                        logger,
                        results_file_path
                    ]
                )
                thread.start()

        return 'OK'
    
    # Reset round training data
    @app.route("/reset_clients", methods=["POST"])
    def reset():
        data_store['clients'] = []
        logger.info('Reset clients')

        return 'OK'
    
    # Reset round training data
    @app.route("/reset_clients_data", methods=["POST"])
    def reset():
        data_store['clients_data'] = {}
        logger.info('Reset clients data')

        return 'OK'

    # Stop training
    @app.route("/stop", methods=["POST"])
    def stop():
        # Stop
        pass

    return app

if __name__ == "__main__":
    app = create_app()

    # Run them both simultaneously so that the app is ready before the first request
    threading.Thread(target=register_with_server, daemon=True).start()
    app.run(debug=False, host="0.0.0.0", port=port)  # Run on all interfaces
