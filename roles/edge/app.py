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
from tasks.traininig.train_round import train_round_edge
from tasks.traininig.start_training import start_training_edge

port = get_local_port()
local_address = get_local_address(port)

device_id = sys.argv[1]
server_id = sys.argv[2]
main_server_address = sys.argv[3]
architecture = sys.argv[4]

# Sample data storage (in-memory)
data_store = {
    'id': device_id,
    'server_id': '',
    'server_address': '',
    'clients': [], # Each client will have it's address
    'current_round_clients': [], # Client Index in 'clients' list
    'current_round_data': {}, # Parameters for current round,
    'num_clients': 0,
    'partition_ids': [],
    'batch_size': 16,
}

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
    results_file_path = os.path.join(results_folder, f'{device_id}_edge.txt')


    # Register with main server
    with app.app_context():
        data = {
            'id': device_id,
            'address': local_address,
            'role': 'edge',
            'server_id': server_id
        }

        requests.post(
            f'http://{main_server_address}/register',
            json=data
        )

    @app.route("/", methods=["GET"])
    def home():
        return jsonify({"message": "Edge Server API!"})
    
    # Register Server
    @app.route("/register_server", methods=["POST"])
    def register_server():
        data = request.json

        # Add client information to data store
        data_store['server_id'] = data.get('id', None)
        data_store['server_address'] = data.get('address', None)
        data_store['partition_ids'] = data.get('partition_ids', None)
        data_store['num_clients'] = data.get('num_clients', None)

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

        batch_size = data.get('batch_size', None)
        learning_rate = data.get('learning_rate', None)
        num_epochs = data.get('num_epochs', None)
        parameters = data.get('parameters', None)

        data_store['batch_size'] = batch_size

        training_data = {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'parameters': parameters
        }

        # Client Selection
        data_store['current_round_clients'] = data_store['clients']

        # Reset data from previous round
        data_store['current_round_data'] = {}

        # Start Training signal for clients
        logger.info('Starting training from edge')

        thread = threading.Thread(
            target=start_training_edge,
            args=[
                data_store['current_round_clients'],
                training_data,
                logger
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

        data_store['current_round_data'][device_id] = {
            'parameters': parameters,
            'accuracy': accuracy,
            'num_samples': num_samples
        }

        # If all clients in this round have sent the data
        if len(data_store['current_round_data']) == len(data_store['current_round_clients']):
            logger.info('Received parameters from all clients in edge server')
            
            thread = threading.Thread(
                target=train_round_edge,
                args=[
                    data_store['current_round_data'],
                    data_store['server_address'],
                    data_store['id'],
                    data_store['num_clients'],
                    data_store['partition_ids'],
                    data_store['batch_size'],
                    round_num,
                    logger,
                    results_file_path
                ]
            )
            thread.start()

        return 'OK'

    # Stop training
    @app.route("/stop", methods=["POST"])
    def stop():
        # Stop
        pass

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=False, host="0.0.0.0", port=port)  # Run on all interfaces
