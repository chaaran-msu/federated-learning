import os
import sys

dirname = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(dirname, '../../'))

from flask import Flask, request, jsonify
import requests
import threading

from roles.utils import get_local_port, get_local_address, get_current_time
from roles.logging import get_logger
from tasks.training.train_round import train_round_client

port = get_local_port()
local_address = get_local_address(port)

device_id = sys.argv[1]
server_id = sys.argv[2]
main_server_address = sys.argv[3]
architecture = sys.argv[4]

# Sample data storage (in-memory)
data_store = {
    'id': device_id,
    'partition_id': None,
    'num_clients': None,
    'server_id': None,
    'server_address': None,
}

resource_store = {
    'avg_cpu_percent_round': 0,
    'peak_cpu_percent_round': 0,
    'avg_cpu_percent': 0,
    'peak_cpu_percent': 0,
    'avg_mem_bytes_round': 0,
    'peak_mem_bytes_round': 0,
    'avg_mem_bytes': 0,
    'peak_mem_bytes': 0,
    'avg_training_time': 0
}

training_data = {
    'round': 0
}

def register_with_server():
    # Register with the main server
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

def create_app():
    app = Flask(__name__)
    # CORS(app)  # Enable Cross-Origin Resource Sharing

    logger = get_logger(
        log_dir=os.path.join(dirname, f'../logs/{architecture}'),
        device_id=device_id,
        role='client'
    )

    # Create results folder
    results_folder = os.path.join(dirname, f'../results/{architecture}')
    os.makedirs(results_folder, exist_ok=True)

    # Results file path
    results_file_path = os.path.join(results_folder, f'{device_id}_client.txt')

    @app.route("/", methods=["GET"])
    def home():
        return jsonify({"message": "Client API!"})

    @app.route('/register_server', methods=["POST"])
    def register_server():
        data = request.json

        data_store['server_id'] = data.get('server_id', None)
        data_store['server_address'] = data.get('server_address', None)
        data_store['partition_id'] = data.get('partition_id', None)
        data_store['num_clients'] = data.get('num_clients', None)

        return 'OK'

    # Start Training
    @app.route("/start_training", methods=["POST"])
    def start_training():
        data = request.json

        training_data['batch_size'] = data.get('batch_size', None)
        training_data['learning_rate'] = data.get('learning_rate', None)
        training_data['num_epochs'] = data.get('num_epochs', None)
        training_data['parameters'] = data.get('parameters', None)

        logger.info('Starting training in client')

        # Training
        thread = threading.Thread(
            target=train_round_client,
            args=[
                data_store['id'],
                data_store['server_address'],
                data_store['num_clients'],
                data_store['partition_id'],
                logger,
                training_data,
                results_file_path,
                resource_store
            ]
        )
        thread.start()

        return 'OK'

     # Stop App
    @app.route("/top", methods=["GET"])
    def top():
        return jsonify(resource_store)

    # Stop App
    @app.route("/stop", methods=["POST"])
    def stop():
        # Kill the process
        pass

    return app

if __name__ == "__main__":
    app = create_app()

    # Run them both simultaneously so that the app is ready before the first request
    threading.Thread(target=register_with_server, daemon=True).start()
    app.run(debug=False, host="0.0.0.0", port=port)  # Run on all interfaces
