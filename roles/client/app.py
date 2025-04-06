import os
import sys

dirname = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(dirname, '../../'))

from flask import Flask, request, jsonify
import requests
import threading

from roles.utils import get_local_port, get_local_address
from roles.logging import get_logger
from tasks.traininig.train_round import train_round_client

port = get_local_port()
local_address = get_local_address(port)

edge_index = sys.argv[1]
client_index = sys.argv[2]
main_server_address = sys.argv[3]
device_id = sys.argv[4]

# Sample data storage (in-memory)
data_store = {
    'id': device_id,
    'partition_id': None,
    'num_clients': None,
    'server_id': None,
    'server_address': None
}

def create_app():
    app = Flask(__name__)
    # CORS(app)  # Enable Cross-Origin Resource Sharing

    logger = get_logger(
        log_dir=os.path.join(dirname, '../logs'),
        device_id=device_id,
        role='client'
    )


    with app.app_context():
        # Register with the main server
        data = {
            'id': device_id,
            'address': local_address,
            'role': 'client',
            'edge_index': edge_index
        }

        requests.post(
            f'http://{main_server_address}/register',
            json=data
        )

    @app.route("/", methods=["GET"])
    def home():
        return jsonify({"message": "Client API!"})

    @app.route('/bind', methods=["POST"])
    def bind():
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

        batch_size = data.get('batch_size', None)
        learning_rate = data.get('learning_rate', None)
        num_epochs = data.get('num_epochs', None)
        parameters = data.get('parameters', None)

        logger.info('Starting training in client')
        logger.info()

        # Training
        thread = threading.Thread(
            target=train_round_client,
            args=[
                data_store['id'],
                data_store['server_address'],
                data_store['num_clients'],
                data_store['partition_id'],
                batch_size,
                parameters,
                learning_rate,
                num_epochs
            ]
        )
        thread.start()

        return 'OK'

    # Stop App
    @app.route("/stop", methods=["POST"])
    def stop():
        # Kill the process
        pass

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=False, host="0.0.0.0", port=port)  # Run on all interfaces
