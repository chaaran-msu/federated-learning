import os
import sys

dirname = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(dirname, '../../'))

from flask import Flask, request, jsonify
import requests

from roles.utils import get_local_port, get_local_address

port = get_local_port()
local_address = get_local_address(port)

edge_index = sys.argv[1]
client_index = sys.argv[2]
main_server_address = sys.argv[3]

# Sample data storage (in-memory)
data_store = {
    'server_address': None,
    'clients': [], # Each client will have it's address
    'current_round_clients': [], # Client Index in 'clients' list
    'current_round_data': {}, # Parameters for current round
}

def create_app():
    app = Flask(__name__)
    # CORS(app)  # Enable Cross-Origin Resource Sharing

    # Register with main server
    with app.app_context():
        data = {
            'address': local_address,
            'role': 'edge',
            'edge_index': edge_index

        }

        requests.get(
            f'http://{main_server_address}/register',
            json=data
        )

    @app.route("/", methods=["GET"])
    def home():
        return jsonify({"message": "Edge Server API!"})

    # Register Clients
    @app.route("/register_client", methods=["POST"])
    def register_client():
        data = request.json

        address = data.get('address', None)

        # Add client information to data store
        data_store['clients'].append({
            'address': address,
        })

    # Start Training
    @app.route("/start_training", methods=["POST"])
    def start_training():
        data = request.json

        batch_size = data.get('batch_size', None)
        learning_rate = data.get('learning_rate', None)
        num_epochs = data.get('num_epochs', None)
        parameters = data.get('parameters', None)

        for client_idx in data_store['current_round_clients']:
            client_url = data_store['clients'][client_idx]['address']

            data = {
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'num_epochs': num_epochs,
                'parameters': parameters
            }

            # Ask the client to start training
            requests.post(f"{client_url}/start_training", json=data)

    # Aggregation
    @app.route("/aggregate", methods=["POST"])
    def aggregate():
        data = request.json

        # Add model parameters to data store
        parameters = data.get('parameters', None)
        accuracy = data.get('accuracy', None)
        num_samples = data.get('num_samples', None)
        partition_id = data.get('partition_id', None)

        data_store['current_round_data'][partition_id] = {
            'parameters': parameters,
            'accuracy': accuracy,
            'num_samples': num_samples
        }

        # If all devices have sent the data, aggregate
        if len(data_store['current_round_data']) == len(data_store['current_round_clients']):
            pass

        # Send parameters up the hierarchy

    # Stop training
    @app.route("/stop", methods=["POST"])
    def stop():
        # Stop
        pass

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5000)  # Run on all interfaces