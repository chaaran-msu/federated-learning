import os
import sys

dirname = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(dirname, '../../'))

from flask import Flask, request, jsonify
import requests

from roles.utils import get_local_port, get_local_address
from tasks.traininig.train_round import train_round

port = get_local_port()
local_address = get_local_address(port)

edge_index = sys.argv[1]
client_index = sys.argv[2]
main_server_address = sys.argv[3]

# Sample data storage (in-memory)
data_store = {
    'partition_id': None,
    'num_clients': None,
    'server_address': None
}

def create_app():
    app = Flask(__name__)
    # CORS(app)  # Enable Cross-Origin Resource Sharing

    with app.app_context():
        # Register with the main server
        data = {
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

        # Training
        round_data = train_round(
            num_clients=data_store['num_clients'],
            partition_id=data_store['partition_id'],
            parameters=parameters,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs
        )

        round_data['partitoin_id'] = data_store['partition_id']

        server_address = data_store['server_address']

        # Send data back to server
        requests.post(
            url=f'{server_address}/aggregate',
            json=round_data
        )

    # Stop App
    @app.route("/stop", methods=["POST"])
    def stop():
        # Kill the process
        pass

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=False, host="0.0.0.0", port=port)  # Run on all interfaces
