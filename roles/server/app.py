import os
import sys

dirname = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(dirname))
sys.path.append(os.path.join(dirname, '../../'))

from flask import Flask, request, jsonify
import requests
import subprocess
import threading

from tasks.aggregation.aggregate import aggregate
from roles.utils import get_local_address, get_local_port
from allocate_resources import allocate_resources, on_ready

# Obtain the server address
server_port = 5000
server_address = get_local_address(server_port)

app = Flask(__name__)
# CORS(app)  # Enable Cross-Origin Resource Sharing

# Sample data storage (in-memory)
data_store = {
    'clients': [], # Each client will have it's address
    'edges': [],
    'current_round_clients': [], # Client Index in 'clients' list,
    'current_round_data': {}
}

# Allocate resources
jobs, num_edges, num_clients = allocate_resources(
    server_address=server_address
)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Edge Server API!"})

# Register Clients
@app.route("/register", methods=["POST"])
def register_client():
    data = request.json

    id = data.get('id', None)
    address = data.get('address', None)
    role = data.get('role', None)
    edge_index = data.get('edge_index', None)

    # Add client information to data store
    if role == 'client':
        data_store['clients'].append({
            'id': id,
            'address': address,
            'edge_index': edge_index
        })
    elif role == 'edge':
        data_store['edges'].append({
            'id': id,
            'address': address,
            'edge_index': edge_index
        })

    print("test:")
    print(data_store['edges'])
    print(data_store['clients'])

    if len(data_store['edges']) == num_edges and len(data_store['clients']) == num_clients:
        threading.Thread(target=on_ready, args=[data_store['edges'], data_store['clients']], daemon=True).start()
        print("Ready!")

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
    
    data_store['current_round_data'][device_id] = {
        'parameters': parameters,
        'accuracy': accuracy,
        'num_samples': num_samples
    }

    # If all devices have sent the data, aggregate
    if len(data_store['current_round_parameters']) == len(data_store['current_round_clients']):
        total_num_samples, aggregated_parameters = aggregate(data_store['current_round_data'])

        # Start Training for next round
        for client_idx in data_store['current_round_clients']:
            client_url = data_store['clients'][client_idx]['address']

            data = {
                'batch_size': 16,
                'learning_rate': 0.1,
                'num_epochs': 1,
                'parameters': aggregated_parameters
            }

            # Ask the client to start training
            requests.post(f"{client_url}/start_training", json=data)

    return 'OK'

# Stop training
@app.route('/stop')
def stop():
    for id in jobs:
        subprocess.run(['scancel', id])
    return f"Successfully canceled jobs"

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=server_port)  # Run on all interfaces
