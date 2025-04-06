import os
import sys

dirname = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(dirname))
sys.path.append(os.path.join(dirname, '../../'))

from flask import Flask, request, jsonify
import requests
import subprocess
import threading
import uuid

from roles.utils import get_local_address, get_local_port
from roles.logging import get_logger
from allocate_resources import allocate_resources
from communication import bind_clients_edges
from tasks.traininig.train_round import train_round_server

# Obtain the server address
server_port = get_local_port()
server_address = get_local_address(server_port)
server_id = uuid.uuid4()


app = Flask(__name__)
# CORS(app)  # Enable Cross-Origin Resource Sharing

# Create a custom logger
logger = get_logger(
    log_dir=os.path.join(dirname, '../logs'),
    device_id=server_id,
    role='server'
)

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

    logger.info("test:")
    logger.info(data_store['edges'])
    logger.info(data_store['clients'])

    if len(data_store['edges']) == num_edges and len(data_store['clients']) == num_clients:
        threading.Thread(target=bind_clients_edges, args=[data_store['edges'], data_store['clients']], daemon=True).start()

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

    # If all devices have sent the data, aggregate and start next round
    if len(data_store['current_round_data']) == len(data_store['current_round_clients']):
        logger.info('Received parameters from all edge servers')
        
        thread = threading.Thread(
            target=train_round_server,
            args=[
                data_store['current_round_data'],
                data['current_round_clients']
            ]
        )
        thread.start()

    return 'OK'

# Stop training
@app.route('/stop')
def stop():
    for id in jobs:
        subprocess.run(['scancel', id])
    return f"Successfully canceled jobs"

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=server_port)  # Run on all interfaces
