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
import time
import psutil

from roles.utils import get_local_address, get_local_port, monitor_cpu_usage, get_current_time
from roles.logging import get_logger

from systems.traditional_fl.allocate_resources import allocate_resources as allocate_resources_traditional_fl
from systems.hierarchical_fl.allocate_resources import allocate_resources as allocate_resources_hierarchical_fl
from systems.multi_level_fl.allocate_resources import allocate_resources as allocate_resources_multi_level_fl

from systems.traditional_fl.registration import registration as registration_traditional_fl
from systems.hierarchical_fl.registration import registration as registration_hierarchical_fl
from systems.multi_level_fl.registration import registration as registration_multi_level_fl


from tasks.training.train_round import train_round_server

start_time = time.time()

# Set system architecture
architecture = sys.argv[1]
num_rounds = int(sys.argv[2])
num_edge_client_rounds = int(sys.argv[3])

if architecture == 'traditional_fl':
    allocate_resources = allocate_resources_traditional_fl
    registration = registration_traditional_fl
elif architecture == 'hierarchical_fl':
    allocate_resources = allocate_resources_hierarchical_fl
    registration = registration_hierarchical_fl
elif architecture == 'multi_hfl':
    allocate_resources = allocate_resources_multi_level_fl
    registration = registration_multi_level_fl

# Obtain the server address
server_port = get_local_port()
server_address = get_local_address(server_port)
device_id = str(uuid.uuid4())

app = Flask(__name__)
# CORS(app)  # Enable Cross-Origin Resource Sharing

# Create a custom logger
logger = get_logger(
    log_dir=os.path.join(dirname, f'../logs/{architecture}_{num_rounds}_{num_edge_client_rounds}'),
    device_id=device_id,
    role='server'
)

# Create results folder
results_folder = os.path.join(dirname, f'../results/{architecture}_{num_rounds}_{num_edge_client_rounds}')
os.makedirs(results_folder, exist_ok=True)

# Results file path
results_file_path = os.path.join(results_folder, f'{device_id}_server.txt')


# Sample data storage (in-memory)
data_store = {
    'all_clients': {}, # Each client will have it's address
    'all_edges': {},
    'clients': [],
    'current_round_clients': [], # Client Index in 'clients' list,
    'current_round_data': {}
}

resources_data = {
    'jobs': set(),
    'num_devices': {
        'clients': 0,
        'edges': 0
    }
}


# Monitor Server thread
server_utilization_store = {
    'avg_cpu_percent': 0,
    'peak_cpu_percent': 0,
    'avg_mem_percent': 0,
    'peak_mem_percent': 0,
}

stop_monitoring_event = threading.Event()

monitor_server_thread = threading.Thread(
    target=monitor_cpu_usage,
    args=[
        psutil.Process(os.getpid()), 
        stop_monitoring_event, 
        server_utilization_store
    ],
    daemon=True
)

# Communication Latencies
latencies = []

# Checkpoint Times
checkpoint_times = {
    'resource_allocation_start': 0,
    'resource_allocation_end': 0,
    'training_start': 0,
    'training_end': 0,
}

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
    server_id = data.get('server_id', None)

    if server_id == device_id:
        data_store['clients'].append({
            'id': id,
            'address': address,
            'server_id': server_id
        })

    # Add client information to data store
    if role == 'client':
        data_store['all_clients'][id] = {
            'address': address,
            'server_id': server_id
        }
    elif role == 'edge':
        data_store['all_edges'][id] = {
            'address': address,
            'server_id': server_id
        }

    logger.info("test:")
    logger.info(data_store['all_edges'])
    logger.info(len(data_store['all_edges']))
    logger.info(data_store['all_clients'])
    logger.info(len(data_store['all_clients']))

    if architecture == 'traditional_fl':
        if len(data_store['all_clients']) == resources_data['num_devices']['clients']:
            # Client Selection in server
            data_store['current_round_clients'] = data_store['clients']
            threading.Thread(target=registration, args=[device_id, server_address, data_store['all_clients'], data_store['current_round_clients'], checkpoint_times], daemon=True).start()
    elif architecture == 'hierarchical_fl':
        if len(data_store['all_edges']) == resources_data['num_devices']['edges'] and len(data_store['all_clients']) == resources_data['num_devices']['clients']:
            # Client Selection in server
            data_store['current_round_clients'] = data_store['clients']
            threading.Thread(target=registration, args=[device_id, server_address, data_store['all_edges'], data_store['all_clients'], data_store['current_round_clients'], checkpoint_times], daemon=True).start()
    elif architecture == 'multi_hfl':
        if len(data_store['all_edges']) == resources_data['num_devices']['edges'] and len(data_store['all_clients']) == resources_data['num_devices']['clients']:
            # Client Selection in server
            data_store['current_round_clients'] = data_store['clients']
            threading.Thread(target=registration, args=[device_id, server_address, data_store['all_edges'], data_store['all_clients'], data_store['current_round_clients'], checkpoint_times], daemon=True).start()

    return 'OK'

# Aggregation
@app.route("/aggregate", methods=["POST"])
def aggregate_clients():
    request_received_time = get_current_time()

    data = request.json

    # Add model parameters to data store
    device_id = data.get('id', None)
    parameters = data.get('parameters', None)
    accuracy = data.get('accuracy', None)
    num_samples = data.get('num_samples', None)
    round_num = int(data.get('round', None))
    timestamp = int(data.get('timestamp', None))
    
    data_store['current_round_data'][device_id] = {
        'parameters': parameters,
        'accuracy': accuracy,
        'num_samples': num_samples
    }

    # Add communication latency
    latencies.append(
        request_received_time - timestamp
    )

    # If all devices have sent the data, aggregate and start next round
    if len(data_store['current_round_data']) == len(data_store['current_round_clients']):
        logger.info('Received parameters from all clients')
        
        thread = threading.Thread(
            target=train_round_server,
            args=[
                server_address,
                num_rounds,
                data_store['current_round_data'],
                data_store['current_round_clients'],
                resources_data['num_devices']['clients'],
                list(range(resources_data['num_devices']['clients'])),
                16,
                round_num,
                logger,
                results_file_path,
                checkpoint_times
            ]
        )
        thread.start()

    return 'OK'

# Reset round training data
@app.route("/reset", methods=["POST"])
def reset():
    data_store['current_round_data'] = {}
    logger.info('Reset previous round data')

    return 'OK'

# Stop training
@app.route('/stop', methods=["POST"])
def stop():
    # Set the stop event for monitoring server
    stop_monitoring_event.set()
    
    # Wait for the thread to stop
    monitor_server_thread.join()

    logger.info(f'Utilization: {server_utilization_store}')

    # Latency metrics
    logger.info(f'Minimum Latency: {min(latencies)}')
    logger.info(f'Maximum Latency: {max(latencies)}')
    logger.info(f'Average Latency: {sum(latencies)/len(latencies)}')

    # Times
    resource_allocation_time = checkpoint_times['resource_allocation_end'] - checkpoint_times['resource_allocation_start']
    logger.info(f'Resource Allocation time: {resource_allocation_time}')
    
    training_time = checkpoint_times['training_end'] - checkpoint_times['training_start']
    logger.info(f'Training time: {training_time}')

    for id in resources_data['jobs']:
        subprocess.run(['scancel', id])

    end_time = time.time()

    logger.info(f'Total time: {end_time - start_time}')

    return f"Successfully canceled jobs"

if __name__ == "__main__":
    # Allocate resources
    allocate_resources_thread = threading.Thread(
        target=allocate_resources,
        args=[
            device_id,
            server_address,
            f'{architecture}_{num_rounds}_{num_edge_client_rounds}',
            num_edge_client_rounds,
            resources_data,
            checkpoint_times
        ],
        daemon=True
    )
    allocate_resources_thread.start()

    # Start monitor server thread
    monitor_server_thread.start()

    # Start the server
    app.run(debug=False, host="0.0.0.0", port=server_port)  # Run on all interfaces
