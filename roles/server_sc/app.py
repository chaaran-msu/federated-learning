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
from collections import defaultdict

from roles.utils import get_local_address, get_local_port, monitor_cpu_usage
from roles.logging import get_logger

from systems.super_client_fl.allocate_resources import allocate_resources
from systems.super_client_fl.registration import registration

from tasks.training.train_round import train_round_server
from tasks.selection.topology_generation import topology_generation_random

start_time = time.time()

# Set system architecture
architecture = sys.argv[1]
num_rounds = int(sys.argv[2])
num_edge_client_rounds = int(sys.argv[3])
num_devices = int(sys.argv[4])
num_clients_per_round = int(sys.argv[5])

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
    'clients': [],
    'current_round_num': 0,
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
    'avg_mem_bytes': 0,
    'peak_mem_bytes': 0,
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
registration_times = defaultdict(int)

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
def register():
    data = request.json

    id = data.get('id', None)
    address = data.get('address', None)

    data_store['all_clients'][id] = {
        'id': id,
        'address': address,
    }

    logger.info("test:")
    logger.info(data_store['all_clients'])
    logger.info(len(data_store['all_clients']))

    if len(data_store['all_clients']) == resources_data['num_devices']['clients']:
        # Client and Topology Selection for first round
        client_topologies = topology_generation_random(
            main_server_id=device_id,
            main_server_address=server_address,
            clients=data_store['all_clients']
        )

        threading.Thread(target=registration, args=[device_id, client_topologies, data_store['current_round_num'], registration_times, checkpoint_times, data_store, True], daemon=True).start()

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
    round_num = int(data.get('round', None))
    
    data_store['current_round_data'][device_id] = {
        'parameters': parameters,
        'accuracy': accuracy,
        'num_samples': num_samples
    }

    # If all devices have sent the data, aggregate and start next round
    if len(data_store['current_round_data']) == len(data_store['current_round_clients']):
        data_store['current_round_num'] += 1
        logger.info('Received parameters from all clients')
        
        thread = threading.Thread(
            target=train_round_server,
            args=[
                data_store,
                device_id,
                server_address,
                num_rounds,
                data_store['current_round_data'],
                data_store['all_clients'],
                data_store['current_round_clients'],
                resources_data['num_devices']['clients'],
                list(range(resources_data['num_devices']['clients'])),
                16,
                data_store['current_round_num'],
                logger,
                results_file_path,
                checkpoint_times,
                True
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

    try:
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
    except Exception as e:
        logger.error(e)

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
            checkpoint_times,
            num_devices
        ],
        daemon=True
    )
    allocate_resources_thread.start()

    # Start monitor server thread
    monitor_server_thread.start()

    app.run(debug=False, host="0.0.0.0", port=server_port)  # Run on all interfaces
