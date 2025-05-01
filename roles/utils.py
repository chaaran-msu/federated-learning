import socket
import subprocess
import time

def get_local_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', 0))
        return s.getsockname()[1]

def get_local_address(port):
    local = socket.gethostbyname(socket.gethostname()) + ":" + str(port)
    return local

def submit(
    device_id,
    server_id,
    main_server_address,
    role,
    time, 
    mem, 
    cpu, 
    cluster,
    architecture,
    num_edge_client_rounds
):
    script = f"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time={time}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpu}
#SBATCH --mem={mem}G
#SBATCH --constraint={cluster}

source ~/.venv/fl/bin/activate

srun python {'~/federated-learning/roles/client/app.py' if role == 'client' else '~/federated-learning/roles/edge/app.py'} {device_id} {server_id} {main_server_address} {architecture} {num_edge_client_rounds}"""

    filename = f"{device_id}.sh"
    with open(filename, 'w') as file:
        file.write(script)

    result = subprocess.run(['sbatch', filename], capture_output=True, text=True)
    #subprocess.run(['rm', filename], check=True)
    output = result.stdout

    return output.strip().split()[-1]

def monitor_cpu_usage(process, stop_event, resource_store, interval=1):
    cpu_utilization = []
    mem_utilization = []

    # Initialize CPU measurement baseline
    process.cpu_percent(interval=None)

    while not stop_event.is_set():
        cpu = process.cpu_percent(interval=None)  # Instant (non-blocking) usage since last call
        mem = process.memory_percent()

        cpu_utilization.append(cpu)
        mem_utilization.append(mem)

        time.sleep(interval)

    # Safely compute stats even if empty
    resource_store['avg_cpu_percent'] = sum(cpu_utilization) / len(cpu_utilization) if cpu_utilization else 0.0
    resource_store['peak_cpu_percent'] = max(cpu_utilization, default=0.0)
    resource_store['avg_mem_percent'] = sum(mem_utilization) / len(mem_utilization) if mem_utilization else 0.0
    resource_store['peak_mem_percent'] = max(mem_utilization, default=0.0)

def get_current_time():
    return time.time()