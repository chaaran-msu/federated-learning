import uuid
import socket
import subprocess

def get_local_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', 0))
        return s.getsockname()[1]

def get_local_address(port):
    local = socket.gethostbyname(socket.gethostname()) + ":" + str(port)
    return local

def submit(time, mem, cpu, cluster, server, role, edge_index, client_index=0):
    device_id = uuid.uuid4()

    script = f"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time={time}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpu}
#SBATCH --mem={mem}G
#SBATCH --constraint={cluster}

source ~/.venv/fl/bin/activate
cd ../../
pwd
srun python {'roles/client/app.py' if role == 'client' else 'roles/edge/app.py'} {edge_index} {client_index} {server} {device_id}> output_{'client' if role == 'client' else 'edge'}{edge_index}{client_index}.log"""

    filename = f"{device_id}.sh"
    with open(filename, 'w') as file:
        file.write(script)

    result = subprocess.run(['sbatch', filename], capture_output=True, text=True)
    #subprocess.run(['rm', filename], check=True)
    output = result.stdout

    return output.strip().split()[-1]
