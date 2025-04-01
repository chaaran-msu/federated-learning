import os
import uuid
import psutil
import subprocess

def get_port(index, start=5000, end=6000):
    used_ports = {conn.laddr.port for conn in psutil.net_connections(kind="inet") if conn.laddr}
    available_ports = [port for port in range(start, end) if port not in used_ports]
    return available_ports[index]

def slurm_edge(time, mem, cpu, cluster, address, edge_index):
    script = f"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time={time}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpu}
#SBATCH --mem={mem}G
#SBATCH --constraint={cluster}

source ~/.venv/fl/bin/activate

cd ~/cse812/federated-learning/src
srun python edge.py {edge_index} {address}
"""

    return script

def slurm_client(time, mem, cpu, cluster, address, edge_index, client_index):
    script = f"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time={time}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpu}
#SBATCH --mem={mem}G
#SBATCH --constraint={cluster}

source ~/.venv/fl/bin/activate

cd ~/cse812/federated-learning/src
srun python client.py {edge_index} {client_index} {address}
"""

    return script

    return script

def submit_slurm(slurm_script):
    filename = f"{uuid.uuid4()}.sh"
    with open(filename, 'w') as file:
        file.write(slurm_script)

    os.system(f"sbatch --export=ALL,SLURM_CPU_BIND= {filename}")
    os.remove(filename)
