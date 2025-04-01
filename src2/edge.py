import os
import sys
import socket
import requests
from config import Config
from flask import Flask, request
from utils import get_port, slurm_client, submit_slurm

edge_index = int(sys.argv[1])
parent_address = sys.argv[2]

app = Flask(__name__)
config = Config("_config.yml")

addresses = set()
time = config.get_time()
port = get_port(edge_index)
hostname = socket.gethostname()
ip = socket.gethostbyname(hostname)
local_address = ip + ":" + str(port)
num_clients = config.get_num_clients(edge_index)

for client_index, client in enumerate(config.get_all_client_info(edge_index)):
    mem = client.get('mem', 1)
    cpu = client.get('cpu', 1)
    cluster = client.get('cluster', 'intel18')

    # slurm = slurm_client(time, mem, cpu, cluster, local_address, edge_index, client_index)
    # submit_slurm(slurm)
    os.system("sbatch example.sh")

@app.route('/')
def home():
    address = request.args.get('address', None)
    print(address)

    if address is not None:
        addresses.add(address)

    if len(addresses) == num_clients:
        requests.get('http://' + parent_address + '/?address=' + local_address)

    return 'OK'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=False)
