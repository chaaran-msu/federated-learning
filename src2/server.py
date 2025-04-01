import socket
from config import Config
from flask import Flask, request
from utils import get_port, slurm_edge, submit_slurm

app = Flask(__name__)
config = Config("_config.yml")

addresses = set()
port = get_port(0)
time = config.get_time()
hostname = socket.gethostname()
num_edges = config.get_num_edges()
ip = socket.gethostbyname(hostname)
local_address = ip + ":" + str(port)

for edge_index, edge in enumerate(config.get_all_edges_info()):
    mem = edge.get('mem', 1)
    cpu = edge.get('cpu', 1)
    cluster = edge.get('cluster', 'intel18')

    slurm = slurm_edge(time, mem, cpu, cluster, local_address, edge_index)
    submit_slurm(slurm)

@app.route('/')
def home():
    address = request.args.get('address', None)
    print(address)

    if address is not None:
        addresses.add(address)

    if len(addresses) == num_edges:
        print("Ready!")

    return 'OK'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=False)
