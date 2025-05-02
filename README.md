# Federated Learning on MSU HPCC

This guide explains how to deploy and run the Federated Learning (FL) system on Michigan State University’s High‑Performance Computing Center (MSU HPCC).

---

## 1  Prerequisites
* **MSU HPCC account & SSH access**  
* **Clone** of this repository  
* **Python 3.9+** on the login *and* dev nodes  
  * Use `module spider python` if system Python is unavailable.  

---

## 2  Log in to HPCC
```bash
ssh <username>@hpcc.msu.edu
```
Replace `<username>` with your MSU NetID.

---

## 3  Move to a development node & start an interactive job
```bash
ssh dev-intel18
salloc -N 1 -n 1 -c 128 --mem=4G --constraint="intel18" --time=02:00:00
```

| Flag | Meaning |
|------|---------|
| `-N 1`              | one node |
| `-n 1`              | one task |
| `-c 128`            | 128 CPU cores (matches *intel18* topology) |
| `--mem=4G`          | 4 GB RAM |
| `--time=02:00:00`   | 2‑hour wall‑clock limit |

Once your allocation starts, the prompt will look like `hpcc-dev1$`.

---

## 4  Create a dedicated virtual environment
Keep all cluster venvs in a hidden folder in `$HOME` to stay organized.

```bash
# One‑time setup
mkdir -p ~/.venv               # if you didn't already have it
python3 -m venv ~/.venv/fl
```

### Activate the environment
```bash
source ~/.venv/fl/bin/activate
```

---

## 5  Clone the repo and install dependencies
```bash
git git@github.com:chaaran-msu/federated-learning.git
cd federated-learning
pip install --upgrade pip
pip install -r requirements.txt   # re‑run whenever requirements.txt changes
```

---

## 6  Start the FL server
```bash
cd federated-learning/roles/server
python app.py <architecture> <num_rounds> <num_edge_client_rounds> <num_clients> <num_clients_per round>
```

| Positional arg               | Description |
|------------------------------|-------------|
| `<architecture>`             | FL topology to deploy (see table below) |
| `<num_rounds>`               | Number of **global** aggregation rounds |
| `<num_edge_client_rounds>`   | Number of **local** rounds per edge‑client block |
| `<num_clients>`   | Number of clients participating in training |
| `<num_clients_per round>`   | Number of clients participating in training in a round |

The last two options are available for only **traditional_fl** architecture. 

### Supported architectures

| Name              | Description                                      | Where to run |
|-------------------|--------------------------------------------------|--------------|
| `traditional_fl`  | Two‑tier: central server ⇆ clients               | Start server **and** each client separately |
| `hierarchical_fl` | Three‑tier: server ⇆ edge ⇆ clients              | Start one server, one + edges, multiple clients |
| `multi_hfl`       | Multi‑edge HFL: server ⇆ *N* edges ⇆ *M* clients  | Start **one** server and as many edges/clients as needed |

#### Example
```bash
# 20 global rounds, 10 local client rounds
python app.py multi_hfl 20 10
```
---

## 8 Configuration Files (optional)

Resource-allocation defaults (nodes, CPU cores, memory, GPU constraints, etc.) are centralized in Python scripts that live next to each FL topology. You can edit the one that matches the **architecture** you intend to run before launching jobs.

| Topology package | Config script path |
|------------------|--------------------|
| Traditional FL   | `systems/traditional_fl/allocate_resources.py` |
| Hierarchical FL  | `systems/hierarchical_fl/allocate_resources.py` |
| Multi-Level HFL  | `systems/multi_level_fl/allocate_resources.py` |

These scripts expose dictionaries and helper functions consumed by both the SLURM wrappers and the launcher utilities, so any parameter you tweak here—node counts, cpu counts, memory sizing, partition constraints, etc.—will propagate automatically.

---

## 9 Running the training for super client architecture
```bash
cd federated-learning/roles/server_sc
python app.py <architecture> <num_rounds> <num_edge_client_rounds>
```

| Positional arg               | Description |
|------------------------------|-------------|
| `<architecture>`             | FL topology to deploy (see table below) |
| `<num_rounds>`               | Number of **global** aggregation rounds |
| `<num_edge_client_rounds>`   | Number of **local** rounds per edge‑client block |

## 10  Deactivating and cleaning up
```bash
deactivate           # Leave the virtualenv
exit                 # Quit the dev node
```
---

Feel free to open an issue or contact the maintainers if you run into problems.
