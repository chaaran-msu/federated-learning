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
python app.py <architecture> <num_rounds> <num_edge_client_rounds>
```

| Positional arg               | Description |
|------------------------------|-------------|
| `<architecture>`             | FL topology to deploy (see table below) |
| `<num_rounds>`               | Number of **global** aggregation rounds |
| `<num_edge_client_rounds>`   | Number of **local** rounds per edge‑client block |

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

## 8  Deactivating and cleaning up
```bash
deactivate           # Leave the virtualenv
exit                 # Quit the dev node
```
---

Feel free to open an issue or contact the maintainers if you run into problems.
