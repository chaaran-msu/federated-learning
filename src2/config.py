import yaml

class Config:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.config = self.read_yaml(file_path)
        self.time = self.config.get("defaults", {}).get("time", "00:05:00")
        self.edges = self.config.get("edges", [])

    def read_yaml(self, file_path: str):
        with open(file_path, 'r') as file:
            return yaml.safe_load(file) or {}

    def get_time(self):
        return self.time

    def get_num_edges(self):
        return len(self.edges)

    def get_num_clients(self, edge_index: int):
        return [len(edge.get("clients", [])) for edge in self.edges][edge_index]

    def get_edge_info(self, edge_index: int):
        if edge_index < 0 or edge_index >= len(self.edges):
            raise IndexError("Edge index out of range.")
        edge = self.edges[edge_index]
        return {
            "mem": edge.get("mem", 1),
            "cpu": edge.get("cpu", 1),
            "cluster": edge.get("cluster", "intel18"),
        }

    def get_all_edges_info(self):
        return [
            {
                "mem": edge.get("mem", 1),
                "cpu": edge.get("cpu", 1),
                "cluster": edge.get("cluster", "intel18"),
            }
            for edge in self.edges
        ]

    def get_client_info(self, edge_index: int, client_index: int):
        if edge_index < 0 or edge_index >= len(self.edges):
            raise IndexError("Edge index out of range.")
        edge = self.edges[edge_index]
        clients = edge.get("clients", [])

        if client_index < 0 or client_index >= len(clients):
            raise IndexError("Client index out of range.")

        client = clients[client_index]
        return {
            "mem": client.get("mem"),
            "cpu": client.get("cpu"),
            "cluster": client.get("cluster")
        }

    def get_all_client_info(self, edge_index: int):
        if edge_index < 0 or edge_index >= len(self.edges):
            raise IndexError("Edge index out of range.")
        edge = self.edges[edge_index]
        clients = edge.get("clients", [])

        return [
            {
                "mem": client.get("mem"),
                "cpu": client.get("cpu"),
                "cluster": client.get("cluster")
            }
            for client in clients
        ]

    def get_global_index(self, edge_index: int, client_index: int):
        num_clients = 0

        for i in range(edge_index):
            num_clients += self.get_num_clients(i)

        return num_clients + client_index + self.get_num_edges() + 1
