import networkx as nx

class Solution:
    def solve(self, spec_path: str = None) -> dict:
        algo_code = '''
import networkx as nx

class BroadCastTopology:
    def __init__(self, src: str, dsts: list[str], num_partitions: int):
        self.src = src
        self.dsts = dsts
        self.num_partitions = int(num_partitions)
        self.paths = {dst: {str(i): None for i in range(self.num_partitions)} for dst in dsts}

    def append_dst_partition_path(self, dst: str, partition: int, path: list):
        partition = str(partition)
        if self.paths[dst][partition] is None:
            self.paths[dst][partition] = []
        self.paths[dst][partition].append(path)

    def set_dst_partition_paths(self, dst: str, partition: int, paths: list):
        partition = str(partition)
        self.paths[dst][partition] = paths

    def set_num_partitions(self, num_partitions: int):
        self.num_partitions = num_partitions

def search_algorithm(src: str, dsts: list[str], G: nx.DiGraph, num_partitions: int):
    """
    A basic broadcast topology optimizer that chooses the lowest-cost path
    for every (destination, partition) pair using Dijkstra’s algorithm.
    """
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    for dst in dsts:
        try:
            path_nodes = nx.dijkstra_path(G, src, dst, weight="cost")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # If no path exists, skip this destination
            continue

        # Build edge list once for reuse across partitions
        edge_sequence = []
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            edge_sequence.append([u, v, G[u][v]])

        # Assign identical path to every partition for this destination
        for partition_id in range(num_partitions):
            bc_topology.set_dst_partition_paths(dst, partition_id, list(edge_sequence))

    return bc_topology
'''
        return {"code": algo_code}
