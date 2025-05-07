import numpy as np
import networkx as nx

from typing import Literal



def create_network_of_agents(
        num_agents: int, 
        adjacency_form: Literal["unweighted", "row-stochastic", "columm-stochastic", "doubly-stochastic"] = "unweighted",
        self_loops: bool = True,
        connected: bool = True,
        graph_algorithm: Literal["erdos_renyi"] = "erdos_renyi",
        erdos_renyi_p: float = 0.3, 
        doubly_stochastic_num_iter: int = 50,
        seed: int = 42
    ):
    # Create communication graph
    G = None
    while G is None:
        match graph_algorithm:
            case "erdos_renyi":
                G = nx.erdos_renyi_graph(n=num_agents, p=erdos_renyi_p, seed=seed)
        if connected != nx.is_connected(G): 
            G = None
            seed += 1
    if self_loops:
        G.add_edges_from([(i, i) for i in range(num_agents)])

    # Create adjacency matrix
    adj_matrix = nx.adjacency_matrix(G).toarray().astype(np.float32)
    match adjacency_form:
        case "unweighted":
            pass
        case "row-stochastic":
            # for i in range(N):
                # adj_matrix[i, :] = adj_matrix[i, :] / np.sum(adj_matrix[i, :])
            adj_matrix = adj_matrix / np.sum(adj_matrix, axis=1, keepdims=True)
            assert np.all(np.isclose(adj_matrix.sum(axis=1), np.ones((adj_matrix.shape[0]))))
        case "column-stochastic":
            adj_matrix = adj_matrix / np.sum(adj_matrix, axis=0, keepdims=True)
            assert np.all(np.isclose(adj_matrix.sum(axis=0), np.ones((adj_matrix.shape[1]))))
        case "doubly-stochastic":
            for _ in range(doubly_stochastic_num_iter):
                adj_matrix = adj_matrix / adj_matrix.sum(axis=0, keepdims=True)
                adj_matrix = adj_matrix / adj_matrix.sum(axis=1, keepdims=True)
                adj_matrix = np.abs(adj_matrix)
            assert np.all(np.isclose(adj_matrix.sum(axis=0), np.ones((adj_matrix.shape[1])))), "Might need a higher `doubly_stochastic_num_iter`"
            assert np.all(np.isclose(adj_matrix.sum(axis=1), np.ones((adj_matrix.shape[0])))), "Might need a higher `doubly_stochastic_num_iter`"
        case _:
            raise RuntimeError("Invalid matrix form")

    return G, adj_matrix