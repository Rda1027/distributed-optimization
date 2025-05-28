import numpy as np
import networkx as nx
# from imports.loss import Loss, QuadraticFunction, TargetLocalizationLoss

from typing import Literal
import numpy.typing as npt



def create_network_of_agents(
        num_agents: int, 
        graph_form: Literal["complete_graph", "binomial_graph", "cycle_graph", "path_graph", "star_graph"] = "binomial_graph",
        binomial_graph_p: float = 0.3,
        dtype = np.float64,
        seed: int = 42
    ) -> tuple[nx.Graph, npt.NDArray]:
    """
        Creates the graph representing a network of agents and its corrisponding adjacency matrix respecting given properties.

        Args:
            num_agents (int):
                Number of agents.
            adjacency_form (Literal["unweighted", "row-stochastic", "columm-stochastic", "doubly-stochastic"]):
                Structure of the adjacency matrix.
            graph_form (Literal["complete_graph", "binomial_graph", "cycle_graph", "path_graph", "star_graph"]):
                Algorithm or structure of the graph.
            binomial_graph_p (float):
                Edge probability for the Erdős-Rényi graph.
            dtype:
                Data type of the adjacency matrix.
            seed (int):
                Seed for non-deterministic operations.

        Returns:
            graph (nx.Graph):
                Communication graph.
            adjacency_matrix  (npt.NDArray):
                Adjacency matrix.
    """
    # Create communication graph
    match graph_form:
        case "complete_graph":
            G = nx.complete_graph(num_agents)
        case "cycle_graph":
            G = nx.cycle_graph(num_agents)
        case "path_graph":
            G = nx.path_graph(num_agents)
        case "star_graph":
            G = nx.star_graph(num_agents-1)
        case "binomial_graph":
            G = None
            while (G is None) or (not nx.is_connected(G)):
                G = nx.binomial_graph(n=num_agents, p=binomial_graph_p, seed=seed)
                seed += 1
        case _:
            raise RuntimeError("Invalid graph algorithm")
    # Add self-loops
    G.add_edges_from([(i, i) for i in range(num_agents)])

    # Create adjacency matrix
    adj_matrix = nx.adjacency_matrix(G).toarray().astype(dtype)
    degrees = np.sum(adj_matrix, axis=0)
    for i in range(num_agents):
        for j in range(num_agents):
            if (i != j) and adj_matrix[i, j] != 0:
                adj_matrix[i, j] = 1 / ( 1 + max(degrees[i], degrees[j]) )
        adj_matrix[i, i] = 1 - (sum(adj_matrix[i]) - adj_matrix[i, i])

    return G, adj_matrix
