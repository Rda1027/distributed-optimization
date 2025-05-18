import numpy as np
import networkx as nx
from imports.loss import Loss, QuadraticFunction, TargetLocalizationLoss

from typing import Literal
import numpy.typing as npt



def create_network_of_agents(
        num_agents: int, 
        graph_form: Literal["complete_graph", "binomial_graph", "cycle_graph", "path_graph", "star_graph"] = "binomial_graph",
        binomial_graph_p: float = 0.3, 
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
    adj_matrix = nx.adjacency_matrix(G).toarray().astype(np.float32)
    degrees = np.sum(adj_matrix, axis=0)
    for i in range(num_agents):
        for j in range(num_agents):
            if (i != j) and adj_matrix[i, j] != 0:
                adj_matrix[i, j] = 1 / ( 1 + max(degrees[i], degrees[j]) )
        adj_matrix[i, i] = 1 - (sum(adj_matrix[i]) - adj_matrix[i, i])

    return G, adj_matrix


def create_quadratic_problem(
        num_agents: int, 
        vars_dim: int,
        seed: int = 42
    ) -> tuple[npt.NDArray, list[QuadraticFunction], QuadraticFunction, npt.NDArray]:
    """
        Creates a network of agents each assigned to a local quadratic function with randomized coefficients.

        Args:
            num_agents (int):
                Number of agents.
            vars_dim (int):
                Dimensions of the parameters.
            seed (int):
                Seed for non-deterministic operations.

        Returns:
            local_quadratics (list[QuadraticFunction]):
                Local quadratic function of each agent.
            global_quadratic (QuadraticFunction):
                Aggregate of the local quadratic functions (i.e., the function to minimize).
            optimal_z (npt.NDArray):
                Optimal set of parameters to minimize the global quadratic function.
    """
    rng = np.random.default_rng(seed)

    # Define local coefficients Q and r
    local_Qs = [ np.diag( rng.uniform(size=vars_dim) ) for _ in range(num_agents)] 
    local_rs = [ rng.normal(size=vars_dim) for _ in range(num_agents) ]
    # Compute global coefficients
    global_Q, global_r = np.sum(local_Qs, axis=0), np.sum(local_rs, axis=0)
    optimal_z = -np.linalg.inv(global_Q) @ global_r

    local_quadratics = [ QuadraticFunction(local_Qs[i], local_rs[i]) for i in range(num_agents) ]
    global_quadratic = QuadraticFunction(global_Q, global_r)

    return local_quadratics, global_quadratic, optimal_z


def create_position_tracking_problem(
        num_robots: int,
        num_targets: int,
        vars_dim: int,
        noise_type: Literal["gaussian", "poisson"] = "gaussian",
        gaussian_mean: float = 0.0,
        gaussian_std: float = 1.0,
        poisson_lambda: float = 1.0,
        noise_ratio: float = 0.0,
        seed: int = 42,
    ) -> tuple[npt.NDArray, list[TargetLocalizationLoss], TargetLocalizationLoss, tuple[npt.NDArray, npt.NDArray, npt.NDArray]]:
    """
        Creates a network of robots each assigned to a local loss function to solve the distributed position tracking problem.

        Args:
            num_robots (int):
                Number of robots.
            num_targets (int):
                Number of targets to track.
            vars_dim (int):
                Dimensionality of the variables.
            noise_type (Literal["gaussian"]):
                Distribution from which the noise is drawn.
            gaussian_mean (float):
                Mean for Gaussian noise.
            gaussian_std (float):
                Standard deviation for Gaussian noise.
            noise_ratio (float):
                Amount of noise to inject into the measured distance.
            seed (int):
                Seed for non-deterministic operations.

        Returns:
            local_losses (list[TargetLocalizationLoss]):
                Local loss function of each agent.
            global_loss (TargetLocalizationLoss):
                Aggregate of the local losses (i.e., the function to minimize).
            (robots_pos, targets_pos_real, est_targets_dists) (tuple[npt.NDArray, npt.NDArray, npt.NDArray]):
                Tuple with the setup of the problem:
                - The position of the robots ([num_robots x vars_dim]),
                - The position of the targets ([num_targets x vars_dim]),
                - The estimated distances ([num_robots x num_targets]).
    """
    def __get_noise(rng):
        match noise_type:
            case "gaussian":
                return rng.normal(gaussian_mean, gaussian_std)
            case "poisson":
                return rng.poisson(poisson_lambda)
    
    rng = np.random.default_rng(seed)

    # Generate targets and robots positions
    targets_pos_real = rng.random(size=(num_targets, vars_dim))
    robots_pos = rng.random(size=(num_robots, vars_dim))

    # Generate noisy distances
    est_targets_dists = np.zeros((num_robots, num_targets))
    for i in range(num_robots):
        for j in range(num_targets):
            est_targets_dists[i, j] = np.linalg.norm(robots_pos[i] - targets_pos_real[j], 2) + noise_ratio*__get_noise(rng)

    local_losses = [ TargetLocalizationLoss(robots_pos[i], est_targets_dists[i]) for i in range(num_robots) ]
    global_loss = Loss(
        fn = lambda z: sum(local_losses[i](z) for i in range(num_robots)),
        fn_grad = lambda z: sum(local_losses[i].grad(z) for i in range(num_robots))
    )

    return local_losses, global_loss, (robots_pos, targets_pos_real, est_targets_dists)