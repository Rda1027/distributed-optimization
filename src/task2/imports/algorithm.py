import numpy as np
from .loss import Agent, AggregativeLoss, Linear

import numpy.typing as npt


def aggregative_step(
        z_i: npt.NDArray, 
        s_i: npt.NDArray, 
        v_i: npt.NDArray, 
        alpha: float, 
        loss: AggregativeLoss, 
        phi: Linear, 
        adj_neighbors: npt.NDArray, 
        s_neighbors: npt.NDArray, 
        v_neighbors: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
        Performs the update step of aggregative optimization for a single agent.

        Args:
            z_i (npt.NDArray):
                Current estimate of the parameters [VARS_DIM]
            s_i (npt.NDArra):
                Current estimate of sigma [VARS_DIM]
            v_i (npt.NDArray)
                Current estimate of the gradient w.r.t. sigma [VARS_DIM]
            alpha (float)
            loss (AggregativeLoss)
            phi (Linear)
            adj_neighbors (npt.NDArray):
                Weights of the neighbors [N_NEIGHBORS x 1]
            s_neighbors (npt.NDArray):
                Estimates of sigma of the neighbors [N_NEIGHBORS x VARS_DIM]
            v_neighbors (npt.NDArray):
                Estimates of the gradient of the neighbors [N_NEIGHBORS x VARS_DIM]

        Returns:
            z_next (npt.NDArray)
            s_next (npt.NDArray)
            v_next (npt.NDArray)
    """
    # Parameters update
    z_next = z_i - alpha * (loss.grad1(z_i, s_i) + v_i * phi.grad(z_i))

    # Innovation update
    s_next = sum(a_ij * s_j for a_ij, s_j in zip(adj_neighbors, s_neighbors)) + (phi(z_next) - phi(z_i)) 
    v_next = sum(a_ij * v_j for a_ij, v_j in zip(adj_neighbors, v_neighbors)) + (loss.grad2(z_next, s_next) - loss.grad2(z_i, s_i)) 

    return z_next, s_next, v_next


def aggregative_optimization(
        agents: list[Agent], 
        z0: npt.NDArray, 
        A: npt.NDArray, 
        alpha: float, 
        num_iters: int
    ) -> npt.NDArray:
    """
        Runs the gradient tracking algorithm

        Args:
            agents (list[Agent]):
                List of the agents.
            z0 (npt.NDArray):
                Initial guess for each agent ([NUM_AGENTS x VARS_DIM]).
            A (npt.NDArray):
                Adjacency matrix of the network of agents.
            alpha (float):
                Step size.
            num_iters (int):
                Number of iterations.

        Returns:
            z_history (npt.NDArray):
                Parameters estimated by each agent at each iteration ([num_iters+1 x NUM_AGENTS x VARS_DIM]).
            sigma_history (npt.NDArray):
                Aggregation function at each iteration ([num_iters+1 x NUM_AGENTS x VARS_DIM])

    """
    num_agents = z0.shape[0]
    vars_dim = z0.shape[1]
    z = np.zeros((num_iters+1, num_agents, vars_dim))
    s = np.zeros((num_iters+1, num_agents, vars_dim))
    v = np.zeros((num_iters+1, num_agents, vars_dim))
    z[0] = z0
    s[0] = np.array([agents[i].phi(z0[i]) for i in range(num_agents)])
    v[0] = np.array([agents[i].loss.grad2(z0[i], s[0, i]) for i in range(num_agents)])

    for k in range(num_iters):
        for i in range(num_agents):
            neighbors = np.nonzero(A[i])[0]

            z[k+1, i], s[k+1, i], v[k+1, i] = aggregative_step(
                z_i = z[k, i],
                s_i = s[k, i],
                v_i = v[k, i],
                alpha = alpha,
                loss = agents[i].loss,
                phi = agents[i].phi,
                adj_neighbors = A[i, neighbors],
                s_neighbors = s[k, neighbors],
                v_neighbors = v[k, neighbors]
            )

    return z, s