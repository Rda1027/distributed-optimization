import numpy as np
import networkx as nx
from .loss import Agent

from typing import Literal, Optional
import numpy.typing as npt



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
        # Parameters update
        for i in range(num_agents):
            z[k+1, i] = z[k, i] - alpha * (agents[i].loss.grad1(z[k, i], s[k, i]) + v[k, i] * agents[i].phi.grad(z[k, i]))

        # Innovation update
        for i in range(num_agents):
            neighbors = np.nonzero(A[i])[0]
            s[k+1, i] = sum(A[i, j] * s[k, j] for j in neighbors) + (agents[i].phi(z[k+1, i]) - agents[i].phi(z[k, i])) 
            v[k+1, i] = sum(A[i, j] * v[k, j] for j in neighbors) + (agents[i].loss.grad2(z[k+1, i], s[k+1, i]) - agents[i].loss.grad2(z[k, i], s[k, i])) 

    return z, s


# def centralized_aggregative(
#         agents: list[Agent], 
#         z0: npt.NDArray, 
#         alpha: float, 
#         num_iters: int
#     ) -> npt.NDArray:
#     """
#         Runs the aggregative optimization algorithm where each agent has access to global information.

#         Args:
#             agents (list[Agent]):
#                 List of the agents.
#             z0 (npt.NDArray):
#                 Initial guess ([NUM_AGENTS x VARS_DIM]).
#             alpha (float):
#                 Step size.
#             num_iters (int):
#                 Number of iterations.

#         Returns:
#             z_history (npt.NDArray):
#                 Parameters estimated at each iteration ([num_iters+1 x NUM_AGENTS x VARS_DIM]).
#             sigma_history (npt.NDArray):
#                 Aggregation function at each iteration ([num_iters+1 x NUM_AGENTS x VARS_DIM])
#     """
#     num_agents = len(agents)
#     vars_dim = z0.shape[1]
#     z = np.zeros((num_iters+1, num_agents, vars_dim))
#     s = np.zeros((num_iters+1, num_agents, vars_dim))
#     z[0] = z0
#     s[0] = (1/num_agents) * sum(agents[i].phi(z0[i]) for i in range(num_agents))

#     for k in range(num_iters):
#         sigma = (1/num_agents) * sum(agents[i].phi(z[k, i]) for i in range(num_agents))
#         s[k+1] = sigma

#         for i in range(num_agents):
#             grad = (
#                 agents[i].loss.grad1(z[k, i], sigma)
#                 + sum(agents[j].loss.grad2(z[k, j], sigma) * (1/num_agents)*agents[i].phi(z[k, i]) for j in range(num_agents))
#             )
#             z[k+1, i] = z[k, i] - alpha * grad

#     return z, s