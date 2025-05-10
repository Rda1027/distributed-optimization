import numpy as np
import networkx as nx
from .loss import Loss

from typing import Literal, Optional
import numpy.typing as npt



def gradient_tracking(
        fn_list: list[Loss], 
        z0: npt.NDArray, 
        A: npt.NDArray, 
        alpha: float, 
        num_iters: int
    ) -> npt.NDArray:
    """
        Runs the gradient tracking algorithm

        Args:
            fn_list (list[Loss]):
                List of local loss functions.
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
    """
    num_agents = z0.shape[0]
    vars_dim = z0.shape[1]
    z = np.zeros((num_iters+1, num_agents, vars_dim))
    s = np.zeros((num_iters+1, num_agents, vars_dim))
    z[0] = z0
    s[0] = np.array([fn_list[i].grad(z0[i]) for i in range(num_agents)])

    for k in range(num_iters):
        # Parameters update
        for i in range(num_agents):
            neighbors = np.nonzero(A[i])[0]
            z[k+1, i] = sum(A[i, j] * z[k, j] for j in neighbors) - alpha * s[k, i]

        # Innovation update
        for i in range(num_agents):
            neighbors = np.nonzero(A[i])[0]
            grad_i_prev = fn_list[i].grad(z[k, i])
            grad_i_curr = fn_list[i].grad(z[k+1, i])
            s[k+1, i] = sum(A[i, j] * s[k, j] for j in neighbors) + (grad_i_curr - grad_i_prev) 

    return z


def gradient_descent(
        fn_list: list[Loss], 
        z0: npt.NDArray, 
        alpha: float, 
        num_iters: int
    ):
    """
        Runs the centralized gradient descent algorithm to minimize the summation of a list of loss functions.

        Args:
            fn_list (list[Loss]):
                List of loss functions for which the summation is minimized.
            z0 (npt.NDArray):
                Initial guess ([VARS_DIM]).
            alpha (float):
                Step size.
            num_iters (int):
                Number of iterations.

        Returns:
            z_history (npt.NDArray):
                Parameters estimated at each iteration ([num_iters+1 x VARS_DIM]).
    """
    num_agents = len(fn_list)
    vars_dim = z0.shape[0]
    z = np.zeros((num_iters+1, vars_dim))
    z[0] = z0

    for k in range(num_iters):
        grad = sum( fn_list[i].grad(z[k]) for i in range(num_agents) )
        z[k+1] = z[k] - alpha * grad

    return z