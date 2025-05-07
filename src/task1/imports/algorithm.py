import numpy as np
import networkx as nx
from .loss import Loss

from typing import Literal, Optional



def gradient_tracking(fn_list: list[Loss], z0, A, alpha, num_iters):
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