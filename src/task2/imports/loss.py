import numpy as np

import numpy.typing as npt


class Agent:
    def __init__(self, target_pos: npt.NDArray, agent_importance: float=1.0, target_weight: float=1.0, barycenter_weight: float=1.0):
        """
            Args:
                target_pos (npt.NDArray):
                    Position of the private target [VARS_DIM].
                agent_importance (float):
                    Importance assigned to the agent in the aggregation function.
                target_weight (float):
                    Weighs the importance of target vicinity in the loss.
                barycenter_weight (float):
                    Weighs the importance of barycenter vicinity in the loss.
        """
        self.phi = Linear(agent_importance)
        self.loss = AggregativeLoss(target_pos, self.phi, target_weight, barycenter_weight)


class AggregativeLoss:
    def __init__(self, target_pos: npt.NDArray, phi, target_weight: float, barycenter_weight: float):
        self.target_pos = target_pos
        self.vars_dim = target_pos.shape[0]
        self.phi = phi
        self.target_weight = target_weight
        self.barycenter_weight = barycenter_weight

    def __call__(self, z_i, sigma):
        return (
            self.target_weight * (1/2)*np.linalg.norm(z_i - self.target_pos)**2 + 
            self.barycenter_weight * (1/2)*np.linalg.norm(z_i - sigma)**2
        )

    def grad1(self, z_i, sigma):
        return self.target_weight*(z_i - self.target_pos) + self.barycenter_weight*(z_i - sigma)

    def grad2(self, z_i, sigma):
        return -self.barycenter_weight*(z_i - sigma)


class Linear:
    def __init__(self, coeff=1.0):
        self.coeff = coeff

    def __call__(self, z):
        return self.coeff * z
    
    def grad(self, z):
        return self.coeff * np.ones_like(z)