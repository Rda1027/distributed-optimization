import numpy as np


class Agent:
    def __init__(self, target_pos, agent_importance=1.0, target_weight=1.0, barycenter_weight=1.0):
        self.phi = Linear(agent_importance)
        self.loss = AggregativeLoss(target_pos, self.phi, target_weight, barycenter_weight)


class AggregativeLoss:
    def __init__(self, target_pos, phi, target_weight, barycenter_weight):
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

    def tot_grad(self, z_i, sigma):
        return self.grad1(z_i, sigma) + self.grad2(z_i, sigma)


class Linear:
    def __init__(self, coeff=1.0):
        self.coeff = coeff

    def __call__(self, z):
        return self.coeff * z
    
    def grad(self, z):
        return self.coeff * np.ones_like(z)