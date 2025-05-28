import numpy as np


class Agent:
    def __init__(self, target_pos):
        self.phi = Identity()
        self.loss = AggregativeLoss(target_pos, self.phi)


class AggregativeLoss:
    def __init__(self, target_pos, phi):
        self.target_pos = target_pos
        self.vars_dim = target_pos.shape[0]
        self.phi = phi

    def __call__(self, z_i, sigma):
        return (1/2)*np.linalg.norm(z_i - self.target_pos)**2 + (1/2)*np.linalg.norm(z_i - sigma)**2

    def grad1(self, z_i, sigma):
        return (z_i - self.target_pos) + (z_i - sigma)

    def grad2(self, z_i, sigma):
        return -(z_i - sigma)


class Identity:
    def __call__(self, z):
        return z
    
    def grad(self, z):
        return np.ones_like(z)