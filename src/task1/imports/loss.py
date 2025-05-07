import numpy as np



class Loss:
    def __init__(self, fn, fn_grad):
        self.__fn = fn
        self.__fn_grad = fn_grad

    def __call__(self, x):
        return self.__fn(x)

    def grad(self, x):
        return self.__fn_grad(x)


class QuadraticFunction(Loss):
    def __init__(self, Q, r):
        super().__init__(
            fn      = lambda z: 0.5 * z.T @ Q @ z + r.T @ z,
            fn_grad = lambda z: Q @ z + r
        )


class TargetLocalizationLoss(Loss):
    def __init__(self, robot_pos, est_targets_dist):
        num_targets = len(est_targets_dist)
        vars_dim = robot_pos.shape[0]

        super().__init__(
            fn      = self.__fn_builder(robot_pos, est_targets_dist, num_targets, vars_dim),
            fn_grad = self.__grad_builder(robot_pos, est_targets_dist, num_targets, vars_dim)
        )

    def __fn_builder(self, robot_pos, est_targets_dist, num_targets, vars_dim):
        def _out(z):
            z = z.reshape(num_targets, vars_dim)
            val = sum( 
                ( est_targets_dist[j]**2 - np.linalg.norm(z[j] - robot_pos, 2)**2 )**2 
                for j in range(len(est_targets_dist)) 
            )
            return val.flatten()
        return _out

    def __grad_builder(self, robot_pos, est_targets_dist, num_targets, vars_dim):
        def _out(z):
            z = z.reshape(num_targets, vars_dim)
            grad = np.concatenate([
                -4 * (est_targets_dist[j]**2 - np.linalg.norm(z[j] - robot_pos, 2)**2) * (z[j] - robot_pos) 
                for j in range(len(est_targets_dist))
            ])
            return grad.flatten()
        return _out