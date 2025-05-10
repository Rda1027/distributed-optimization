import numpy as np

import numpy.typing as npt



def get_average_estimated_distance(robots_estimates: npt.NDArray, targets_pos_real: npt.NDArray) -> float:
    """
        Computes the average distance between the target's estimated and real positions.

        Args:
            robots_estimates (npt.NDArray):
                Estimated position by the robots ([NUM_ROBOTS x NUM_TARGETS x VARS_DIM]).
            targets_pos_real (npt.NDArray):
                Real position of the targets ([NUM_TARGETS x VARS_DIM]).
        
        Returns:
            average_distance (float):
                Average distance between estimated and real positions, considering each individual estimate.
    """
    num_robots = len(robots_estimates)
    num_targets = len(targets_pos_real)
    dists = 0
    
    for i in range(num_robots):
        for j in range(num_targets):
            dists += np.linalg.norm( robots_estimates[i, j] - targets_pos_real[j] )
    
    return dists / (num_robots*num_targets)