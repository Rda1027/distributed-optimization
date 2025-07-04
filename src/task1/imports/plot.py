import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np

from .loss import Loss, QuadraticFunction, TargetLocalizationLoss

from typing import Optional
import numpy.typing as npt


def plot_loss_quadratic(loss_fn: list[QuadraticFunction], history_z: npt.NDArray, label: str):
    """
        Plots the cost function for the quadratic task, handling both centralized and distributed cases.

        Args:
            loss_fn (list[QuadraticFunction]):
                Loss of each agent.
            history_z (npt.NDArray):
                History of the estimates [num_iters x vars_dim] or [num_iters x num_agents x vars_dim].
            label (str)
    """
    if history_z.ndim == 2: # Centralized case
        plt.plot([ loss_fn(z) for z in history_z ], label=label)
    elif history_z.ndim == 3: # Distributed case
        plt.plot([ sum(loss_fn[i](z[i]) for i in range(len(loss_fn))) for z in history_z ], label=label)


def plot_gradient_quadratic(loss_fn: list[QuadraticFunction], history_z: npt.NDArray, label: str):
    """
        Plots the norm of the gradient of the cost for the quadratic task, handling both centralized and distributed cases.

        Args:
            loss_fn (list[QuadraticFunction]):
                Loss of each agent.
            history_z (npt.NDArray):
                History of the estimates [num_iters x vars_dim] or [num_iters x num_agents x vars_dim].
            label (str)
    """
    if history_z.ndim == 2: # Centralized case
        plt.plot([ np.linalg.norm( loss_fn.grad(z), 2 ) for z in history_z ], label=label)
    elif history_z.ndim == 3: # Distributed case
        plt.plot([ np.linalg.norm( np.sum([loss_fn[i].grad(z[i]) for i in range(len(loss_fn))], axis=0), 2 ) for z in history_z ], label=label)
    plt.yscale("log")


def plot_distance_to_optimum_quadratic(loss_fn: list[QuadraticFunction], history_z: npt.NDArray, optimum: float, label: str):
    """
        Plots the distance of the cost function in the quadratic case to the optimum, handling both centralized and distributed cases.

        Args:
            loss_fn (list[QuadraticFunction]):
                Loss of each agent.
            history_z (npt.NDArray):
                History of the estimates [num_iters x vars_dim] or [num_iters x num_agents x vars_dim].
            optimum (float):
                Optimum of the sum of the cost functions.
            label (str)
    """
    if history_z.ndim == 2: # Centralized case
        plt.plot([ abs(loss_fn(z) - optimum) for z in history_z ], label=label)
    elif history_z.ndim == 3: # Distributed case
        plt.plot([ abs(sum(loss_fn[i](z[i]) for i in range(len(loss_fn))) - optimum) for z in history_z ], label=label)
    plt.yscale("log")


def plot_loss_tracking(loss_fn: list[TargetLocalizationLoss], history_z: npt.NDArray, label: str):
    """
        Plots the loss function of the aggregative optimization task, handling both centralized and distributed cases.

        Args:
            loss_fn (list[TargetLocalizationLoss]):
                Loss of each agent.
            history_z (npt.NDArray):
                History of the estimates [num_iters x num_targets x vars_dim] or [num_iters x num_robots x num_targets x vars_dim].
            label (str)
    """
    if history_z.ndim == 3: # Centralized case
        plt.plot([ sum(loss_fn[i](z) for i in range(len(loss_fn))) for z in history_z ], label=label)
    elif history_z.ndim == 4: # Distributed case
        plt.plot([ sum(loss_fn[i](z[i]) for i in range(len(loss_fn))) for z in history_z ], label=label)
    plt.yscale("log")


def plot_gradient_tracking(loss_fn: list[TargetLocalizationLoss], history_z: npt.NDArray, label: str):
    """
        Plots the norm of the gradient of the loss of the aggregative optimization task, handling both centralized and distributed cases.

        Args:
            loss_fn (list[TargetLocalizationLoss]):
                Loss of each agent.
            history_z (npt.NDArray):
                History of the estimates [num_iters x num_targets x vars_dim] or [num_iters x num_robots x num_targets x vars_dim].
            label (str)
    """
    if history_z.ndim == 3: # Centralized case
        plt.plot([ np.linalg.norm( sum(loss_fn[i].grad(z) for i in range(len(loss_fn))) ) for z in history_z ], label=label)
    elif history_z.ndim == 4: # Distributed case
        plt.plot([ np.linalg.norm( np.sum([loss_fn[i].grad(z[i]) for i in range(len(loss_fn))], axis=0) ) for z in history_z ], label=label)
    plt.yscale("log")



def plot_scenario(
        robots_pos: npt.NDArray, 
        targets_pos_real: npt.NDArray, 
        est_targets_dists: Optional[npt.NDArray] = None, 
        est_targets_pos: Optional[npt.NDArray] = None
    ):
    """
        Plots an istance of the position tracking problem.

        Args:
            robots_pos (npt.NDArray):
                Position of the robots ([NUM_ROBOTS x VARS_DIM]).
            targets_pos_real (npt.NDArray):
                Position of the targets ([NUM_TARGETS x VARS_DIM]).
            est_targets_dists (Optional[npt.NDArray]):
                Noisy distance measured by the robots ([NUM_ROBOTS x NUM_TARGETS]). 
                If provided, the radius that satisfy the distance for each robot is plotted.
            est_targets_pos (Optional[npt.NDArray]):
                Position of the targets estimated by the robots ([NUM_ROBOTS x NUM_TARGETS x VARS_DIM]).
                If provided, they will be plotted.
    """
    def __get_color(i):
        return matplotlib.colormaps["tab10"](i % 10)

    # Plot robots
    for i in range(len(robots_pos)):
        plt.plot(robots_pos[i][0], robots_pos[i][1], "s", color=__get_color(i), label=f"Robot-{i}")

    # Plot real targets
    plt.plot(
        [targets_pos_real[i, 0] for i in range(len(targets_pos_real))], 
        [targets_pos_real[i, 1] for i in range(len(targets_pos_real))], 
        "X", color="tab:red", label="Target"
    )

    # Plot robots estimated distance to target
    if est_targets_dists is not None:
        for i in range(len(est_targets_dists)):
            for j in range(len(est_targets_dists[i])):
                plt.gca().add_patch(
                    plt.Circle(robots_pos[i], est_targets_dists[i,j], color=__get_color(i), fill=False, linestyle="--", alpha=0.6)
                )
    
    # Plot robots' estimates
    if est_targets_pos is not None:
        for i in range(len(est_targets_pos)):
            plt.plot(
                [est_targets_pos[i, j, 0] for j in range(len(est_targets_pos[i]))], 
                [est_targets_pos[i, j, 1] for j in range(len(est_targets_pos[i]))], 
                "o", color=__get_color(i), alpha=0.6, label=f"Robot-{i} estimates"
            )

    plt.axis("scaled")
    plt.legend(bbox_to_anchor=(1, 1))



def plot_animation(
        robots_pos: npt.NDArray, 
        targets_pos_real: npt.NDArray, 
        history_estimates: npt.NDArray, 
        ff_threshold: Optional[int] = None,
        sample_size: int = 1
    ) -> matplotlib.animation.FuncAnimation:
    """
        Generates an animation for the position tracking problem.

        Args:
            robots_pos (npt.NDArray):
                Position of the robots ([NUM_ROBOTS x VARS_DIM]).
            targets_pos_real (npt.NDArray):
                Position of the targets ([NUM_TARGETS x VARS_DIM]).
            history_estimates (npt.NDArray):
                Evolution of the estimates of the robots ([NUM_ITERS x NUM_ROBOTS x NUM_TARGETS x VARS_DIM]).
            ff_threshold (Optional[int]):
                If set, after the provided number of frames, the animation will be speed up.
            sample_size (int):
                Step for sampling frames.

        Returns:
            anim (matplotlib.animation.FuncAnimation):
                Animation of the problem.
    """
    xlim_min = min(np.min(history_estimates[:, :, :, 0]), np.min(robots_pos[:, 0]), np.min(targets_pos_real[:, 0])) - 0.1
    xlim_max = max(np.max(history_estimates[:, :, :, 0]), np.max(robots_pos[:, 0]), np.max(targets_pos_real[:, 0])) + 0.1
    ylim_min = min(np.min(history_estimates[:, :, :, 1]), np.min(robots_pos[:, 1]), np.min(targets_pos_real[:, 1])) - 0.1
    ylim_max = max(np.max(history_estimates[:, :, :, 1]), np.max(robots_pos[:, 1]), np.max(targets_pos_real[:, 1])) + 0.1

    def __update(k):
        plt.clf()
        plot_scenario(robots_pos, targets_pos_real, est_targets_pos=history_estimates[k])
        plt.title(f"k={k}")
        plt.xlim(xlim_min, xlim_max)
        plt.ylim(ylim_min, ylim_max)

    if ff_threshold is not None:
        frames = [*range(0, ff_threshold, 1)] + [*range(ff_threshold, len(history_estimates), sample_size)]
    else:
        frames = range(0, len(history_estimates), sample_size)

    plt.figure(figsize=(10, 5))
    anim = matplotlib.animation.FuncAnimation(plt.gcf(), __update, frames, interval=100, blit=False)
    return anim