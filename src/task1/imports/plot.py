import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np

from typing import Optional
import numpy.typing as npt



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
        plt.plot(robots_pos[i][0], robots_pos[i][1], "s", color=__get_color(i))

    # Plot real targets
    plt.plot(
        [targets_pos_real[i, 0] for i in range(len(targets_pos_real))], 
        [targets_pos_real[i, 1] for i in range(len(targets_pos_real))], 
        "X", color="tab:red", label="One Piece"
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
                "o", color=__get_color(i), alpha=0.6, label=f"{i}-Maybe One Piece?"
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
        frames = [*range(0, ff_threshold, sample_size)] + [*range(ff_threshold, len(history_estimates), 10*sample_size)]
    else:
        frames = range(0, len(history_estimates), sample_size)

    plt.figure(figsize=(10, 5))
    anim = matplotlib.animation.FuncAnimation(plt.gcf(), __update, frames, interval=100, blit=False)
    return anim