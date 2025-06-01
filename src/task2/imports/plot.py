import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np

from typing import Optional
import numpy.typing as npt



def plot_scenario(
        agents,
        robots_pos: npt.NDArray, 
        target_pos: npt.NDArray,
        draw_line_to_target = True,
        past_positions = None,
        show_legend = True
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
    
    if draw_line_to_target:
        # Draw line between robots and targets
        for i in range(len(robots_pos)):
            plt.plot([robots_pos[i, 0], target_pos[i, 0]], [robots_pos[i, 1], target_pos[i, 1]], "--", color=__get_color(i), alpha=0.5)

    if past_positions is not None:
        # Draw trajectory
        for i in range(len(robots_pos)):
            plt.plot(past_positions[:, i, 0], past_positions[:, i, 1], color=__get_color(i), alpha=0.5)

    # Plot robots
    for i in range(len(robots_pos)):
        plt.plot(robots_pos[i, 0], robots_pos[i, 1], "s", color=__get_color(i), label=f"Robot-{i}")

    # Plot targets
    for i in range(len(target_pos)):
        plt.plot(target_pos[i, 0], target_pos[i, 1], "x", color=__get_color(i), label=f"Target-{i}")

    # Plot barycenter
    barycenter = np.mean([ agents[i].phi(robots_pos[i]) for i in range(len(agents)) ], axis=0)
    plt.plot(barycenter[0], barycenter[1], "^", label="Barycenter", color="black")

    plt.axis("scaled")
    if show_legend:
        plt.legend(bbox_to_anchor=(1, 1))



def plot_animation(
        agents,
        history_estimates: npt.NDArray, 
        targets_pos: npt.NDArray, 
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
    xlim_min = min(np.min(history_estimates[:, :, 0]), np.min(targets_pos[:, 0])) - 0.1
    xlim_max = max(np.max(history_estimates[:, :, 0]), np.max(targets_pos[:, 0])) + 0.1
    ylim_min = min(np.min(history_estimates[:, :, 1]), np.min(targets_pos[:, 1])) - 0.1
    ylim_max = max(np.max(history_estimates[:, :, 1]), np.max(targets_pos[:, 1])) + 0.1

    def __update(k):
        plt.clf()
        plot_scenario(agents, history_estimates[k], targets_pos, past_positions=history_estimates[:k])
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