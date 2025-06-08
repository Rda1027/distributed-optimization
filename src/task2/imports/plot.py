import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
from .loss import Agent

from typing import Optional
import numpy.typing as npt



def plot_loss(agents: list[Agent], history_z: npt.NDArray, history_sigma: npt.NDArray, label: str):
    """
        Plots the loss function.

        Args:
            agents (list[Agent]):
                List of agents.
            history_z (npt.NDArray):
                Evolution of the position estimates [NUM_ITERS, NUM_AGENTS, VARS_DIM].
            history_sigma (npt.NDArray):
                Evolution of the sigma estimates [NUM_ITERS, NUM_AGENTS, VARS_DIM].
            label (str)
    """
    plt.plot([ sum(agents[i].loss(z[i], s[i]) for i in range(len(agents))) for z, s in zip(history_z, history_sigma) ], label=label)
    plt.yscale("log")


def plot_gradient(agents: list[Agent], history_z: npt.NDArray, history_sigma: npt.NDArray, history_v: npt.NDArray, label: str, precise: bool=False):
    """
        Plots the norm of the gradient of the loss.

        Args:
            agents (list[Agent]):
                List of agents.
            history_z (npt.NDArray):
                Evolution of the position estimates [NUM_ITERS, NUM_AGENTS, VARS_DIM].
            history_sigma (npt.NDArray):
                Evolution of the sigma estimates [NUM_ITERS, NUM_AGENTS, VARS_DIM].
            label (str)
            precise(bool):
                If True, the real sigma and the gradient w.r.t. sigma are used. 
                Otherwise, the estimates are used.
    """
    if precise:
        history_grad = []
        for z in history_z:
            # Compute sigma and grad2 in a centralized way
            sigma = np.mean([ agents[i].phi(z[i]) for i in range(len(agents)) ], axis=0)
            global_grad2 = sum( agents[j].loss.grad2(z[j], sigma) for j in range(len(agents)) )
            
            history_grad.append(
                np.linalg.norm( 
                    np.sum(
                        [ agents[i].loss.grad1(z[i], sigma) + global_grad2 * (1/len(agents))*agents[i].phi.grad(z[i]) for i in range(len(agents)) ], 
                        axis=0
                    )
                ) 
            )
    else:
        history_grad = [ 
            np.linalg.norm( 
                np.sum(
                    [ agents[i].loss.grad1(z[i], s[i]) + v[i] * agents[i].phi.grad(z[i]) for i in range(len(agents)) ], 
                    axis=0
                )
            ) for z, s, v in zip(history_z, history_sigma, history_v) 
        ]

    plt.plot(history_grad, label=label)
    plt.yscale("log")


def plot_scenario(
        agents,
        robots_pos: npt.NDArray, 
        target_pos: npt.NDArray,
        draw_line_to_target: bool = True,
        past_positions: Optional[npt.NDArray] = None,
        show_legend: bool = True
    ):
    """
        Plots an istance of the position tracking problem.

        Args:
            robots_pos (npt.NDArray):
                Position of the robots ([NUM_ROBOTS x VARS_DIM]).
            target_pos (npt.NDArray):
                Position of the private targets ([NUM_ROBOTS x VARS_DIM]).
            draw_line_to_target (bool):
                If True, a line connecting each robot to its private target is drawn.
            past_positions (Optional[npt.NDArray]):
                If provided, the trajectories of the robots are drawn. It should be the history of the estimates ([NUM_ITERS x NUM_ROBOTS x VARS_DIM]).
            show_legend (bool)
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
        agents: list[Agent],
        history_estimates: npt.NDArray, 
        targets_pos: npt.NDArray, 
        ff_threshold: Optional[int] = None,
        sample_size: int = 1
    ) -> matplotlib.animation.FuncAnimation:
    """
        Generates an animation for the position tracking problem.

        Args:
            agents (list[Agent]):
                List of agents.
            history_estimates (npt.NDArray):
                Evolution of the position estimates of the robots ([NUM_ITERS x NUM_ROBOTS x VARS_DIM]).
            targets_pos (npt.NDArray):
                Position of the private targets [NUM_ROBOTS x VARS_DIM].
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