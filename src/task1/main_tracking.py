import argparse
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from imports.algorithm import gradient_tracking, gradient_descent
from imports.loss import QuadraticFunction, TargetLocalizationLoss
from imports.scenarios import create_network_of_agents, create_quadratic_problem, create_position_tracking_problem
from imports.plot import plot_scenario, plot_animation
from imports.utils import get_average_estimate_error, get_average_consensus_error

plt.rcParams["font.family"] = "cmr10"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["axes.formatter.use_mathtext"] = True
plt.rcParams["font.size"] = 12
plt.rcParams["legend.fontsize"] = 12



def plot_loss(loss_fn, history_z, label):
    if history_z.ndim == 3: # Centralized case
        plt.plot([ loss_fn(z) for z in history_z ], label=label)
    elif history_z.ndim == 4: # Distributed case
        plt.plot([ sum(loss_fn[i](z[i]) for i in range(len(z))) for z in history_z ], label=label)
    plt.yscale("log")


def plot_gradient(loss_fn, history_z, label):
    if history_z.ndim == 3: # Centralized case
        plt.plot([ np.linalg.norm( loss_fn.grad(z) ) for z in history_z ], label=label)
    elif history_z.ndim == 4: # Distributed case
        plt.plot([ np.linalg.norm( np.sum([loss_fn[i].grad(z[i]) for i in range(len(z))], axis=0) ) for z in history_z ], label=label)
    plt.yscale("log")



def tracking_animate(num_robots, num_targets, graph_form, alpha, num_iters, noise_args, seed):
    """
        Plain run that plots an animation of the estimations.
    """
    rng = np.random.default_rng(seed)
    vars_dim = 2
    
    G, A = create_network_of_agents(num_robots, graph_form, seed=int(rng.integers(0, 2**32)))
    local_losses, global_loss, (robots_pos, targets_pos_real, est_targets_dists) = create_position_tracking_problem(
        num_robots = num_robots,
        num_targets = num_targets,
        vars_dim = vars_dim,
        seed = int(rng.integers(0, 2**32)),
        **noise_args
    )
    z0 = rng.random(size=(num_robots, num_targets*vars_dim))

    history_z = gradient_tracking(local_losses, z0.copy(), A, alpha, num_iters)
    history_z = history_z.reshape(-1, num_robots, num_targets, vars_dim)

    print(
        f"Loss: {sum( local_losses[i](history_z[-1, i].flatten()) for i in range(num_robots) ):.10f}"
        f" | Average estimated distance error: {get_average_estimate_error(history_z[-1], targets_pos_real):.10f}"
    )

    anim = plot_animation(robots_pos, targets_pos_real, history_z, ff_threshold=50)
    plt.show()


def tracking_comparison(num_robots, num_targets, vars_dim, graph_forms, alpha, num_iters, noise_args, seed):
    """
        Experiment to compare the algorithm with different graphs.
    """
    rng = np.random.default_rng(seed)
    network_seed = int(rng.integers(0, 2**32))
    history_z = {}
    
    local_losses, global_loss, (robots_pos, targets_pos_real, est_targets_dists) = create_position_tracking_problem(
        num_robots = num_robots,
        num_targets = num_targets,
        vars_dim = vars_dim,
        seed = int(rng.integers(0, 2**32)),
        **noise_args
    )
    z0 = rng.random(size=(num_robots, num_targets*vars_dim))

    for graph_form in graph_forms:
        G, A = create_network_of_agents(num_robots, graph_form, seed=network_seed)

        history_z[graph_form] = gradient_tracking(local_losses, z0.copy(), A, alpha, num_iters)
        history_z[graph_form] = history_z[graph_form].reshape(-1, num_robots, num_targets, vars_dim)

    for graph_form in graph_forms:
        print(
            f"{f'[{graph_form}]':<20} Loss: {sum( local_losses[i](history_z[graph_form][-1, i].flatten()) for i in range(num_robots) ):.10f}"
            f" | Average estimated distance error: {get_average_estimate_error(history_z[graph_form][-1], targets_pos_real):.10f}"
        )

    plt.figure(figsize=(16, 8))

    plt.subplot(2, 2, 1)
    plt.title("Loss")
    for graph_form in graph_forms:
        plot_loss(local_losses, history_z[graph_form], f"{graph_form.replace('_', '-')}")
    plt.xlabel("$k$")
    plt.ylabel("$l(z^k)$ (log)")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.title("Norm of loss gradient")
    for graph_form in graph_forms:
        plot_gradient(local_losses, history_z[graph_form], f"{graph_form.replace('_', '-')}")
    plt.xlabel("$k$")
    plt.ylabel("$\\left\\Vert \\nabla l(z^k) \\right\\Vert_2$ (log)")
    plt.legend()

    # plt.subplot(2, 2, 3)
    # plt.title("Average estimated error")
    # for graph_form in graph_forms:
    #     plt.plot([get_average_estimate_error(z, targets_pos_real) for z in history_z[graph_form]], label=f"{graph_form.replace('_', '-')}")
    # plt.xlabel("Iteration")
    # plt.yscale("log")
    # plt.legend()

    plt.subplot(2, 2, 4)
    plt.title("Consensus error")
    for graph_form in graph_forms:
        plt.plot([get_average_consensus_error(z) for z in history_z[graph_form]], label=f"{graph_form.replace('_', '-')}")
    plt.xlabel("$k$")
    plt.ylabel("Average consensus error (log)")
    plt.yscale("log")
    plt.legend()

    plt.tight_layout()
    plt.show()


def tracking_centralized(num_robots, num_targets, vars_dim, graph_form, alpha, num_iters, noise_args, seed):
    """
        Experiment to compare the algorithm with the centralized gradient method.
    """
    rng = np.random.default_rng(seed)
    
    G, A = create_network_of_agents(num_robots, graph_form, seed=int(rng.integers(0, 2**32)))
    local_losses, global_loss, (robots_pos, targets_pos_real, est_targets_dists) = create_position_tracking_problem(
        num_robots = num_robots,
        num_targets = num_targets,
        vars_dim = vars_dim,
        seed = int(rng.integers(0, 2**32)),
        **noise_args
    )
    z0 = rng.random(size=(num_robots, num_targets*vars_dim))

    history_z = gradient_tracking(local_losses, z0.copy(), A, alpha, num_iters)
    history_z = history_z.reshape(-1, num_robots, num_targets, vars_dim)
    history_z_centr = gradient_descent(local_losses, z0[0].copy(), alpha, num_iters)
    history_z_centr = history_z_centr.reshape(-1, 1, num_targets, vars_dim)

    print(
        f"Loss {'gradient tracking':<20}: {sum( local_losses[i](history_z[-1, i].flatten()) for i in range(num_robots) ):.10f}"
        f" | Average estimated distance error: {get_average_estimate_error(history_z[-1], targets_pos_real):.10f}"
    )
    print(
        f"Loss {'centralized gradient':<20}: {sum( local_losses[i](history_z_centr[-1, 0].flatten()) for i in range(num_robots) ):.10f}"
        f" | Average estimated distance error: {get_average_estimate_error(history_z_centr[-1], targets_pos_real):.10f}"
    )

    plt.figure(figsize=(16, 5))

    plt.subplot(1, 2, 1)
    plt.title("Loss")
    plot_loss(local_losses, history_z, f"Gradient tracking")
    plt.plot([ sum(local_losses[i](z.flatten()) for i in range(num_robots)) for z in history_z_centr ], label="Centralized gradient")
    plt.xlabel("$k$")
    plt.ylabel("$l(z^k)$ (log)")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Norm of loss gradient")
    plot_gradient(local_losses, history_z, f"Gradient tracking")
    plt.plot([ np.linalg.norm(np.sum([local_losses[i].grad(z.flatten()) for i in range(num_robots)], axis=0)) for z in history_z_centr ], label="Centralized gradient")
    plt.xlabel("$k$")
    plt.ylabel("$\\left\\Vert \\nabla l(z^k) \\right\\Vert_2$ (log)")
    plt.legend()

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Model training")
    parser.add_argument("--animation", action="store_true", default=False)
    parser.add_argument("--few-robots-1-target", action="store_true", default=False)
    parser.add_argument("--few-robots-many-targets", action="store_true", default=False)
    parser.add_argument("--many-robots-many-targets", action="store_true", default=False)
    parser.add_argument("--centralized", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42, help="Initialization seed")
    args = parser.parse_args()

    if args.animation:
        print("--- Running with animation ---")
        tracking_animate(
            num_robots = 5,
            num_targets = 2,
            graph_form = "complete_graph",
            alpha = 1e-3,
            num_iters = 5000,
            noise_args = {
                "noise_ratio": 0.01,
                "noise_type": "gaussian",
                "gaussian_mean": 0.0,
                "gaussian_std": 1.0,
            },
            seed = 42
        )

    if args.few_robots_1_target:
        print("--- Comparison with few robots and 1 target ---")
        tracking_comparison(
            num_robots = 5,
            num_targets = 1,
            vars_dim = 2,
            graph_forms = ["complete_graph", "binomial_graph", "cycle_graph", "star_graph", "path_graph"],
            alpha = 1e-2,
            num_iters = 500,
            noise_args = {
                "noise_type": "gaussian",
                "gaussian_mean": 0.0,
                "gaussian_std": 1.0,
                "noise_ratio": 0.01,
            },
            seed = 42
        )

    if args.few_robots_many_targets:
        print("--- Comparison with few robots and many targets ---")
        tracking_comparison(
            num_robots = 5,
            num_targets = 3,
            vars_dim = 2,
            graph_forms = ["complete_graph", "binomial_graph", "cycle_graph", "star_graph", "path_graph"],
            alpha = 1e-2,
            num_iters = 500,
            noise_args = {
                "noise_type": "gaussian",
                "gaussian_mean": 0.0,
                "gaussian_std": 1.0,
                "noise_ratio": 0.01,
            },
            seed = 42
        )

    if args.many_robots_many_targets:
        print("--- Comparison with many robots and many targets ---")
        tracking_comparison(
            num_robots = 15,
            num_targets = 3,
            vars_dim = 2,
            graph_forms = ["complete_graph", "binomial_graph", "cycle_graph", "star_graph", "path_graph"],
            alpha = 1e-2,
            num_iters = 500,
            noise_args = {
                "noise_type": "gaussian",
                "gaussian_mean": 0.0,
                "gaussian_std": 1.0,
                "noise_ratio": 0.01,
            },
            seed = 42
        )

    if args.centralized:
        print("\n--- Comparison with centralized gradient ---")
        tracking_centralized(
            num_robots = 5,
            num_targets = 3,
            vars_dim = 2,
            graph_form = "complete_graph",
            alpha = 1e-2,
            num_iters = 2000,
            noise_args = {
                "noise_ratio": 0.1,
                "noise_type": "gaussian",
                "gaussian_mean": 0.0,
                "gaussian_std": 1.0,
            },
            seed = 42
        )