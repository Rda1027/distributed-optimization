import argparse
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os

from imports.algorithm import aggregative_optimization
from imports.scenarios import create_network_of_agents, create_aggregative_problem
from imports.plot import plot_scenario, plot_loss, plot_gradient

plt.rcParams["font.family"] = "cmr10"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["axes.formatter.use_mathtext"] = True
plt.rcParams["font.size"] = 26
plt.rcParams["legend.fontsize"] = 26



def aggregative_animate(num_agents, target_weight, barycenter_weight, graph_form, alpha, num_iters, agents_importance, seed, out_dir, to_plot_timesteps):
    """
        Plain run that plots an animation of the agents.
    """
    os.makedirs(os.path.join("figs", out_dir), exist_ok=True)
    rng = np.random.default_rng(seed)
    vars_dim = 2
    
    # Create problem
    G, A = create_network_of_agents(num_agents, graph_form, seed=int(rng.integers(0, 2**32)))
    agents, targets_pos = create_aggregative_problem( num_agents, vars_dim, target_weight, barycenter_weight, agents_importance, int(rng.integers(0, 2**32)) )
    z0 = rng.random(size=(num_agents, vars_dim))

    # Solve problem
    history_z, history_sigma, history_v = aggregative_optimization(
        agents = agents,
        z0 = z0.copy(),
        A = A,
        alpha = alpha,
        num_iters = num_iters
    )

    # Results + animation
    print(
        f"Loss: {sum( agents[i].loss(history_z[-1, i], history_sigma[-1, i]) for i in range(num_agents) ):.10f}"
    )

    xlim_min = min(np.min(history_z[to_plot_timesteps, :, 0]), np.min(targets_pos[:, 0])) - 0.1
    xlim_max = max(np.max(history_z[to_plot_timesteps, :, 0]), np.max(targets_pos[:, 0])) + 0.1
    ylim_min = min(np.min(history_z[to_plot_timesteps, :, 1]), np.min(targets_pos[:, 1])) - 0.1
    ylim_max = max(np.max(history_z[to_plot_timesteps, :, 1]), np.max(targets_pos[:, 1])) + 0.1

    plt.figure(figsize=(18, 5.5))
    for i, t in enumerate(to_plot_timesteps):
        plt.subplot(1, len(to_plot_timesteps), i+1)
        plt.title(f"$k = {t}$")
        plot_scenario(
            agents, robots_pos=history_z[t], target_pos=targets_pos, 
            draw_line_to_target = True,
            past_positions = history_z[:t],
            show_legend=(i == (len(to_plot_timesteps)-1)))
        if i == (len(to_plot_timesteps)-1): 
            plt.legend(labelspacing=0.25, bbox_to_anchor=(1.0, 1.1))
        plt.xlim(xlim_min, xlim_max)
        plt.ylim(ylim_min, ylim_max)
    plt.savefig(f"figs/{out_dir}/anim.pdf", bbox_inches="tight")
    plt.close()


def aggregative_comparison(num_agents, vars_dim, graph_forms, alpha, num_iters, seed, out_dir):
    """
        Experiment to compare the algorithm with different graph patterns.
    """
    os.makedirs(os.path.join("figs", out_dir), exist_ok=True)
    rng = np.random.default_rng(seed)
    network_seed = int(rng.integers(0, 2**32))
    history_z = {}
    history_sigma = {}
    history_v = {}
    
    # Define the same problem for all graph patterns
    agents, targets_pos = create_aggregative_problem(num_agents, vars_dim, 1.0, 1.0, None, int(rng.integers(0, 2**32)))
    z0 = rng.random(size=(num_agents, vars_dim))

    # Solve for all graph patterns.
    for graph_form in graph_forms:
        G, A = create_network_of_agents(num_agents, graph_form, seed=network_seed)
        history_z[graph_form], history_sigma[graph_form], history_v[graph_form] = aggregative_optimization(agents, z0.copy(), A, alpha, num_iters)

    # Present results
    for graph_form in graph_forms:
        print(
            f"{f'[{graph_form}]':<20} Loss: {sum( agents[i].loss(history_z[graph_form][-1, i], history_sigma[graph_form][-1, i]) for i in range(num_agents) ):.10f}"
        )


    def __label_normalize(label):
        return label.replace("_", "-").replace("-graph", "")

    plt.figure(figsize=(8, 5))
    for graph_form in graph_forms:
        plot_loss(agents, history_z[graph_form], history_sigma[graph_form], f"{__label_normalize(graph_form)}")
    plt.xlabel("$k$")
    plt.ylabel("$l(z^k)$ (log)")
    plt.legend(ncol=3, loc="upper center", columnspacing=0.8, labelspacing=0.25, bbox_to_anchor=(0.4, 1.35))
    plt.savefig(f"figs/{out_dir}/loss.pdf", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    for graph_form in graph_forms:
        plot_gradient(agents, history_z[graph_form], history_sigma[graph_form], history_v[graph_form], f"{__label_normalize(graph_form)}")
    plt.xlabel("$k$")
    plt.ylabel("Gradient norm (log)")
    plt.legend(ncol=3, loc="upper center", columnspacing=0.8, labelspacing=0.25, bbox_to_anchor=(0.4, 1.35))
    plt.savefig(f"figs/{out_dir}/gradient.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Model training")
    parser.add_argument("--seed", type=int, default=42, help="Initialization seed")
    args = parser.parse_args()

    # aggregative_animate(
    #     num_agents = 5,
    #     barycenter_weight = 1.0,
    #     target_weight = 1.0,
    #     graph_form = "binomial_graph",
    #     alpha = 1e-2,
    #     num_iters = 5000,
    #     agents_importance = [1.0, 1.0, 1.0, 1.0, 1.0],
    #     seed = args.seed,
    #     out_dir = "plain_anim",
    #     to_plot_timesteps = [0, 50, 5000]
    # )

    # aggregative_animate(
    #     num_agents = 5,
    #     barycenter_weight = 0.2,
    #     target_weight = 1.0,
    #     graph_form = "binomial_graph",
    #     alpha = 1e-2,
    #     num_iters = 5000,
    #     agents_importance = [1.0, 1.0, 1.0, 1.0, 1.0],
    #     seed = args.seed,
    #     out_dir = "target_anim",
    #     to_plot_timesteps = [0, 50, 5000]
    # )

    # aggregative_animate(
    #     num_agents = 5,
    #     barycenter_weight = 1.0,
    #     target_weight = 0.2,
    #     graph_form = "binomial_graph",
    #     alpha = 1e-2,
    #     num_iters = 5000,
    #     agents_importance = [1.0, 1.0, 1.0, 1.0, 1.0],
    #     seed = args.seed,
    #     out_dir = "barycenter_anim",
    #     to_plot_timesteps = [0, 50, 5000]
    # )

    # aggregative_animate(
    #     num_agents = 5,
    #     barycenter_weight = 1.0,
    #     target_weight = 1.0,
    #     graph_form = "binomial_graph",
    #     alpha = 1e-2,
    #     num_iters = 5000,
    #     agents_importance = [3.0, 0.5, 0.5, 0.5, 0.5],
    #     seed = args.seed,
    #     out_dir = "importance_anim",
    #     to_plot_timesteps = [0, 50, 5000]
    # )

    aggregative_comparison(
        num_agents = 5,
        vars_dim = 2,
        graph_forms = ["complete_graph", "binomial_graph", "cycle_graph", "star_graph", "path_graph"],
        alpha = 1e-2,
        num_iters = 5000,
        seed = args.seed,
        out_dir = "few_agents"
    )

    aggregative_comparison(
        num_agents = 15,
        vars_dim = 2,
        graph_forms = ["complete_graph", "binomial_graph", "cycle_graph", "star_graph", "path_graph"],
        alpha = 1e-2,
        num_iters = 5000,
        seed = args.seed,
        out_dir = "more_agents"
    )

    aggregative_comparison(
        num_agents = 30,
        vars_dim = 2,
        graph_forms = ["complete_graph", "binomial_graph", "cycle_graph", "star_graph", "path_graph"],
        alpha = 1e-2,
        num_iters = 5000,
        seed = 42,
        out_dir = "lots_agents"
    )