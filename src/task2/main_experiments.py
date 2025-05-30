import argparse
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from imports.algorithm import aggregative_optimization
from imports.loss import AggregativeLoss
from imports.scenarios import create_network_of_agents, create_aggregative_problem
from imports.plot import plot_scenario, plot_animation
# from imports.utils import get_average_estimate_error, get_average_consensus_error

plt.rcParams["font.family"] = "cmr10"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["axes.formatter.use_mathtext"] = True
plt.rcParams["font.size"] = 12
plt.rcParams["legend.fontsize"] = 12



def plot_loss(agents, history_z, history_sigma, label):
    """
        Plots the loss function, handling both centralized and distributed cases.
    """
    plt.plot([ sum(agents[i].loss(z[i], s[i]) for i in range(len(agents))) for z, s in zip(history_z, history_sigma) ], label=label)
    plt.yscale("log")


def plot_gradient(agents, history_z, history_sigma, label):
    """
        Plots the norm of the gradient of the loss, handling both centralized and distributed cases.
    """
    plt.plot([ np.linalg.norm( np.sum([agents[i].loss.tot_grad(z[i], s[i]) for i in range(len(agents))], axis=0) ) for z, s in zip(history_z, history_sigma) ], label=label)
    plt.yscale("log")



def aggregative_animate(num_agents, target_weight, barycenter_weight, graph_form, alpha, num_iters, agents_importance, seed):
    """
        Plain run that plots an animation of the estimations.
    """
    rng = np.random.default_rng(seed)
    vars_dim = 2
    
    # Create problem
    G, A = create_network_of_agents(num_agents, graph_form, seed=int(rng.integers(0, 2**32)))
    agents, targets_pos = create_aggregative_problem( num_agents, vars_dim, target_weight, barycenter_weight, agents_importance, int(rng.integers(0, 2**32)) )
    z0 = rng.random(size=(num_agents, vars_dim))

    # Solve problem
    history_z, history_sigma = aggregative_optimization(
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
    anim = plot_animation(agents, history_estimates=history_z, targets_pos=targets_pos, ff_threshold=None, sample_size=1)
    plt.show()


def aggregative_comparison(num_agents, vars_dim, graph_forms, alpha, num_iters, seed):
    """
        Experiment to compare the algorithm with different graph patterns.
    """
    rng = np.random.default_rng(seed)
    network_seed = int(rng.integers(0, 2**32))
    history_z = {}
    history_sigma = {}
    
    # Define the same problem for all graph patterns
    agents, targets_pos = create_aggregative_problem(num_agents, vars_dim, 1.0, 1.0, None, int(rng.integers(0, 2**32)))
    z0 = rng.random(size=(num_agents, vars_dim))

    # Solve for all graph patterns.
    for graph_form in graph_forms:
        G, A = create_network_of_agents(num_agents, graph_form, seed=network_seed)
        history_z[graph_form], history_sigma[graph_form] = aggregative_optimization(agents, z0.copy(), A, alpha, num_iters)

    # Present results
    for graph_form in graph_forms:
        print(
            f"{f'[{graph_form}]':<20} Loss: {sum( agents[i].loss(history_z[graph_form][-1, i], history_sigma[graph_form][-1, i]) for i in range(num_agents) ):.10f}"
        )

    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    plt.title("Loss")
    for graph_form in graph_forms:
        plot_loss(agents, history_z[graph_form], history_sigma[graph_form], f"{graph_form.replace('_', '-')}")
    plt.xlabel("$k$")
    plt.ylabel("$l(z^k)$ (log)")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Norm of loss gradient")
    for graph_form in graph_forms:
        plot_gradient(agents, history_z[graph_form], history_sigma[graph_form], f"{graph_form.replace('_', '-')}")
    plt.xlabel("$k$")
    plt.ylabel("$\\left\\Vert \\nabla l(z^k) \\right\\Vert_2$ (log)")
    plt.legend()

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Model training")
    parser.add_argument("--animation", action="store_true", default=False)
    parser.add_argument("--animation-target", action="store_true", default=False)
    parser.add_argument("--animation-barycenter", action="store_true", default=False)
    parser.add_argument("--animation-importance", action="store_true", default=False)
    parser.add_argument("--few-agents", action="store_true", default=False)
    parser.add_argument("--more-agents", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42, help="Initialization seed")
    args = parser.parse_args()

    if args.animation:
        print("--- Running with animation ---")
        aggregative_animate(
            num_agents = 5,
            barycenter_weight = 1.0,
            target_weight = 1.0,
            graph_form = "binomial_graph",
            alpha = 1e-2,
            num_iters = 5000,
            agents_importance = [1.0, 1.0, 1.0, 1.0, 1.0],
            seed = args.seed
        )

    if args.animation_target:
        print("--- Running with animation (more weight for targets) ---")
        aggregative_animate(
            num_agents = 5,
            barycenter_weight = 0.2,
            target_weight = 1.0,
            graph_form = "binomial_graph",
            alpha = 1e-2,
            num_iters = 5000,
            agents_importance = [1.0, 1.0, 1.0, 1.0, 1.0],
            seed = args.seed
        )

    if args.animation_barycenter:
        print("--- Running with animation (more weight for barycenter) ---")
        aggregative_animate(
            num_agents = 5,
            barycenter_weight = 1.0,
            target_weight = 0.2,
            graph_form = "binomial_graph",
            alpha = 1e-2,
            num_iters = 5000,
            agents_importance = [1.0, 1.0, 1.0, 1.0, 1.0],
            seed = args.seed
        )

    if args.animation_importance:
        print("--- Running with animation (differen agents importance) ---")
        aggregative_animate(
            num_agents = 5,
            barycenter_weight = 1.0,
            target_weight = 1.0,
            graph_form = "binomial_graph",
            alpha = 1e-2,
            num_iters = 5000,
            agents_importance = [4.6, 0.1, 0.1, 0.1, 0.1],
            seed = args.seed
        )

    if args.few_agents:
        print("--- Comparison with few agents ---")
        aggregative_comparison(
            num_agents = 5,
            vars_dim = 2,
            graph_forms = ["complete_graph", "binomial_graph", "cycle_graph", "star_graph", "path_graph"],
            alpha = 1e-2,
            num_iters = 5000,
            seed = args.seed
        )

    if args.more_agents:
        print("--- Comparison with more agents ---")
        aggregative_comparison(
            num_agents = 15,
            vars_dim = 2,
            graph_forms = ["complete_graph", "binomial_graph", "cycle_graph", "star_graph", "path_graph"],
            alpha = 1e-2,
            num_iters = 5000,
            seed = args.seed
        )