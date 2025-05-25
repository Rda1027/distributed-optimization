import argparse
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from imports.algorithm import gradient_tracking, gradient_descent
from imports.loss import QuadraticFunction
from imports.scenarios import create_network_of_agents, create_quadratic_problem
from imports.utils import get_average_consensus_error

plt.rcParams["font.family"] = "cmr10"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["axes.formatter.use_mathtext"] = True
plt.rcParams["font.size"] = 12
plt.rcParams["legend.fontsize"] = 12



def plot_loss(loss_fn, history_z, label):
    """
        Plots the cost function, handling both centralized and distributed cases.
    """
    if history_z.ndim == 2: # Centralized case
        plt.plot([ loss_fn(z) for z in history_z ], label=label)
    elif history_z.ndim == 3: # Distributed case
        plt.plot([ sum(loss_fn[i](z[i]) for i in range(len(z))) for z in history_z ], label=label)


def plot_gradient(loss_fn, history_z, label):
    """
        Plots the norm of the gradient of the cost, handling both centralized and distributed cases.
    """
    if history_z.ndim == 2: # Centralized case
        plt.plot([ np.linalg.norm( loss_fn.grad(z), 2 ) for z in history_z ], label=label)
    elif history_z.ndim == 3: # Distributed case
        plt.plot([ np.linalg.norm( np.sum([loss_fn[i].grad(z[i]) for i in range(len(z))], axis=0), 2 ) for z in history_z ], label=label)
    plt.yscale("log")


def plot_distance_to_optimum(loss_fn, history_z, optimum, label):
    """
        Plots the distance of the loss computed at the estimates to the optimum, handling both centralized and distributed cases.
    """
    if history_z.ndim == 2: # Centralized case
        plt.plot([ abs(loss_fn(z) - optimum) for z in history_z ], label=label)
    elif history_z.ndim == 3: # Distributed case
        plt.plot([ abs(sum(loss_fn[i](z[i]) for i in range(len(z))) - optimum) for z in history_z ], label=label)
    plt.yscale("log")



def quadratic_comparison(num_agents, vars_dim, graph_forms, alpha, num_iters, seed):
    """
        Experiment to compare the gradient tracking algorithm with different graph patterns.
    """
    rng = np.random.default_rng(seed)
    network_seed = int(rng.integers(0, 2**32))
    problem_seed = int(rng.integers(0, 2**32))
    history_z = {}
    estimated_cost = {}

    # Define the same problem for all the graph patterns
    local_quadratics, global_quadratic, optimal_z = create_quadratic_problem(num_agents, vars_dim, seed=problem_seed)
    optimal_cost = global_quadratic(optimal_z)
    z0 = rng.random(size=(num_agents, vars_dim))

    # Solve for each graph pattern
    for graph_form in graph_forms:
        G, A = create_network_of_agents(num_agents, graph_form, seed=network_seed)
        history_z[graph_form] = gradient_tracking(local_quadratics, z0.copy(), A, alpha, num_iters)
        estimated_cost[graph_form] = sum( local_quadratics[i](history_z[graph_form][-1, i]) for i in range(num_agents) )

    # Present results
    print(f"Cost {'optimal':<15}: {optimal_cost:.10f}")
    for graph_form in graph_forms:
        print(f"Cost {graph_form:<15}: {estimated_cost[graph_form]:.10f} | Diff: {abs(estimated_cost[graph_form] - optimal_cost):.10f}")

    plt.figure(figsize=(16, 8))

    plt.subplot(2, 2, 1)
    plt.title("Cost")
    for graph_form in graph_forms:
        plot_loss(local_quadratics, history_z[graph_form], f"{graph_form.replace('_', '-')}")
    plt.plot([optimal_cost]*(num_iters+1), "--", label="optimum")
    plt.xlabel("$k$")
    plt.ylabel("$f(z^k)$")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.title("Norm of cost gradient")
    for graph_form in graph_forms:
        plot_gradient(local_quadratics, history_z[graph_form], f"{graph_form.replace('_', '-')}")
    plt.xlabel("$k$")
    plt.ylabel("$\\left\\Vert \\nabla f(z^k) \\right\\Vert_2$ (log)")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.title("Distance to optimum")
    for graph_form in graph_forms:
        plot_distance_to_optimum(local_quadratics, history_z[graph_form], optimal_cost, f"{graph_form.replace('_', '-')}")
    plt.xlabel("$k$")
    plt.ylabel("$| f(z_k) - f(z^*) |$ (log)")
    plt.legend()

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


def quadratic_centralized(num_agents, vars_dim, graph_form, alpha, num_iters, seed):
    """
        Experiment to compare the gradient tracking algorithm with the centralized gradient method.
    """
    rng = np.random.default_rng(seed)

    # Define the problem
    G, A = create_network_of_agents(num_agents, graph_form, seed=int(rng.integers(0, 2**32)))
    local_quadratics, global_quadratic, optimal_z = create_quadratic_problem(num_agents, vars_dim, seed=int(rng.integers(0, 2**32)))
    optimal_cost = global_quadratic(optimal_z)
    z0 = rng.random(size=(num_agents, vars_dim))

    # Solve problem
    history_z = gradient_tracking(local_quadratics, z0.copy(), A, alpha, num_iters)
    history_z_centr = gradient_descent(local_quadratics, z0[0].copy(), alpha, num_iters)
    estimated_cost = sum(local_quadratics[i](history_z[-1, i]) for i in range(num_agents))
    estimated_cost_centr = global_quadratic(history_z_centr[-1])

    # Present results
    print(f"Cost {'optimal':<20}: {optimal_cost:.10f}")
    print(f"Cost {'gradient tracking':<20}: {estimated_cost:.10f} | Diff: {abs(estimated_cost - optimal_cost):.10f}")
    print(f"Cost {'centralized gradient':<20}: {estimated_cost_centr:.10f} | Diff: {abs(estimated_cost_centr - optimal_cost):.10f}")

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 3, 1)
    plt.title("Cost")
    plot_loss(local_quadratics, history_z, "Gradient tracking")
    plot_loss(global_quadratic, history_z_centr, "Centralized gradient")
    plt.plot([optimal_cost]*(num_iters+1), "--", label="Optimum")
    plt.xlabel("$k$")
    plt.ylabel("$f(z^k)$")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.title("Norm of cost gradient")
    plot_gradient(local_quadratics, history_z, "Gradient tracking")
    plot_gradient(global_quadratic, history_z_centr, "Centralized gradient")
    plt.xlabel("$k$")
    plt.ylabel("$\\left\\Vert \\nabla f(z^k) \\right\\Vert_2$ (log)")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.title("Distance to optimum")
    plot_distance_to_optimum(local_quadratics, history_z, optimal_cost, "Gradient tracking")
    plot_distance_to_optimum(global_quadratic, history_z_centr, optimal_cost, "Centralized gradient")
    plt.xlabel("$k$")
    plt.ylabel("$| f(z_k) - f(z^*) |$ (log)")
    plt.legend()

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Model training")
    parser.add_argument("--few-agents-low-dim", action="store_true", default=False)
    parser.add_argument("--few-agents-high-dim", action="store_true", default=False)
    parser.add_argument("--many-agents-low-dim", action="store_true", default=False)
    parser.add_argument("--many-agents-high-dim", action="store_true", default=False)
    parser.add_argument("--centralized-small", action="store_true", default=False)
    parser.add_argument("--centralized-large", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42, help="Initialization seed")
    args = parser.parse_args()

    if args.few_agents_low_dim:
        print("\n--- Comparison with few agents and low dimensionality ---")
        quadratic_comparison(
            num_agents = 5,
            vars_dim = 3,
            graph_forms = ["complete_graph", "binomial_graph", "cycle_graph", "star_graph", "path_graph"],
            alpha = 5e-2,
            num_iters = 5000,
            seed = args.seed
        )

    if args.few_agents_high_dim:
        print("\n--- Comparison with few agents and high dimensionality ---")
        quadratic_comparison(
            num_agents = 5,
            vars_dim = 15,
            graph_forms = ["complete_graph", "binomial_graph", "cycle_graph", "star_graph", "path_graph"],
            alpha = 5e-2,
            num_iters = 5000,
            seed = args.seed
        )

    if args.many_agents_low_dim:
        print("\n--- Comparison with many agents and low dimensionality ---")
        quadratic_comparison(
            num_agents = 15,
            vars_dim = 3,
            graph_forms = ["complete_graph", "binomial_graph", "cycle_graph", "star_graph", "path_graph"],
            alpha = 5e-2,
            num_iters = 5000,
            seed = args.seed
        )

    if args.many_agents_high_dim:
        print("\n--- Comparison with many agents and high dimensionality ---")
        quadratic_comparison(
            num_agents = 15,
            vars_dim = 15,
            graph_forms = ["complete_graph", "binomial_graph", "cycle_graph", "star_graph", "path_graph"],
            alpha = 5e-2,
            num_iters = 5000,
            seed = args.seed
        )

    if args.centralized_small:
        print("\n--- Comparison with centralized gradient ---")
        quadratic_centralized(
            num_agents = 5,
            vars_dim = 3,
            graph_form = "complete_graph",
            alpha = 1e-1,
            num_iters = 1000,
            seed = args.seed
        )

    if args.centralized_large:
        print("\n--- Comparison with centralized gradient ---")
        quadratic_centralized(
            num_agents = 15,
            vars_dim = 3,
            graph_form = "complete_graph",
            alpha = 5e-2,
            num_iters = 5000,
            seed = args.seed
        )