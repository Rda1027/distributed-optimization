import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from imports.algorithm import gradient_tracking, gradient_descent
from imports.loss import QuadraticFunction
from imports.scenarios import create_network_of_agents, create_quadratic_problem
from imports.utils import get_average_consensus_error
from imports.plot import \
    plot_loss_quadratic as plot_loss, \
    plot_gradient_quadratic as plot_gradient, \
    plot_distance_to_optimum_quadratic as plot_distance_to_optimum
    
plt.rcParams["font.family"] = "cmr10"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["axes.formatter.use_mathtext"] = True
plt.rcParams["font.size"] = 26
plt.rcParams["legend.fontsize"] = 26



def quadratic_comparison(num_agents, vars_dim, graph_forms, alpha, num_iters, seed, out_dir):
    os.makedirs(os.path.join("figs", out_dir), exist_ok=True)
    rng = np.random.default_rng(seed)
    network_seed = int(rng.integers(0, 2**32))
    problem_seed = int(rng.integers(0, 2**32))
    history_z = {}
    estimated_cost = {}

    # Define the same problem for all the graph forms
    local_quadratics, global_quadratic, optimal_z = create_quadratic_problem(num_agents, vars_dim, seed=problem_seed)
    optimal_cost = global_quadratic(optimal_z)
    z0 = rng.random(size=(num_agents, vars_dim))

    # Solve for each graph form
    for graph_form in graph_forms:
        G, A = create_network_of_agents(num_agents, graph_form, seed=network_seed)
        history_z[graph_form] = gradient_tracking(local_quadratics, z0.copy(), A, alpha, num_iters)
        estimated_cost[graph_form] = sum( local_quadratics[i](history_z[graph_form][-1, i]) for i in range(num_agents) )

    print(f"Cost {'optimal':<15}: {optimal_cost:.10f}")
    for graph_form in graph_forms:
        print(f"Cost {graph_form:<15}: {estimated_cost[graph_form]:.10f} | Diff: {abs(estimated_cost[graph_form] - optimal_cost):.10f}")

    def __label_normalize(label):
        return label.replace("_", "-").replace("-graph", "")

    plt.figure(figsize=(8, 5))
    for graph_form in graph_forms:
        plot_loss(local_quadratics, history_z[graph_form], f"{__label_normalize(graph_form)}")
    plt.plot([optimal_cost]*(num_iters+1), "--", label="optimum")
    plt.xlabel("$k$")
    plt.ylabel("$f(z^k)$")
    plt.legend(ncol=3, loc="upper center", columnspacing=0.8, labelspacing=0.25, bbox_to_anchor=(0.4, 1.35))
    plt.savefig(f"figs/{out_dir}/cost.pdf", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    for graph_form in graph_forms:
        plot_gradient(local_quadratics, history_z[graph_form], f"{__label_normalize(graph_form)}")
    plt.xlabel("$k$")
    plt.ylabel("$\\left\\Vert \\nabla f(z^k) \\right\\Vert_2$ (log)")
    plt.legend(ncol=3, loc="upper center", columnspacing=0.8, labelspacing=0.25, bbox_to_anchor=(0.4, 1.35))
    plt.savefig(f"figs/{out_dir}/gradient.pdf", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    for graph_form in graph_forms:
        plot_distance_to_optimum(local_quadratics, history_z[graph_form], optimal_cost, f"{__label_normalize(graph_form)}")
    plt.xlabel("$k$")
    plt.ylabel("Distance to optimum (log)")
    plt.legend(ncol=3, loc="upper center", columnspacing=0.8, labelspacing=0.25, bbox_to_anchor=(0.4, 1.35))
    plt.savefig(f"figs/{out_dir}/distance.pdf", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    for graph_form in graph_forms:
        plt.plot([get_average_consensus_error(z) for z in history_z[graph_form]], label=f"{__label_normalize(graph_form)}")
    plt.xlabel("$k$")
    plt.ylabel("Avg consensus error (log)")
    plt.yscale("log")
    plt.legend(ncol=3, loc="upper center", columnspacing=0.8, labelspacing=0.25, bbox_to_anchor=(0.4, 1.35))
    plt.savefig(f"figs/{out_dir}/consensus.pdf", bbox_inches="tight")
    plt.close()


def quadratic_centralized(num_agents, vars_dim, graph_form, alpha, num_iters, seed, out_dir):
    os.makedirs(os.path.join("figs", out_dir), exist_ok=True)
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

    print(f"Cost {'optimal':<20}: {optimal_cost:.10f}")
    print(f"Cost {'gradient tracking':<20}: {estimated_cost:.10f} | Diff: {abs(estimated_cost - optimal_cost):.10f}")
    print(f"Cost {'centralized gradient':<20}: {estimated_cost_centr:.10f} | Diff: {abs(estimated_cost_centr - optimal_cost):.10f}")

    plt.figure(figsize=(8, 5))
    plot_loss(local_quadratics, history_z, "Distributed")
    plot_loss(global_quadratic, history_z_centr, "Centralized")
    plt.plot([optimal_cost]*(num_iters+1), "--", label="Optimum")
    plt.xlabel("$k$")
    plt.ylabel("$f(z^k)$")
    plt.legend(ncol=2, loc="upper center", columnspacing=0.8, labelspacing=0.25, bbox_to_anchor=(0.4, 1.35))
    plt.savefig(f"figs/{out_dir}/loss.pdf", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plot_gradient(local_quadratics, history_z, "Distributed")
    plot_gradient(global_quadratic, history_z_centr, "Centralized")
    plt.xlabel("$k$")
    plt.ylabel("$\\left\\Vert \\nabla f(z^k) \\right\\Vert_2$ (log)")
    plt.legend(ncol=2, loc="upper center", columnspacing=0.8, labelspacing=0.25, bbox_to_anchor=(0.4, 1.25))
    plt.savefig(f"figs/{out_dir}/gradient.pdf", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plot_distance_to_optimum(local_quadratics, history_z, optimal_cost, "Distributed")
    plot_distance_to_optimum(global_quadratic, history_z_centr, optimal_cost, "Centralized")
    plt.xlabel("$k$")
    plt.ylabel("Distance to optimum (log)")
    plt.legend(ncol=2, loc="upper center", columnspacing=0.8, labelspacing=0.25, bbox_to_anchor=(0.4, 1.25))
    plt.savefig(f"figs/{out_dir}/distance.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # quadratic_comparison(
    #     num_agents = 5,
    #     vars_dim = 3,
    #     graph_forms = ["complete_graph", "binomial_graph", "cycle_graph", "star_graph", "path_graph"],
    #     alpha = 5e-2,
    #     num_iters = 5000,
    #     seed = 42,
    #     out_dir = "5_3"
    # )

    # quadratic_comparison(
    #     num_agents = 5,
    #     vars_dim = 15,
    #     graph_forms = ["complete_graph", "binomial_graph", "cycle_graph", "star_graph", "path_graph"],
    #     alpha = 5e-2,
    #     num_iters = 5000,
    #     seed = 42,
    #     out_dir = "5_15"
    # )

    # quadratic_comparison(
    #     num_agents = 15,
    #     vars_dim = 3,
    #     graph_forms = ["complete_graph", "binomial_graph", "cycle_graph", "star_graph", "path_graph"],
    #     alpha = 5e-2,
    #     num_iters = 5000,
    #     seed = 42,
    #     out_dir = "15_3"
    # )

    # quadratic_comparison(
    #     num_agents = 15,
    #     vars_dim = 15,
    #     graph_forms = ["complete_graph", "binomial_graph", "cycle_graph", "star_graph", "path_graph"],
    #     alpha = 5e-2,
    #     num_iters = 5000,
    #     seed = 42,
    #     out_dir = "15_15"
    # )

    # quadratic_comparison(
    #     num_agents = 30,
    #     vars_dim = 3,
    #     graph_forms = ["complete_graph", "binomial_graph", "cycle_graph", "star_graph", "path_graph"],
    #     alpha = 5e-2,
    #     num_iters = 5000,
    #     seed = 42,
    #     out_dir = "30_3"
    # )

    # quadratic_comparison(
    #     num_agents = 30,
    #     vars_dim = 15,
    #     graph_forms = ["complete_graph", "binomial_graph", "cycle_graph", "star_graph", "path_graph"],
    #     alpha = 5e-2,
    #     num_iters = 5000,
    #     seed = 42,
    #     out_dir = "30_15"
    # )

    quadratic_centralized(
        num_agents = 15,
        vars_dim = 3,
        graph_form = "complete_graph",
        alpha = 5e-2,
        num_iters = 5000,
        seed = 42,
        out_dir = "centralized"
    )