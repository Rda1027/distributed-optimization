import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from imports.algorithm import gradient_tracking, gradient_descent
from imports.loss import QuadraticFunction, TargetLocalizationLoss
from imports.scenarios import create_network_of_agents, create_quadratic_problem, create_position_tracking_problem
from imports.utils import get_average_estimate_error, get_average_consensus_error
from imports.plot import plot_scenario, plot_animation, \
    plot_loss_tracking as plot_loss, \
    plot_gradient_tracking as plot_gradient
    
plt.rcParams["font.family"] = "cmr10"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["axes.formatter.use_mathtext"] = True
plt.rcParams["font.size"] = 26
plt.rcParams["legend.fontsize"] = 16



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



def tracking_comparison(num_robots, num_targets, vars_dim, graph_forms, alpha, num_iters, noise_args, seed, out_dir):
    """
        Experiment to compare the algorithm with different graphs.
    """
    os.makedirs(os.path.join("figs", out_dir), exist_ok=True)
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

    plt.figure(figsize=(8, 5.5))
    for graph_form in graph_forms:
        plot_loss(local_losses, history_z[graph_form], f"{graph_form.replace('_', '-')}")
    plt.xlabel("$k$")
    plt.ylabel("$l(z^k)$ (log)")
    plt.legend()
    plt.savefig(f"figs/{out_dir}/loss.pdf", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5.5))
    for graph_form in graph_forms:
        plot_gradient(local_losses, history_z[graph_form], f"{graph_form.replace('_', '-')}")
    plt.xlabel("$k$")
    plt.ylabel("$\\left\\Vert \\nabla l(z^k) \\right\\Vert_2$ (log)")
    plt.legend()
    plt.savefig(f"figs/{out_dir}/gradient.pdf", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5.5))
    for graph_form in graph_forms:
        plt.plot([get_average_consensus_error(z) for z in history_z[graph_form]], label=f"{graph_form.replace('_', '-')}")
    plt.xlabel("$k$")
    plt.ylabel("Average consensus error (log)")
    plt.yscale("log")
    plt.legend()
    plt.savefig(f"figs/{out_dir}/consensus.pdf", bbox_inches="tight")
    plt.close()


def tracking_centralized(num_robots, num_targets, vars_dim, graph_form, alpha, num_iters, noise_args, seed, out_dir):
    """
        Experiment to compare the algorithm with the centralized gradient method.
    """
    os.makedirs(os.path.join("figs", out_dir), exist_ok=True)
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
    history_z_centr = history_z_centr.reshape(-1, num_targets, vars_dim)

    print(
        f"Loss {'gradient tracking':<20}: {sum( local_losses[i](history_z[-1, i].flatten()) for i in range(num_robots) ):.10f}"
        f" | Average estimated distance error: {get_average_estimate_error(history_z[-1], targets_pos_real):.10f}"
    )
    print(
        f"Loss {'centralized gradient':<20}: {sum( local_losses[i](history_z_centr[-1].flatten()) for i in range(num_robots) ):.10f}"
        f" | Average estimated distance error: {get_average_estimate_error(np.expand_dims(history_z_centr[-1], 0), targets_pos_real):.10f}"
    )

    plt.figure(figsize=(8, 5.5))
    plot_loss(local_losses, history_z, f"Gradient tracking")
    plot_loss(local_losses, history_z_centr, f"Centralized gradient")
    plt.xlabel("$k$")
    plt.ylabel("$l(z^k)$ (log)")
    plt.legend()
    plt.savefig(f"figs/{out_dir}/loss.pdf", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5.5))
    plot_gradient(local_losses, history_z, f"Gradient tracking")
    plot_gradient(local_losses, history_z_centr, f"Centralized gradient")
    plt.xlabel("$k$")
    plt.ylabel("$\\left\\Vert \\nabla l(z^k) \\right\\Vert_2$ (log)")
    plt.legend()
    plt.savefig(f"figs/{out_dir}/gradient.pdf", bbox_inches="tight")
    plt.close()
    

def tracking_noise(num_robots, num_targets, vars_dim, graph_form, alpha, num_iters, noise_type, noise_args_list, seed, out_dir):
    """
        Experiment to compare the algorithm with the centralized gradient method.
    """
    os.makedirs(os.path.join("figs", out_dir), exist_ok=True)
    rng = np.random.default_rng(seed)
    network_seed = int(rng.integers(0, 2**32))
    problem_seed = int(rng.integers(0, 2**32))
    history_z_list = []
    local_losses_list = []
    
    match noise_type:
        case "gaussian":
            labels = [f"{noise_args['noise_ratio']} x N({noise_args['gaussian_mean']}, {noise_args['gaussian_std']}^2)" for noise_args in noise_args_list]
            labels_math = [f"${noise_args['noise_ratio']} \\times \\mathcal{{N}}({noise_args['gaussian_mean']}, {noise_args['gaussian_std']}^2)$" for noise_args in noise_args_list]
        case "poisson":
            labels = [f"{noise_args['noise_ratio']} x Pois({noise_args['poisson_lambda']})" for noise_args in noise_args_list]
            labels_math = [f"${noise_args['noise_ratio']} \\times \\text{{Pois}}({noise_args['poisson_lambda']})$" for noise_args in noise_args_list]


    G, A = create_network_of_agents(num_robots, graph_form, seed=network_seed)
    z0 = rng.random(size=(num_robots, num_targets*vars_dim))

    for noise_args in noise_args_list:
        local_losses, global_loss, (robots_pos, targets_pos_real, est_targets_dists) = create_position_tracking_problem(
            num_robots = num_robots,
            num_targets = num_targets,
            vars_dim = vars_dim,
            seed = problem_seed,
            noise_type = noise_type,
            **noise_args
        )

        history_z = gradient_tracking(local_losses, z0.copy(), A, alpha, num_iters)
        history_z = history_z.reshape(-1, num_robots, num_targets, vars_dim)

        history_z_list.append(history_z)
        local_losses_list.append(local_losses)

    for j in range(len(noise_args_list)):
        print(
            f"{f'[{labels[j]}]':<25} Loss: {sum( local_losses_list[j][i](history_z_list[j][-1, i].flatten()) for i in range(num_robots) ):.10f}"
            f" | Average estimated distance error: {get_average_estimate_error(history_z_list[j][-1], targets_pos_real):.10f}"
        )

    plt.figure(figsize=(8, 5.5))
    for i in range(len(noise_args_list)):
        plot_loss(local_losses_list[i], history_z_list[i], f"{labels_math[i]}")
    plt.xlabel("$k$")
    plt.ylabel("$l(z^k)$ (log)")
    plt.legend()
    plt.savefig(f"figs/{out_dir}/loss.pdf", bbox_inches="tight")
    plt.close()


    plt.figure(figsize=(8, 5.5))
    for i in range(len(noise_args_list)):
        plot_gradient(local_losses_list[i], history_z_list[i], f"{labels_math[i]}")
    plt.xlabel("$k$")
    plt.ylabel("$\\left\\Vert \\nabla l(z^k) \\right\\Vert_2$ (log)")
    plt.legend()
    plt.savefig(f"figs/{out_dir}/gradient.pdf", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5.5))
    for i in range(len(noise_args_list)):
        plt.plot([get_average_consensus_error(z) for z in history_z_list[i]], label=f"{labels_math[i]}")
    plt.xlabel("$k$")
    plt.ylabel("Average consensus error (log)")
    plt.yscale("log")
    plt.legend()
    plt.savefig(f"figs/{out_dir}/consensus.pdf", bbox_inches="tight")
    plt.close()


def tracking_noise_cancel(num_robots_list, num_targets, vars_dim, graph_form, alpha, num_iters, noise_args, num_runs, seed, out_dir):
    """
        Experiment to compare the algorithm with different graphs.
    """
    os.makedirs(os.path.join("figs", out_dir), exist_ok=True)
    rng = np.random.default_rng(seed)
    avg_errors = { num_robots: [] for num_robots in num_robots_list }

    for num_robots in num_robots_list:
        for _ in range(num_runs):
            network_seed = int(rng.integers(0, 2**32))
            G, A = create_network_of_agents(num_robots, graph_form, seed=network_seed)
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
            avg_errors[num_robots].append( get_average_estimate_error(history_z[-1], targets_pos_real) )

    # avg_errors[5] = [0.14538857979853864, 0.22011508054269316, 0.022209494157823563, 0.010105849184295605, 0.032827508560079134]
    # avg_errors[10] = [0.05386962789161569, 0.002399417010700353, 0.011834966346604543, 0.013243467544485178, 0.04437449405645583]
    # avg_errors[20] = [0.022799214238276015, 0.022358734087471317, 0.021496782304989807, 0.02279178908842746, 0.022168612780828486]
    # avg_errors[30] = [0.00887548274204436, 0.05063906241620565, 0.02200728466410506, 0.022970892701172, 0.020384202368753396]
    # avg_errors[40] = [0.017486975114028254, 0.035248185038051986, 0.02569937707384771, 0.03326274683235674, 0.016224513112034607]
    # avg_errors[50] = [0.01593948935325988, 0.018958392834824807, 0.038610031341481145, 0.03313471814676305, 0.016121926085157733]
    # avg_errors[60] = [0.007999930741731725, 0.004284004000857931, 0.006618230172244889, 0.02449810968971262, 0.021153314554882174]
    # avg_errors[70] = [0.008281488226116565, 0.007771293943549792, 0.009663824595218863, 0.017037109542120933, 0.02266034236197115]
    # avg_errors[80] = [0.021168837832657595, 0.020827590781402915, 0.006352009146178313, 0.011631328333029716, 0.007788002854093032]
    # avg_errors[90] = [0.0225750644770729, 0.005691150806330921, 0.027846122980458136, 0.009885289627167358, 0.008632246578248199]
    # avg_errors[100] = [0.010309475013145074, 0.013785708678175851, 0.011551678894983469, 0.012653024588578365, 0.00673816591088237]

    plt.figure(figsize=(8, 5.5))
    plt.errorbar(
        list(sorted(avg_errors.keys())), 
        [ np.mean(avg_errors[num_robots]) for num_robots in sorted(avg_errors.keys()) ], 
        [ np.std(avg_errors[num_robots]) for num_robots in sorted(avg_errors.keys()) ], 
        marker = "o",
        ecolor = "#2596be90"
    )
    plt.xlabel("Num robots")
    plt.ylabel("Average tracking error (log)")
    plt.yscale("log")
    plt.savefig(f"figs/{out_dir}/avg_tracking.pdf", bbox_inches="tight")



if __name__ == "__main__":
    # tracking_comparison(
    #     num_robots = 5,
    #     num_targets = 1,
    #     vars_dim = 2,
    #     graph_forms = ["complete_graph", "binomial_graph", "cycle_graph", "star_graph", "path_graph"],
    #     alpha = 1e-2,
    #     num_iters = 10000,
    #     noise_args = {
    #         "noise_ratio": 0.05,
    #         "noise_type": "gaussian",
    #         "gaussian_mean": 0.0,
    #         "gaussian_std": 1.0,
    #     },
    #     seed = 42,
    #     out_dir = "5_1_2"
    # )

    # tracking_comparison(
    #     num_robots = 5,
    #     num_targets = 3,
    #     vars_dim = 2,
    #     graph_forms = ["complete_graph", "binomial_graph", "cycle_graph", "star_graph", "path_graph"],
    #     alpha = 1e-2,
    #     num_iters = 10000,
    #     noise_args = {
    #         "noise_ratio": 0.05,
    #         "noise_type": "gaussian",
    #         "gaussian_mean": 0.0,
    #         "gaussian_std": 1.0,
    #     },
    #     seed = 42,
    #     out_dir = "5_3_2"
    # )

    # tracking_comparison(
    #     num_robots = 15,
    #     num_targets = 3,
    #     vars_dim = 2,
    #     graph_forms = ["complete_graph", "binomial_graph", "cycle_graph", "star_graph", "path_graph"],
    #     alpha = 1e-2,
    #     num_iters = 10000,
    #     noise_args = {
    #         "noise_ratio": 0.05,
    #         "noise_type": "gaussian",
    #         "gaussian_mean": 0.0,
    #         "gaussian_std": 1.0,
    #     },
    #     seed = 42,
    #     out_dir = "15_3_2"
    # )

    tracking_comparison(
        num_robots = 30,
        num_targets = 3,
        vars_dim = 2,
        graph_forms = ["complete_graph", "binomial_graph", "cycle_graph", "star_graph", "path_graph"],
        alpha = 1e-2,
        num_iters = 10000,
        noise_args = {
            "noise_ratio": 0.05,
            "noise_type": "gaussian",
            "gaussian_mean": 0.0,
            "gaussian_std": 1.0,
        },
        seed = 42,
        out_dir = "30_3_2"
    )

    # tracking_centralized(
    #     num_robots = 15,
    #     num_targets = 3,
    #     vars_dim = 2,
    #     graph_form = "complete_graph",
    #     alpha = 1e-2,
    #     num_iters = 10000,
    #     noise_args = {
    #         "noise_ratio": 0.05,
    #         "noise_type": "gaussian",
    #         "gaussian_mean": 0.0,
    #         "gaussian_std": 1.0,
    #     },
    #     seed = 42,
    #     out_dir = "centralized"
    # )

    # tracking_noise(
    #         num_robots = 15,
    #         num_targets = 1,
    #         vars_dim = 2,
    #         graph_form = "complete_graph",
    #         alpha = 1e-2,
    #         num_iters = 5000,
    #         noise_type = "gaussian",
    #         noise_args_list = [
    #             { "noise_ratio": 0.1, "gaussian_mean": 0.0, "gaussian_std": 0.5 },
    #             { "noise_ratio": 0.1, "gaussian_mean": 0.0, "gaussian_std": 1.0 },
    #             { "noise_ratio": 0.1, "gaussian_mean": 0.0, "gaussian_std": 1.5 },
    #             { "noise_ratio": 0.1, "gaussian_mean": 1.0, "gaussian_std": 0.5 },
    #             { "noise_ratio": 0.1, "gaussian_mean": 1.0, "gaussian_std": 1.0 },
    #             { "noise_ratio": 0.1, "gaussian_mean": 1.0, "gaussian_std": 1.5 },
    #         ],
    #         seed = 42,
    #         out_dir = "gaussian"
    #     )

    # tracking_noise(
    #         num_robots = 15,
    #         num_targets = 1,
    #         vars_dim = 2,
    #         graph_form = "complete_graph",
    #         alpha = 1e-2,
    #         num_iters = 5000,
    #         noise_type = "poisson",
    #         noise_args_list = [
    #             { "noise_ratio": 0.1, "poisson_lambda": 1.0 },
    #             { "noise_ratio": 0.1, "poisson_lambda": 4.0 },
    #             { "noise_ratio": 0.1, "poisson_lambda": 10.0 },
    #         ],
    #         seed = 42,
    #         out_dir = "poisson"
    #     )
        
    # tracking_noise(
    #         num_robots = 15,
    #         num_targets = 1,
    #         vars_dim = 2,
    #         graph_form = "complete_graph",
    #         alpha = 1e-2,
    #         num_iters = 5000,
    #         noise_type = "gaussian",
    #         noise_args_list = [
    #             { "noise_ratio": 0.01, "gaussian_mean": 0.0, "gaussian_std": 1.0 },
    #             { "noise_ratio": 0.05, "gaussian_mean": 0.0, "gaussian_std": 1.0 },
    #             { "noise_ratio": 0.1, "gaussian_mean": 0.0, "gaussian_std": 1.0 },
    #             { "noise_ratio": 0.2, "gaussian_mean": 0.0, "gaussian_std": 1.0 },
    #             { "noise_ratio": 0.5, "gaussian_mean": 0.0, "gaussian_std": 1.0 },
    #         ],
    #         seed = 42,
    #         out_dir = "rates"
    #     )


    # tracking_noise_cancel(
    #     num_robots_list = [],
    #     num_targets = 1,
    #     vars_dim = 2,
    #     graph_form = "complete_graph",
    #     alpha = 1e-2,
    #     num_iters = 5000,
    #     noise_args = { 
    #         "noise_type": "gaussian",
    #         "noise_ratio": 0.05, 
    #         "gaussian_mean": 0.0, 
    #         "gaussian_std": 1.0 
    #     },
    #     num_runs = 5,
    #     seed = 42,
    #     out_dir = "average"
    # )