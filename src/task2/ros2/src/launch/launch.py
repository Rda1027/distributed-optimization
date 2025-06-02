import os
import sys
import numpy as np
from launch_ros.actions import Node
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory

sys.path.append( os.path.join(os.path.dirname(__file__), "../../../../../.") )
from imports.scenarios import create_network_of_agents


PACKAGE = "aggregative"

NUM_AGENTS = 5
VARS_DIM = 2

GRAPH_PATTERN = "binomial_graph"
LOSS_TARGET_WEIGHT = 1.0
LOSS_BARYCENTER_WEIGHT = 1.0
AGENTS_IMPORTANCE = [1.0] * NUM_AGENTS

MAX_ITERS = 500
ALPHA = 1e-2

COMMUNICATION_TIME = 1e-1

SEED = 42    
rng = np.random.default_rng(SEED)



def generate_launch_description():
    G, A = create_network_of_agents(NUM_AGENTS, GRAPH_PATTERN, seed=int(rng.integers(0, 2**32)))
    targets_pos = rng.random(size=(NUM_AGENTS, VARS_DIM)) * 10
    z0 = rng.random(size=(NUM_AGENTS, VARS_DIM)) * 10

    to_launch_nodes = []

    # Create the nodes of the agents
    for i in range(NUM_AGENTS):
        neighbors = np.nonzero(A[i])[0]

        to_launch_nodes.append(
            Node(
                package = PACKAGE,
                namespace = f"agent_{i}",
                executable = "agent_aggregative",
                parameters = [{
                    "id": i,
                    "communication_time": COMMUNICATION_TIME,
                    "max_iters": MAX_ITERS,
                    "alpha": ALPHA,
                    "z0": z0[i].tolist(),
                    "neighbors": neighbors.tolist(),
                    "neighbors_weights": A[i, neighbors].tolist(),
                    "target_pos": targets_pos[i].tolist(),
                    "loss_target_weight": LOSS_TARGET_WEIGHT,
                    "loss_barycenter_weight": LOSS_BARYCENTER_WEIGHT,
                    "agent_importance": AGENTS_IMPORTANCE[i],
                }],
                # output = "screen",
                # prefix = f"xterm -title 'agent_{i}' -hold -e",
            )
        )

    # Create the node of the visualizer middleware
    to_launch_nodes.append(
        Node(
            package = PACKAGE,
            namespace = f"visualizer",
            executable = "scenario_visualizer",
            parameters = [
                {
                    "num_agents": NUM_AGENTS,
                    "vars_dim": VARS_DIM,
                    "targets_pos": targets_pos.flatten().tolist(),
                    "max_iters": MAX_ITERS
                }
            ],
            # output = "screen",
            # prefix = f"xterm -title 'visualizer' -hold -e",
        )
    )

    # Create the node of rviz
    rviz_config_dir = get_package_share_directory(PACKAGE)
    rviz_config_file = os.path.join(rviz_config_dir, "rviz_config.rviz")
    to_launch_nodes.append(
        Node(
            package = "rviz2",
            executable = "rviz2",
            arguments = ["-d", rviz_config_file],
        )
    )


    return LaunchDescription(to_launch_nodes)
