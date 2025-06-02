import os
import sys
import numpy as np
from time import sleep
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray as MsgFloat

from .utils import format_message, unpack_message

sys.path.append( os.path.join(os.path.dirname(__file__), "../../../.") )
from imports.loss import AggregativeLoss, Linear
from imports.algorithm import aggregative_step



class Agent(Node):
    def __init__(self):
        super().__init__(
            "aggregative_agent",
            allow_undeclared_parameters = True,
            automatically_declare_parameters_from_overrides = True,
        )
        # Configuration parameters
        self.id                 = self.get_parameter("id").value
        communication_time      = self.get_parameter("communication_time").value
        self.max_iters          = self.get_parameter("max_iters").value
        self.alpha              = self.get_parameter("alpha").value
        self.neighbors          = self.get_parameter("neighbors").value
        self.neighbors_weights  = np.array(self.get_parameter("neighbors_weights").value)
        # Loss parameters
        target_pos              = np.array(self.get_parameter("target_pos").value)
        loss_target_weight      = self.get_parameter("loss_target_weight").value
        loss_barycenter_weight  = self.get_parameter("loss_barycenter_weight").value
        agent_importance        = self.get_parameter("agent_importance").value
        
        # Optimization parameters
        self.curr_k     = 0
        self.phi        = Linear(agent_importance)
        self.loss       = AggregativeLoss(target_pos, self.phi, loss_target_weight, loss_barycenter_weight)
        self.curr_z     = np.array(self.get_parameter("z0").value)
        self.curr_sigma = self.phi(self.curr_z) # s: estimate of the global barycenter position
        self.curr_grad2 = self.loss.grad2(self.curr_z, self.curr_sigma) # v: estimate of the gradient of the loss w.r.t. the second parameter
        self.vars_dim   = len(self.curr_z)

        # Topics initialization
        self.received_queue = {j: [] for j in self.neighbors}
        self.publisher = self.create_publisher(MsgFloat, f"/topic_{self.id}", 10)
        for j in self.neighbors:
            self.create_subscription(MsgFloat, f"/topic_{j}", self.__neighbors_topic_callback, 10)

        # Communication polling
        self.timer = self.create_timer(communication_time, self.__communication_callback)

        print(f"I am agent: {self.id:d}")


    def __send_state(self):
        self.publisher.publish(
            format_message(self.id, self.curr_k, self.curr_sigma, self.curr_grad2, self.curr_z)
        )


    def __neighbors_topic_callback(self, msg_raw):
        msg = unpack_message(msg_raw)
        
        # Check if already received
        if any( self.received_queue[msg["id"]][i]["k"] == msg["k"] for i in range(len(self.received_queue[msg["id"]])) ):
            return

        self.received_queue[msg["id"]].append( msg )


    def __has_received_from_all_neighbors(self, k: int):
        for i in self.neighbors:
            if (len(self.received_queue[i]) == 0) or (self.received_queue[i][0]["k"] != k): # received_queue is ordered by k for each neighbor
                return False
        return True


    def __communication_callback(self):
        if self.curr_k == 0:
            self.__send_state()
            self.curr_k += 1
        elif self.__has_received_from_all_neighbors(self.curr_k-1):
            self.curr_z, self.curr_sigma, self.curr_grad2 = aggregative_step(
                z_i = self.curr_z,
                s_i = self.curr_sigma,
                v_i = self.curr_grad2,
                alpha = self.alpha,
                loss = self.loss,
                phi = self.phi,
                adj_neighbors = self.neighbors_weights,
                s_neighbors = [ self.received_queue[i][0]["sigma_est"] for i in self.neighbors ],
                v_neighbors = [ self.received_queue[i][0]["grad2_est"] for i in self.neighbors ],
            )

            for i in self.neighbors:
                self.received_queue[i].pop(0)

            self.__send_state()

            self.curr_k += 1
            if self.curr_k > self.max_iters:
                self.destroy_node()


def main():
    rclpy.init()

    agent = Agent()
    sleep(1) # For synchronization
    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        print("Agent stopped")
    finally:
        rclpy.shutdown()