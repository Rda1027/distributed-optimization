import numpy as np
import matplotlib
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Float64MultiArray as MsgFloat

from .utils import format_message, unpack_message

from typing import Literal


WHITE = (1.0, 1.0, 1.0, 1.0)


class Visualizer(Node):
    def __init__(self):
        super().__init__(
            "visualizer",
            allow_undeclared_parameters = True,
            automatically_declare_parameters_from_overrides = True,
        )

        # Setup
        self.num_agents = self.get_parameter("num_agents").value
        self.vars_dim = self.get_parameter("vars_dim").value
        self.targets_pos = np.array(self.get_parameter("targets_pos").value).reshape(self.num_agents, self.vars_dim)
        self.max_iters = self.get_parameter("max_iters").value
        self.curr_k = 0

        # Setup subscriptions
        self.received_queue = { i: [] for i in range(self.num_agents) }
        for i in range(self.num_agents):
            self.create_subscription( MsgFloat, f"/topic_{i}", self.__rviz_callback, 10 )
        # Setup publishers for rviz
        self.agents_pos_publisher = self.create_publisher(MarkerArray, f"/rviz_agents_topic", 10)
        self.targets_pos_publisher = self.create_publisher(MarkerArray, f"/rviz_targets_topic", 10)
        self.barycenter_pos_publisher = self.create_publisher(Marker, f"/rviz_barycenter_topic", 10)


    def __get_color(self, i):
        return matplotlib.colormaps["tab10"](i % 10)


    def __has_received_from_all_agents(self, k: int):
        for i in range(self.num_agents):
            if (len(self.received_queue[i]) == 0) or (self.received_queue[i][0]["k"] != k): # received_queue is ordered by k for each neighbor
                return False
        return True


    def __create_marker(self, id: int, type: Literal["agent", "target", "barycenter"], position, color):
        marker = Marker()

        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]
        marker.header.frame_id = "map"
        marker.ns = f"{type}_{id}"
        marker.id = id
        match type:
            case "agent": marker.type = Marker.SPHERE
            case "target": marker.type = Marker.CUBE
            case "barycenter": marker.type = Marker.CYLINDER
            case _: marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0

        return marker


    def __rviz_callback(self, msg_raw):
        msg = unpack_message(msg_raw)

        # Check if message is new
        if all( self.received_queue[msg["id"]][i]["k"] != msg["k"] for i in range(len(self.received_queue[msg["id"]])) ):
            self.received_queue[msg["id"]].append( msg )
        
        # Create targets' markers
        target_markers = MarkerArray()
        for i in range(self.num_agents):
            target_markers.markers.append( self.__create_marker(i, "target", self.targets_pos[i], self.__get_color(i)) )
        self.targets_pos_publisher.publish(target_markers)

        if self.__has_received_from_all_agents(self.curr_k):
            self.curr_k += 1
            barycenter = np.mean(np.array([ self.received_queue[i][0]["z"] for i in range(self.num_agents) ]), axis=0)
            agents_markers = MarkerArray()

            # Prepares agents' markers
            for i in range(self.num_agents):
                z = self.received_queue[i][0]["z"]
                agents_markers.markers.append( self.__create_marker(i, "agent", z, self.__get_color(i)) )
                self.received_queue[i].pop(0)

            # Publish markers
            self.agents_pos_publisher.publish( agents_markers )
            self.barycenter_pos_publisher.publish( self.__create_marker(0, "barycenter", barycenter, WHITE) )

            if self.curr_k > self.max_iters:
                self.destroy_node()


def main():
    rclpy.init()

    visualizer = Visualizer()
    try:
        rclpy.spin(visualizer)
    except KeyboardInterrupt:
        print("Visualizer stopped")
    finally:
        rclpy.shutdown()
