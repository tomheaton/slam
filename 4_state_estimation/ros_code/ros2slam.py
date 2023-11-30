#!/usr/bin/env python3
# coding=utf-8

import sys, os
import rclpy
from rclpy.node import Node
from tf_transformations import euler_from_quaternion
import matplotlib.pyplot as plt
import numpy as np

from nav_msgs.msg import Odometry
from eufs_msgs.msg import ConeArrayWithCovariance

from utils import *

# Questions:
# for ros1 we have something that guarantees the subscribers
# receiving a message at the same time, im not sure how to do that here (of if we need it?)

class GraphSLAM(Node):
    def __init__(self):
        super().__init__('graphslam_node')

        # Poses - (x, y, yaw)
        self.poses = np.ndarray((0,3), dtype=np.float32)
        self.poses_covars = np.zeros((0,3,3))

        # NOTE(gonk): Perhaps using separata arrays for blue/yellow cones is better(?)
        # Blue and yellow cones that build the map (x,y,color)
        self.landmarks = np.ndarray((0, 3), dtype=np.float32)
        self.landmarks_covars = np.zeros((0,2,2)) # array of 2x2 matrices

        # Data obtained from observations at the current time
        self.cone_data = {
            ConeTypes.BLUE_CONES          : np.ndarray((0,2), dtype=np.float32),
            ConeTypes.YELLOW_CONES        : np.ndarray((0,2), dtype=np.float32),
            ConeTypes.ORANGE_CONES        : np.ndarray((0,2), dtype=np.float32),
            ConeTypes.BIG_ORANGE_CONES    : np.ndarray((0,2), dtype=np.float32),
            ConeTypes.UNKNOWN_COLOR_CONES : np.ndarray((0,2), dtype=np.float32)
        }
        self.cone_covars = {
            ConeTypes.BLUE_CONES          : np.ndarray((0,2,2), dtype=np.float32),
            ConeTypes.YELLOW_CONES        : np.ndarray((0,2,2), dtype=np.float32),
            ConeTypes.ORANGE_CONES        : np.ndarray((0,2,2), dtype=np.float32),
            ConeTypes.BIG_ORANGE_CONES    : np.ndarray((0,2,2), dtype=np.float32),
            ConeTypes.UNKNOWN_COLOR_CONES : np.ndarray((0,2,2), dtype=np.float32)
        }

        # Make subscribers so that we can get data from bag/eufs
        self.odom_sub  = self.create_subscription(
            Odometry, "/ground_truth/odom", self.odom_callback, 2
        );
        self.cones_sub = self.create_subscription(
            ConeArrayWithCovariance, "/fusion/cones", self.cones_callback, 2
        );
        self.odom_sub   # prevent ununsed variable - from ros2 website
        self.cones_sub  # prevent ununsed variable - from ros2 website

    @classmethod
    def measurement_func(xi, xj):
        """Computes predicted measurement
        Returns homogeneous 4x4 matrix"""
        return np.linalg.inv(se2v2t(xi)) @ se2v2t(xj)

    @classmethod
    def error_func(xi, xj, z):
        """
        Computes difference between predicted measurement and actual measurement
        xi,xj - graph nodes
        z - measurement

        There are various ways to compute the error, what I used here is called the
        'Chordal-Based SE2 Error' - it seemed better because it is less computational
        intensive and eliminates the use of 't2v' multiple times (there are more benefits)
        Essentially makes jacobian easier to compute
        """
        return GraphSLAM.measurement_func(xi, xj).flatten() - se2v2t(z).flatten()

    @classmethod
    def error_func2(xi, xj, z):
        return GraphSLAM


    def odom_callback(self, msg):
        """
        Odometry:
        # Pose
        * Pose pose
            - Point position: float64 x,y,z
            - Quarternion orientation: float x,y,z,w
        * float64[36] covariance - covariance matrix
            - (x,y,z, rot about X, rot about Y, rot about Z)
        NOTE(gonk): the last 3 items of the 6x6 covariane matrix represent
        euler angles and NOT quarternions
        #TwistWithCovariance
        Velocity stuff
        """

        # Get yaw from odom quarternions
        _, _, yaw = euler_from_quaternion(
                [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,\
                 msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        )

        # Append current pose to the poses array
        self.poses = np.vstack((self.poses,
            np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, yaw])
        ))

        # NOTE(gonk): we're using groundtruth for now so the covariance is gonna be 0
        # but this is how we could get the covar matrices for the car poses (not tested)
        """
        # Covariance matrix
        msg_covar = np.array(msg.pose.covariance, dtype=np.float64).reshape(6,6)
        # The covariance matrix from the msg contains more info than we need
        # so we have to extract that information and store it
        # since we need x,y,yaw covariances we need the top-left 2x2 block
        # and the raw/column of yaw
        covar_matrix = msg_covar[:2, :2]
        # NOTE(gonk): since we have euler angles i think yaw is the on the last col/row
        # since it is rotation on the vertical axis - which i think is the z axis of the sim
        # NOTE(gonk): there's probably a nicer way to do this
        covar_matrix = np.vstack((covar_matrix, msg_covar[5, :2]))
        covar_matrix = np.hstack((covar_matrix, [[i] for i in msg_covar[:3, 5]]))
        covar_matrix[2,2] = msg_covar[5,5]
        # Visualisation
        os.system('clear')
        print(f'{msg.pose.covariance=}\n{msg_covar=}\n{covar_matrix=}')
        """

        # Visualisation - Plot
        self.plot()

    def cones_callback(self, msg):
        """
        ConeArrayWithCovariance
        e.g:ConeArrayWithCovariance[] blue_cones
        Point: float64 x,y,z
        float64[4] covariance matrix
        NOTE(gonk): They're 2x2 since they don't have the z
        """

        current_cone_data = {
            ConeTypes.BLUE_CONES          : msg.blue_cones,
            ConeTypes.YELLOW_CONES        : msg.yellow_cones,
            ConeTypes.ORANGE_CONES        : msg.orange_cones,
            ConeTypes.BIG_ORANGE_CONES    : msg.big_orange_cones,
            ConeTypes.UNKNOWN_COLOR_CONES : msg.unknown_color_cones
        }

        for cone_type in current_cone_data.keys():
            # Clear that cone type
            self.cone_data[cone_type] = np.ndarray((0,2), dtype=np.float32)
            self.cone_covars[cone_type] = np.ndarray((0,2,2), dtype=np.float32)
            # Add each cone to that cone type array
            for i, cone in enumerate(current_cone_data[cone_type]):
                # Add cone to observation
                self.cone_data[cone_type] = np.vstack((self.cone_data[cone_type],
                    np.array([cone.point.x, cone.point.y])))
                # Add cone covar to cone_covars
                self.cone_covars[cone_type] = np.vstack((self.cone_covars[cone_type],
                    [np.array(cone.covariance).reshape(2,2)]))

                # EXTREMELY Bad Data association - just for testing
                # Only storing yellow and blue cones for now - for simplicity 
                if (cone_type == ConeTypes.BLUE_CONES or cone_type == ConeTypes.YELLOW_CONES):
                    landmark_point = np.array([cone.point.x, cone.point.y, cone_type.value])

                    # Turn landmark point into world space coordinates instead of relative to car
                    theta = self.poses[-1][2]
                    c, s = np.cos(theta), np.sin(theta)
                    R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))

                    landmark_point = np.dot(R, landmark_point)
                    landmark_point += np.array([self.poses[-1][0],self.poses[-1][1], 0])

                    # Only store it if we haven't seen it before
                    if not self.has_cone_been_seen_before(landmark_point):
                        self.landmarks = np.vstack((self.landmarks, landmark_point))


        # Visualisation
        os.system('clear')
        print(f'blue cones covar len:\n{len(self.cone_covars[ConeTypes.BLUE_CONES])}')
        print(f'blue cone covars:\n{self.cone_covars[ConeTypes.BLUE_CONES]}')

    def has_cone_been_seen_before(self, cone) ->  bool:
        """Worst data association that u could possibly think of - just here to test the rest of the code
        All it does is calculate the distance between the cone and all the cones previously encountered
        and if the distance is smaller than the treshhold then we know that it's the same cone - this
        obviously won't be what we're actually gonna do but I'm just using it to test code
        NOTE(gonk):It doesn't seem to be working as intended tho
        """
        threshold = 1.5

        for c in self.landmarks:
            if c[2] == cone[2] and np.sqrt((c[0]-cone[0])**2 + (c[1] - cone[1])**2) < threshold:
            # if np.sqrt((c[0]-cone[0])**2 + (c[1] - cone[1])**2) < threshold:
                return True
        return False

    def plot(self):
        """Plots plt graph"""
        # Make cone data relative to car
        theta = self.poses[-1][2]
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        # R = np.linalg.inv(np.array(((c, -s), (s, c))))

        cone_data_to_plot = {
            ConeTypes.BLUE_CONES          : np.ndarray((0,2), dtype=np.float32),
            ConeTypes.YELLOW_CONES        : np.ndarray((0,2), dtype=np.float32),
            ConeTypes.ORANGE_CONES        : np.ndarray((0,2), dtype=np.float32),
            ConeTypes.BIG_ORANGE_CONES    : np.ndarray((0,2), dtype=np.float32),
            ConeTypes.UNKNOWN_COLOR_CONES : np.ndarray((0,2), dtype=np.float32)
        }

        for cone_type in ConeTypes:
            for cone in self.cone_data[cone_type]:
                cone = np.dot(R, cone)
                cone += self.poses[-1][:2]
                cone_data_to_plot[cone_type] = np.vstack((cone_data_to_plot[cone_type], cone))

        blue_cones          = cone_data_to_plot[ConeTypes.BLUE_CONES]
        yellow_cones        = cone_data_to_plot[ConeTypes.YELLOW_CONES]
        orange_cones        = cone_data_to_plot[ConeTypes.ORANGE_CONES]
        big_orange_cones    = cone_data_to_plot[ConeTypes.BIG_ORANGE_CONES]
        unknown_color_cones = cone_data_to_plot[ConeTypes.UNKNOWN_COLOR_CONES]

        blue_map = self.landmarks[self.landmarks[:,2] == ConeTypes.BLUE_CONES.value]
        yellow_map = self.landmarks[self.landmarks[:,2] == ConeTypes.YELLOW_CONES.value]

        plt.cla()
        plt.gcf().canvas.mpl_connect(
            'key_release_event', lambda event:
            [exit(0) if event.key == 'escape' else None])

        # -----------
        # POSES - current pose is drawn in purple
        # -----------
        plt.plot(self.poses[:,0], self.poses[:,1], color='black', marker='.', markersize=10, linestyle='None', label='Car Positions')
        plt.plot(self.poses[-1, 0], self.poses[-1,1], color='purple', marker='.', markersize=10, linestyle='None', label='Current Car Pose')

        # -----------
        # CONE OBSERVATIONS
        # -----------
        plt.plot(blue_cones[:,0], blue_cones[:,1], color='blue', marker='.', markersize=10, linestyle='None', label='Blue Cones')
        plt.plot(yellow_cones[:,0], yellow_cones[:,1], color=(0.9, 0.9, 0), marker='.', markersize=10, linestyle='None', label='Yellow Cones')
        plt.plot(orange_cones[:,0], orange_cones[:,1], color=(0.85, 0.6, 0.), marker='.', markersize=10, linestyle='None', label='Orange Cones')
        plt.plot(big_orange_cones[:,0], big_orange_cones[:,1], color=(1., 0.5, 0.0), marker='.', markersize=10, linestyle='None', label='Big Orange Cones')
        plt.plot(unknown_color_cones[:,0], unknown_color_cones[:,1], color=(0.3, 0.3, 0.3), marker='.', markersize=10, linestyle='None', label='Unknown cones')

        # -----------
        # DRAWING PREVIOUS OBSERVATIONS - NOTE(gonk):yellow map drawn green
        # -----------
        plt.plot(blue_map[:,0], blue_map[:,1], color=(0.0, 0.0, 0.4), marker='.', markersize=10, linestyle='None', label='Blue cones in map', alpha=0.5)
        plt.plot(yellow_map[:,0], yellow_map[:,1], color=(0.0, 1.0, 0.0), marker='.', markersize=10, linestyle='None', label='Yellow cones in map', alpha=0.5)

        plt.title('GraphSLAM')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.xlim([-20, 50])
        plt.ylim([-50, 20])

        plt.grid(True)
        plt.pause(0.001)

def main(args=None):
    try:
        # Initialise graphslam node
        rclpy.init(args=args)

        graphslam_node = GraphSLAM()
        graphslam_node.get_logger().info("Started GraphSLAM node...")

        # Run graphslam node continiously
        rclpy.spin(graphslam_node)

        # Clean up the node
        shutdown(graphslam_node)

    # This is just so the ros node is cleaned up if we use CTRL-C to stop the program
    except KeyboardInterrupt:
        shutdown(graphslam_node)
        sys.exit()

def shutdown(node):
        """Cleans up the node and shuts down the rclpy"""
        node.get_logger().info("Shutting down GraphSLAM node!!!")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    # NOTE: Change the path to whatever bagfile u wanna play
    rosbag_path = os.path.expanduser("")
    rosbag_path = os.path.join(os.path.dirname(__file__), '../bagfiles/two_laps.bag')

    # Play bagfile
    os.system(f"ros2 bag play {rosbag_path} &")

    main()
