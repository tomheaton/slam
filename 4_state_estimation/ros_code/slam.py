#!/usr/bin/env python3
# coding=utf-8

import rospy
import message_filters
# from sensor_msgs.msg import Image, CameraInfo
import math
from tf.transformations import euler_from_quaternion
import matplotlib
matplotlib.use("GTKAgg")
import matplotlib.pyplot as plt
import numpy as np

# Sensor messages
from nav_msgs.msg import Odometry
from eufs_msgs.msg import ConeArrayWithCovariance

# TODO s:
# * get cone covariance matrices
# * Use velocity from audometry data

class SLAM:
    def __init__(self):
        # Data that comes in every frame
        self.map_data = {
            'poses':
            {
                # x,y,yaw
                'points': np.zeros((0,3)),
                # TODO: maybe use a tringular matrix instead? is it faster?
                # TODO: how do we want to represent the covariance matrix?
                # NOTE(gonk): right now it's just an array of arrays of length 36
                'covars': np.zeros((0,36))
            },
            # cones are stored as [array of Xs, array of Ys, 2x2 covar matrix]
            'cones':
            {
                'blue_cones':          [np.zeros(0), np.zeros(0), np.zeros((0,2,2))],
                'yellow_cones':        [np.zeros(0), np.zeros(0), np.zeros((0,2,2))],
                'orange_cones':        [np.zeros(0), np.zeros(0), np.zeros((0,2,2))],
                'big_orange_cones':    [np.zeros(0), np.zeros(0), np.zeros((0,2,2))],
                'unknown_color_cones': [np.zeros(0), np.zeros(0), np.zeros((0,2,2))],
            }
        }


        self.odom_sub = message_filters.Subscriber("/ground_truth/odom", Odometry)
        self.cones_sub = message_filters.Subscriber("/fusion/cones", ConeArrayWithCovariance)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.odom_sub, self.cones_sub], queue_size=1, slop=0.1)
        self.ts.registerCallback(self.callback)

    def callback(self, odom, cones):
        """Callback function"""
        self.get_map_data(odom, cones)
        self.plot()

        return 0

    def get_map_data(self, odom, cones):
        """
        Gets data from ROS messages and stores to self.map_data as numpy arrays
        The messages we get are:
        * Odometry
        * ConeArrayWithCovariance

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

        ConeArrayWithCovariance
        e.g:ConeArrayWithCovariance[] blue_cones
        Point: float64 x,y,z
        float64[4] covariance matrix
        NOTE(gonk): They're 2x2 since they don't have the z

        """
        self.cone_data = {'blue_cones': cones.blue_cones,
                          'yellow_cones': cones.yellow_cones,
                          'orange_cones': cones.orange_cones,
                          'big_orange_cones': cones.big_orange_cones,
                          'unknown_color_cones': cones.unknown_color_cones}

        # Get yaw from odom quarternions
        _, _, yaw = euler_from_quaternion(
                [odom.pose.pose.orientation.x, odom.pose.pose.orientation.y,\
                odom.pose.pose.orientation.z, odom.pose.pose.orientation.w]
        )

        self.map_data['poses']['points'] = np.vstack((
            self.map_data['poses']['points'],
            np.array([odom.pose.pose.position.x, odom.pose.pose.position.y, yaw])
        ))
        # TODO: is using np.matrix better ?
        self.map_data['poses']['covars'] = np.vstack((
            self.map_data['poses']['covars'],
            np.array(odom.pose.covariance, dtype=np.float64),
        ))

        for cone_type in self.cone_data.keys():
            cone_data = self.cone_data[cone_type]
            cone_data_len = len(cone_data)
            cone_x = np.zeros(cone_data_len)
            cone_y = np.zeros(cone_data_len)
            cone_cov = np.zeros((2*cone_data_len, 2))

            for i, cone in enumerate(cone_data):
                cone_x[i] = cone.point.x
                cone_y[i] = cone.point.y
                cone_cov[2*i : 2*i+2, :] = np.array(cone.covariance).reshape((2,2))

            self.map_data['cones'][cone_type] = [cone_x, cone_y, cone_cov]

        # rospy.loginfo(self.map_data['cones']['blue_cones'])

        # print(self.map_data)
        # print('shape: ' + self.map_data['poses'].shape())
        print(self.map_data['poses']['points'].shape)

    def plot(self):
        """Plots plt graph"""
        car_poses = self.map_data['poses']['points']
        blue_cones = self.map_data['cones']['blue_cones']
        yellow_cones = self.map_data['cones']['yellow_cones']

        theta = car_poses[-1][2]
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))

        # Make cone data relative to car
        # By first rotating the points
        blue_cones[0], blue_cones[1] = np.dot(R, np.vstack((blue_cones[0], blue_cones[1])))
        yellow_cones[0], yellow_cones[1] = np.dot(R, np.vstack((yellow_cones[0], yellow_cones[1])))

        # And then adding to the car coords
        blue_cones[0] += car_poses[-1][0]
        blue_cones[1] += car_poses[-1][1]
        yellow_cones[0] += car_poses[-1][0]
        yellow_cones[1] += car_poses[-1][1]

        plt.cla()
        plt.gcf().canvas.mpl_connect(
            'key_release_event', lambda event:
            [exit(0) if event.key == 'escape' else None])

        car_plot, = plt.plot(car_poses[:,0], car_poses[:,1], color='black', marker='.', markersize=10, linestyle='None', label='Car Positions')

        blue, = plt.plot(blue_cones[0], blue_cones[1], color='blue', marker='.', markersize=10, linestyle='None', label='Blue Cone Position')
        yellow, = plt.plot(yellow_cones[0], yellow_cones[1], \
                           color=(0.9, 0.9, 0), marker='.', markersize=10, linestyle='None', label='Yellow Cones')


        plt.title('GraphSLAM')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.xlim([-20, 50])
        plt.ylim([-50, 20])

        plt.grid(True)
        plt.pause(0.001)




def main():
    """Initialise GraphSLAM algorithm with input data from EUFS Simulator"""
    rospy.init_node('graphslam_node')
    localization = SLAM()
    try:
        rospy.loginfo("Started GraphSLAM...")
        rospy.spin()
    except rospy.ROSInterruptException:
        # Shutdown
        rospy.loginfo("Shutting down GraphSLAM!")


if __name__ == "__main__":
    main()
