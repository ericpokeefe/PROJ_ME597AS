#!/usr/bin/env python3

import sys
import os
import numpy as np
import math
import yaml

import time

import rospy
import tf
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Pose, Twist

from map_astar import MapProcessor, AStar

class Navigation:
    """! Navigation node class.
    This class should serve as a template to implement the path planning and
    path follower components to move the turtlebot from position A to B.
    """

    def __init__(self, node_name='Navigation'):
        """! Class constructor.
        @param  None.
        @return An instance of the Navigation class.
        """
        # ROS related variables
        self.node_name = node_name
        self.rate = 0

        # Path planner/follower related variables
        self.path = Path()
        self.goal_pose = PoseStamped()
        self.ttbot_pose = PoseStamped()

        # Covarance check
        self.cov_check = False

        # From Jupyter Notebook
        self.mp = MapProcessor('/home/ericokeefe/PycharmProjects/PROJ_ME597AS/src/final_project/maps/map')
        self.mp.inflate_map(self.mp.rect_kernel(6, 1))
        self.mp.get_graph_from_map()

    def init_app(self):
        """! Node intialization.
        @param  None
        @return None.
        """
        # ROS node initialization
        rospy.init_node(self.node_name, anonymous=True)
        self.rate = rospy.Rate(30)

        # Subscribers
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.__goal_pose_cbk, queue_size=1)
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.__ttbot_pose_cbk, queue_size=1)

        # Publishers
        self.path_pub = rospy.Publisher('global_plan', Path, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.show_path_pub = rospy.Publisher('astar_plan', Path, queue_size=1)

    def __goal_pose_cbk(self, data):
        """! Callback to catch the goal pose.
        @param  data    PoseStamped object from RVIZ.
        @return None.
        """
        self.goal_pose = data
        rospy.loginfo('goal_pose:{:.4f},{:.4f}'.format(self.goal_pose.pose.position.x, self.goal_pose.pose.position.y))

    def __ttbot_pose_cbk(self, data):
        """! Callback to catch the position of the vehicle.
        @param  data    PoseWithCovarianceStamped object from amcl.
        @return None.
        """
        self.ttbot_pose = data.pose
        cov = data.pose.covariance
        #rospy.loginfo('ttbot_pose:{:.4f}'.format(self.ttbot_pose.pose.position.x, self.ttbot_pose.pose.position.y))
        #rospy.loginfo('ttbot_pose:{}'.format(cov))
        self.cov_check = True
        for i in cov:
            if abs(i) > 0.05:
                self.cov_check = False
                break

    def a_star_path_planner(self, start_pose, end_pose):
        """! A Star path planner.
        @param  start_pose    PoseStamped object containing the start of the path to be created.
        @param  end_pose      PoseStamped object containing the end of the path to be created.
        @return path          Path object containing the sequence of waypoints of the created path.
        """

        path = Path()

        # log info
        print("A* running")
        #rospy.loginfo('A* planner.\n> start:{} \nyaw:{} \n> end:{} \nyaw:{}'.format(start_pose.pose.position,
        #                                                                            start_yaw, end_pose.pose.position,
        #                                                                            end_yaw))

        # Convert pose to pixels
        x_offset = 95
        y_offset = 102

        x_conv = -0.1
        y_conv = 0.1

        start_pix_x = int(start_pose.pose.position.y / x_conv) + x_offset
        start_pix_y = int(start_pose.pose.position.x / y_conv) + y_offset
        end_pix_x = int(end_pose.pose.position.y / x_conv) + x_offset
        end_pix_y = int(end_pose.pose.position.x / y_conv) + y_offset

        print(start_pix_x, start_pix_y, end_pix_x, end_pix_y)

        # set root and end points
        self.mp.map_graph.root = str(start_pix_x) + "," + str(start_pix_y)
        self.mp.map_graph.end = str(end_pix_x) + "," + str(end_pix_y)
        print(self.mp.map_graph.root, self.mp.map_graph.end)

        # check to ensure start and end points are available
        if (self.mp.map_graph.root not in self.mp.map_graph.g) or (self.mp.map_graph.end not in self.mp.map_graph.g):
            print("pick a new spot\n\n")
            return

        # From Jupyter Notebook
        # Run AStar algo
        as_maze = AStar(self.mp.map_graph)
        start = time.time()
        as_maze.solve(self.mp.map_graph.g[self.mp.map_graph.root], self.mp.map_graph.g[self.mp.map_graph.end])
        end = time.time()
        print('Elapsed Time: %.3f' % (end - start))
        path_as, dist_as = as_maze.reconstruct_path(self.mp.map_graph.g[self.mp.map_graph.root],
                                                    self.mp.map_graph.g[self.mp.map_graph.end])

        # generates useful path
        for point in path_as:
            pix_x, pix_y = (int(i) for i in point.split(','))

            # convert pixels to pose
            pose_x = (pix_y - x_offset) * x_conv
            pose_y = (pix_x - y_offset) * y_conv

            point_pose = PoseStamped()
            point_pose.header.frame_id = 'map'
            point_pose.pose.position.x = (-1 * pose_x) - 0.7
            point_pose.pose.position.y = (-1 * pose_y) - 0.7

            path.poses.append(point_pose)

        print('path generated successfully. YAY!')
        print(path)
        print(self.mp.map_graph.root, self.mp.map_graph.end)
        print("a* complete\n\n")

        path.header.frame_id = 'map'

        # publish path for Rviz to display on screen
        self.show_path_pub.publish(path)

        return path

    def get_path_idx(self, path, vehicle_pose):
        """! Path follower.
        @param  path             Path object containing the sequence of waypoints of the created path.
        @param  vehicle_pose     PoseStamped object containing the current vehicle position.
        @return idx              Position int the path pointing to the next goal pose to follow.
        """

        wp_dist_flag = 100

        for i, waypoint in enumerate(path.poses):
            wp_x = waypoint.pose.position.x
            wp_y = waypoint.pose.position.y
            bot_x = vehicle_pose.pose.position.x
            bot_y = vehicle_pose.pose.position.y

            wp_dist = (wp_x - bot_x)**2 + (wp_y - bot_y)**2 # Squared distance, runs faster

            if wp_dist <= wp_dist_flag:
                wp_dist_flag = wp_dist
                idx = i

        # set next waypoint as target
        if idx + 1 < len(path.poses):
            idx = idx + 1
        print("target waypoint:", idx)

        return idx

    def path_follower(self, vehicle_pose, current_goal_pose):
        """! Path follower.
        @param  vehicle_pose           PoseStamped object containing the current vehicle pose.
        @param  current_goal_pose      PoseStamped object containing the current target from the created path. This is different from the global target.
        @return path                   Path object containing the sequence of waypoints of the created path.
        """

        wp_x = current_goal_pose.pose.position.x
        wp_y = current_goal_pose.pose.position.y
        bot_x = vehicle_pose.pose.position.x
        bot_y = vehicle_pose.pose.position.y

        wp_dist = np.sqrt((wp_x - bot_x) ** 2 + (wp_y - bot_y) ** 2)

        dx = wp_x - bot_x
        dy = wp_y - bot_y

        heading = np.arctan2(dy, dx)

        if wp_dist < 0.05:
            speed = 0
        else:
            speed = min(0.1, wp_dist)

        return speed, heading

    def move_ttbot(self, speed, heading):
        """! Function to move turtlebot passing directly a heading angle and the speed.
        @param  speed     Desired yaw angle.
        @param  heading   Desired speed.
        @return path      object containing the sequence of waypoints of the created path.
        """
        cmd_vel = Twist()

        # calculate current turtle bot orientation(yaw0
        bot_pose = self.ttbot_pose

        bot_quat = (bot_pose.pose.orientation.x, bot_pose.pose.orientation.y, bot_pose.pose.orientation.z,
                    bot_pose.pose.orientation.w)
        bot_euler = tf.transformations.euler_from_quaternion(bot_quat)
        bot_yaw = bot_euler[2]

        # calculate steering angle
        steering_angle = 2 * (heading - bot_yaw)

        cmd_vel.linear.x = speed
        cmd_vel.angular.z = steering_angle

        self.cmd_vel_pub.publish(cmd_vel)

    def spin_ttbot(self, ang_vel):
        """! Function to spin turtlebot to find initial pose.
        @param  ang_vel   angular velocity
        """

        cmd_vel = Twist()
        cmd_vel.linear.x = 0
        cmd_vel.angular.z = ang_vel

        self.cmd_vel_pub.publish(cmd_vel)

    def run(self):
        """! Main loop of the node. You need to wait until a new pose is published, create a path and then
        drive the vehicle towards the final pose.
        @param none
        @return none
        """

        '''
            Main loop
        '''
        path_complete = False
        goal_flag = self.goal_pose
        timeout = False
        path = None

        while not rospy.is_shutdown():
            # 0. Localize robot initial position
            if self.cov_check == False:
                # spin turtlebot to find initial pose
                print('cov_check failed\n\n')
                self.spin_ttbot(0.5)
                continue

            # 1. Create the path to follow
            if goal_flag != self.goal_pose:
                # passed cov_check
                print('passed cov_check')
                self.move_ttbot(0, 0)

                print('received goal')

                # Run A* path planning
                path = self.a_star_path_planner(self.ttbot_pose, self.goal_pose)

            # Reset goal_flag to current goal pose
            goal_flag = self.goal_pose

            # if no path is returned, stop moving
            if path is None:
                self.move_ttbot(0, 0)
                continue

            # Find next point to go towards and set current goal
            idx = self.get_path_idx(path, self.ttbot_pose)
            current_goal = path.poses[idx]

            # Route to waypoint (speed and heading)
            speed, heading = self.path_follower(self.ttbot_pose, current_goal)

            # move ttbot to waypoint
            self.move_ttbot(speed, heading)

            self.rate.sleep()
        rospy.signal_shutdown("[{}] Finished Cleanly".format(self.name))


if __name__ == "__main__":
    nav = Navigation(node_name='Navigation')
    nav.init_app()
    try:
        nav.run()
    except rospy.ROSInterruptException:
        print("program interrupted before completion", file=sys.stderr)