#!/usr/bin/env python3

import sys
import os
import numpy as np
import math
import yaml
import rospkg

import time

from PIL import Image, ImageOps

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from graphviz import Graph

from copy import copy, deepcopy

import rospy
import tf
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Pose, Twist

import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image as cvImage

class Map():
    def __init__(self, map_name):
        self.map_im, self.map_df, self.limits = self.__open_map(map_name)
        self.image_array = self.__get_obstacle_map(self.map_im, self.map_df)

    def __repr__(self):
        fig, ax = plt.subplots(dpi=150)
        ax.imshow(self.image_array, extent=self.limits, cmap=cm.gray)
        ax.plot()
        return ""

    def __open_map(self, map_name):
        # Open the YAML file which contains the map name and other
        # configuration parameters
        f = open(map_name + '.yaml', 'r')
        map_df = pd.json_normalize(yaml.safe_load(f))
        # Open the map image
        #map_name = map_df.image[0]
        im = Image.open(map_name + '.pgm')
        size = 608, 384
        im.thumbnail(size)
        im = ImageOps.grayscale(im)
        # Get the limits of the map. This will help to display the map
        # with the correct axis ticks.
        xmin = map_df.origin[0][0]
        xmax = map_df.origin[0][0] + im.size[0] * map_df.resolution[0]
        ymin = map_df.origin[0][1]
        ymax = map_df.origin[0][1] + im.size[1] * map_df.resolution[0]

        return im, map_df, [xmin, xmax, ymin, ymax]

    def __get_obstacle_map(self, map_im, map_df):
        img_array = np.reshape(list(self.map_im.getdata()), (self.map_im.size[1], self.map_im.size[0]))
        up_thresh = self.map_df.occupied_thresh[0] * 255
        low_thresh = self.map_df.free_thresh[0] * 255

        for j in range(self.map_im.size[0]):
            for i in range(self.map_im.size[1]):
                if img_array[i, j] > up_thresh:
                    img_array[i, j] = 255
                else:
                    img_array[i, j] = 0
        return img_array


class Queue():
    def __init__(self, init_queue=[]):
        self.queue = copy(init_queue)
        self.start = 0
        self.end = len(self.queue) - 1

    def __len__(self):
        numel = len(self.queue)
        return numel

    def __repr__(self):
        q = self.queue
        tmpstr = ""
        for i in range(len(self.queue)):
            flag = False
            if (i == self.start):
                tmpstr += "<"
                flag = True
            if (i == self.end):
                tmpstr += ">"
                flag = True

            if (flag):
                tmpstr += '| ' + str(q[i]) + '|\n'
            else:
                tmpstr += ' | ' + str(q[i]) + '|\n'

        return tmpstr

    def __call__(self):
        return self.queue

    def initialize_queue(self, init_queue=[]):
        self.queue = copy(init_queue)

    def sort(self, key=str.lower):
        self.queue = sorted(self.queue, key=key)

    def push(self, data):
        self.queue.append(data)
        self.end += 1

    def pop(self):
        p = self.queue.pop(self.start)
        self.end = len(self.queue) - 1
        return p


class Node():
    def __init__(self, name):
        self.name = name
        self.children = []
        self.weight = []

    def __repr__(self):
        return self.name

    def add_children(self, node, w=None):
        if w == None:
            w = [1] * len(node)
        self.children.extend(node)
        self.weight.extend(w)


class Tree():
    def __init__(self, name):
        self.name = name
        self.root = 0
        self.end = 0
        self.g = {}
        self.g_visual = Graph('G')


    def __call__(self):
        for name, node in self.g.items():
            if (self.root == name):
                self.g_visual.node(name, name, color='red')
            elif (self.end == name):
                self.g_visual.node(name, name, color='blue')
            else:
                self.g_visual.node(name, name)
            for i in range(len(node.children)):
                c = node.children[i]
                w = node.weight[i]
                # print('%s -> %s'%(name,c.name))
                if w == 0:
                    self.g_visual.edge(name, c.name)
                else:
                    self.g_visual.edge(name, c.name, label=str(w))
        return self.g_visual

    def add_node(self, node, start=False, end=False):
        self.g[node.name] = node
        if (start):
            self.root = node.name
        elif (end):
            self.end = node.name

    def set_as_root(self, node):
        # These are exclusive conditions
        self.root = True
        self.end = False

    def set_as_end(self, node):
        # These are exclusive conditions
        self.root = False
        self.end = True

class AStar():
    def __init__(self, in_tree):
        self.in_tree = in_tree
        self.q = Queue()
        self.dist = {name: np.Inf for name, node in in_tree.g.items()}
        self.h = {name: 0 for name, node in in_tree.g.items()}

        for name, node in in_tree.g.items():
            start = tuple(map(int, name.split(',')))
            end = tuple(map(int, self.in_tree.end.split(',')))
            self.h[name] = (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2

        self.via = {name: 0 for name, node in in_tree.g.items()}
        for __, node in in_tree.g.items():
            self.q.push(node)

    def __get_f_score(self, node):
        return self.dist[node.name] + self.h[node.name]

    def solve(self, sn, en):
        self.dist[sn.name] = 0
        while len(self.q) > 0:
            self.q.sort(key=self.__get_f_score)
            u = self.q.pop()
            # print(u.name,self.q.queue)
            if u.name == en.name:
                break
            for i in range(len(u.children)):
                c = u.children[i]
                w = u.weight[i]
                new_dist = self.dist[u.name] + w
                if new_dist < self.dist[c.name]:
                    self.dist[c.name] = new_dist
                    self.via[c.name] = u.name

    def reconstruct_path(self, sn, en):
        start_key = sn.name
        end_key = en.name
        dist = self.dist[end_key]
        u = end_key
        path = [u]
        while u != start_key:
            u = self.via[u]
            path.append(u)
        path.reverse()
        return path, dist


class MapProcessor():
    def __init__(self, name):
        self.map = Map(name)
        self.inf_map_img_array = np.zeros(self.map.image_array.shape)
        self.map_graph = Tree(name)

    def __modify_map_pixel(self, map_array, i, j, value, absolute):
        if ((i >= 0) and
                (i < map_array.shape[0]) and
                (j >= 0) and
                (j < map_array.shape[1])):
            if absolute:
                map_array[i][j] = value
            else:
                map_array[i][j] += value

    def __inflate_obstacle(self, kernel, map_array, i, j, absolute):
        dx = int(kernel.shape[0] // 2)
        dy = int(kernel.shape[1] // 2)
        if (dx == 0) and (dy == 0):
            self.__modify_map_pixel(map_array, i, j, kernel[0][0], absolute)
        else:
            for k in range(i - dx, i + dx):
                for l in range(j - dy, j + dy):
                    self.__modify_map_pixel(map_array, k, l, kernel[k - i + dx][l - j + dy], absolute)

    def inflate_map(self, kernel, absolute=True):
        # Perform an operation like dilation, such that the small wall found during the mapping process
        # are increased in size, thus forcing a safer path.
        self.inf_map_img_array = np.zeros(self.map.image_array.shape)
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.map.image_array[i][j] == 0:
                    self.__inflate_obstacle(kernel, self.inf_map_img_array, i, j, absolute)
        r = np.max(self.inf_map_img_array) - np.min(self.inf_map_img_array)
        if r == 0:
            r = 1
        self.inf_map_img_array = (self.inf_map_img_array - np.min(self.inf_map_img_array)) / r

    def get_graph_from_map(self):
        # Create the nodes that will be part of the graph, considering only valid nodes or the free space
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.inf_map_img_array[i][j] == 0:
                    node = Node('%d,%d' % (i, j))
                    self.map_graph.add_node(node)
        # Connect the nodes through edges
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.inf_map_img_array[i][j] == 0:
                    if (i > 0):
                        if self.inf_map_img_array[i - 1][j] == 0:
                            # add an edge up
                            child_up = self.map_graph.g['%d,%d' % (i - 1, j)]
                            self.map_graph.g['%d,%d' % (i, j)].add_children([child_up], [1])
                    if (i < (self.map.image_array.shape[0] - 1)):
                        if self.inf_map_img_array[i + 1][j] == 0:
                            # add an edge down
                            child_dw = self.map_graph.g['%d,%d' % (i + 1, j)]
                            self.map_graph.g['%d,%d' % (i, j)].add_children([child_dw], [1])
                    if (j > 0):
                        if self.inf_map_img_array[i][j - 1] == 0:
                            # add an edge to the left
                            child_lf = self.map_graph.g['%d,%d' % (i, j - 1)]
                            self.map_graph.g['%d,%d' % (i, j)].add_children([child_lf], [1])
                    if (j < (self.map.image_array.shape[1] - 1)):
                        if self.inf_map_img_array[i][j + 1] == 0:
                            # add an edge to the right
                            child_rg = self.map_graph.g['%d,%d' % (i, j + 1)]
                            self.map_graph.g['%d,%d' % (i, j)].add_children([child_rg], [1])
                    if ((i > 0) and (j > 0)):
                        if self.inf_map_img_array[i - 1][j - 1] == 0:
                            # add an edge up-left
                            child_up_lf = self.map_graph.g['%d,%d' % (i - 1, j - 1)]
                            self.map_graph.g['%d,%d' % (i, j)].add_children([child_up_lf], [np.sqrt(2)])
                    if ((i > 0) and (j < (self.map.image_array.shape[1] - 1))):
                        if self.inf_map_img_array[i - 1][j + 1] == 0:
                            # add an edge up-right
                            child_up_rg = self.map_graph.g['%d,%d' % (i - 1, j + 1)]
                            self.map_graph.g['%d,%d' % (i, j)].add_children([child_up_rg], [np.sqrt(2)])
                    if ((i < (self.map.image_array.shape[0] - 1)) and (j > 0)):
                        if self.inf_map_img_array[i + 1][j - 1] == 0:
                            # add an edge down-left
                            child_dw_lf = self.map_graph.g['%d,%d' % (i + 1, j - 1)]
                            self.map_graph.g['%d,%d' % (i, j)].add_children([child_dw_lf], [np.sqrt(2)])
                    if ((i < (self.map.image_array.shape[0] - 1)) and (j < (self.map.image_array.shape[1] - 1))):
                        if self.inf_map_img_array[i + 1][j + 1] == 0:
                            # add an edge down-right
                            child_dw_rg = self.map_graph.g['%d,%d' % (i + 1, j + 1)]
                            self.map_graph.g['%d,%d' % (i, j)].add_children([child_dw_rg], [np.sqrt(2)])

    def gaussian_kernel(self, size, sigma=1):
        size = int(size) // 2
        x, y = np.mgrid[-size:size + 1, -size:size + 1]
        normal = 1 / (2.0 * np.pi * sigma ** 2)
        g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
        r = np.max(g) - np.min(g)
        sm = (g - np.min(g)) * 1 / r
        return sm

    def rect_kernel(self, size, value):
        m = np.ones(shape=(size, size))
        return m

    def draw_path(self, path):
        path_tuple_list = []
        path_array = copy(self.inf_map_img_array)
        for idx in path:
            tup = tuple(map(int, idx.split(',')))
            path_tuple_list.append(tup)
            path_array[tup] = 0.5
        return path_array


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
        self.pkgpath = None

        # Path planner/follower related variables
        self.path = Path()
        self.goal_pose = PoseStamped()
        self.ttbot_pose = PoseStamped()
        self.wp_dist = None

        # Covarance check
        self.cov_check = False

        # From Jupyter Notebook
        rospack = rospkg.RosPack()
        self.pkgpath = rospack.get_path("final_project")
        self.mp = MapProcessor(self.pkgpath + '/maps/map')
        self.mp.inflate_map(self.mp.rect_kernel(12, 1))
        self.mp.get_graph_from_map()

        # OpenCV Initializations
        self.bridge = CvBridge()
        cv2.namedWindow("Image Window", 1)
        self.cv_image = None

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
        rospy.Subscriber("/camera/rgb/image_raw", cvImage, self.image_callback)

        # Publishers
        self.path_pub = rospy.Publisher('global_plan', Path, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)

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

    def image_callback(self, image_msg):
        # rospy.loginfo(image_msg.header)

        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(image_msg, "passthrough")
        except CvBridgeError:
            rospy.loginfo("CvBridge Error")

        print(self.cv_image.shape)
        self.show_image(self.cv_image)

    def show_image(self, image):
        cv2.imshow("Image Window", image)
        cv2.waitKey(3)

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
        x_offset = 180
        y_offset = 200

        x_conv = -0.05
        y_conv = 0.05

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
            point_pose.pose.position.x = (-1 * pose_x) - 1
            point_pose.pose.position.y = (-1 * pose_y) - 1

            path.poses.append(point_pose)

        print('path generated successfully. YAY!')
        print(path)
        print(self.mp.map_graph.root, self.mp.map_graph.end)
        print("a* complete\n\n")

        path.header.frame_id = 'map'

        # publish path for Rviz to display on screen
        self.path_pub.publish(path)

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
        if idx + 5 < len(path.poses):
            idx = idx + 5
        elif idx + 3 < len(path.poses):
            idx = idx + 3
        elif idx + 1 < len(path.poses):
            idx = idx + 1

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

        self.wp_dist = np.sqrt((wp_x - bot_x) ** 2 + (wp_y - bot_y) ** 2)

        dx = wp_x - bot_x
        dy = wp_y - bot_y

        heading = np.arctan2(dy, dx)
        speed = min(0.2, (2 * self.wp_dist))

        return speed, heading

    def move_ttbot(self, speed, heading):
        """! Function to move turtlebot passing directly a heading angle and the speed.
        @param  speed     Desired yaw angle.
        @param  heading   Desired speed.
        @return path      object containing the sequence of waypoints of the created path.
        """
        cmd_vel = Twist()

        # calculate current turtle bot orientation (yaw)
        bot_pose = self.ttbot_pose

        bot_quat = (bot_pose.pose.orientation.x, bot_pose.pose.orientation.y, bot_pose.pose.orientation.z,
                    bot_pose.pose.orientation.w)
        bot_euler = tf.transformations.euler_from_quaternion(bot_quat)
        bot_yaw = bot_euler[2]

        # calculate steering angle
        steering_angle = heading - bot_yaw

        if abs(steering_angle) > 0.3:
            speed = -0.05

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

    def calc_goal_yaw(self):
        goal_quat = (self.goal_pose.pose.orientation.x,
                     self.goal_pose.pose.orientation.y,
                     self.goal_pose.pose.orientation.z,
                     self.goal_pose.pose.orientation.w)
        goal_euler = tf.transformations.euler_from_quaternion(goal_quat)
        goal_yaw = goal_euler[2]

        return goal_yaw

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
        idx_prev = 0

        while not rospy.is_shutdown():
            # 0. Localize robot initial position
            '''
            if self.cov_check == False:
                # spin turtlebot to find initial pose
                print('cov_check failed')
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
                idx_prev = 0

            # Reset goal_flag to current goal pose
            goal_flag = self.goal_pose

            # if no path is returned, stop moving
            if path is None:
                self.move_ttbot(0, 0)
                continue

            # Find next point to go towards and set current goal
            idx = self.get_path_idx(path, self.ttbot_pose)

            if idx > idx_prev:
                idx_prev = idx
            else:
                idx = idx_prev

            current_goal = path.poses[idx]

            # Checks for final waypoint

            if((idx + 1) == len(path.poses)) and (self.wp_dist < 0.05):
                print("reached goal pose")
                speed = 0
                heading = self.calc_goal_yaw()
            elif (idx + 5) > len(path.poses):
                print("approaching goal pose")
                speed, heading = self.path_follower(self.ttbot_pose, current_goal)
                speed = 0.5 * speed
            else:
                print("target waypoint:", idx)
                # Route to waypoint (speed and heading)
                speed, heading = self.path_follower(self.ttbot_pose, current_goal)

            # move ttbot to waypoint
            self.move_ttbot(speed, heading)
            '''

            self.rate.sleep()
        rospy.signal_shutdown("[{}] Finished Cleanly".format(self.name))

if __name__ == "__main__":
    nav = Navigation(node_name='Navigation')
    nav.init_app()
    try:
        nav.run()
    except rospy.ROSInterruptException:
        print("program interrupted before completion", file=sys.stderr)
