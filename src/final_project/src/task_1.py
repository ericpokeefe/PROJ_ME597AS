#!/usr/bin/env python3

import rospy
import sys
import tf

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Pose, Twist
from sensor_msgs.msg import LaserScan

class Explorer:
    """ Exploring node class
    This class follows a wall and maps an unknown environment.
    """

    def __init__(self, node_name='Explorer'):
        # ROS related variables
        self.node_name = node_name
        self.rate = 0

    def init_app(self):
        """Node initialization
        """
        # ROS node initialization
        rospy.init_node(self.node_name, anonymous=True)
        self.rate = rospy.Rate(30)

        # Subscribers
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.__ttbot_pose_cbk, queue_size=1)
        rospy.Subscriber('/scan', LaserScan, self.__laser_scan_cbk, queue_size=1)

        # Publishers
        self.path_pub = rospy.Publisher('global_plan', Path, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.show_path_pub = rospy.Publisher('astar_plan', Path, queue_size=1)

        rospy.loginfo('Node Initialized!')

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

    def __laser_scan_cbk(self, data):
        # self.laser_min = data.range_min
        # self.laser_max = data.range_max
        self.laser_ranges = data.ranges
        '''
        rospy.loginfo('0: {:.4f}, 89: {:.4f}, 179: {:.4f}, 269; {:.4f}'.format(self.laser_ranges[0],
                                                                               self.laser_ranges[89],
                                                                               self.laser_ranges[179],
                                                                               self.laser_ranges[269]))
        '''

        # find front, left, back, right distances
        self.fov = 2  # (total fov / 2) - 1
        self.front_dist = min(min(self.laser_ranges[0:(0 + self.fov)], self.laser_ranges[(359 - (self.fov - 1)):359]))
        self.left_dist = min(self.laser_ranges[(89 - self.fov):(89 + self.fov)])
        self.back_dist = min(self.laser_ranges[(179 - self.fov):(179 + self.fov)])
        self.right_dist = min(self.laser_ranges[(269 - self.fov):(269 + self.fov)])

        rospy.loginfo('front: {:.2f}, left: {:.2f}, back: {:.2f}, right: {:.2f}'.format(self.front_dist,
                                                                                        self.left_dist,
                                                                                        self.back_dist,
                                                                                        self.right_dist))

        # find minimum distance and corresponding index
        self.laser_range_min = 100
        self.laser_range_min_index = None

        for (i, item) in enumerate(self.laser_ranges):
            if item < self.laser_range_min:
                self.laser_range_min = item
                self.laser_range_min_index = i

        rospy.loginfo('range_min = {:.4f}, min_index: {:d}'.format(self.laser_range_min, self.laser_range_min_index))

    def move_ttbot(self, speed, steer_angle):
        """! Function to move turtlebot passing directly a speed and steering angle.
        @param  speed     Desired speed.
        @param  steer_angle   Desired steering angle.
        """
        cmd_vel = Twist()

        cmd_vel.linear.x = speed
        cmd_vel.angular.z = steer_angle

        self.cmd_vel_pub.publish(cmd_vel)

    def find_wall(self):
        # state 0 in state machine
        # rotates right until self.front_dist is minimized

        return

    def approach_wall(self):
        # state 1 in state machine
        # drives straight until self.front_dist hits desired distance

        return

    def follow_wall(self):
        # state 2 in state machine
        # turns left until self.right_dist is minimized
        # drives straight with left/right corrections to follow wall on right side of vehicle

        return

    def run(self):
        """ Main loop of node to run
        """
        while not rospy.is_shutdown():
            #rospy.loginfo('Explorer node alive ...')

            # TODO: implement state machine

            self.rate.sleep()
        # rospy.signal_shutdown("[{}] Finished Cleanly".format(self.name))


if __name__ == '__main__':
    ex = Explorer(node_name='explorer')
    ex.init_app()

    try:
        ex.run()
    except rospy.RROSInterruptException:
        print("program interrupted before completion", file=sys.stderr)