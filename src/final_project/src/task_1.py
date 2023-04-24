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
        # TODO: might not need this
        # rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.__goal_pose_cbk, queue_size=1)

        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.__ttbot_pose_cbk, queue_size=1)
        #rospy.Subscriber('/scan', LaserScan, self.__laser_scan_cbk, queue_size=1)

        # Publishers
        self.path_pub = rospy.Publisher('global_plan', Path, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.show_path_pub = rospy.Publisher('astar_plan', Path, queue_size=1)

        rospy.loginfo('Node Initialized!')

    # TODO: Might not need this
    """
    def __goal_pose_cbk(self, data):
        # Callback to catch the goal pose.
        #@param  data    PoseStamped object from RVIZ.
        #@return None.
    
        self.goal_pose = data
        rospy.loginfo('goal_pose:{:.4f},{:.4f}'.format(self.goal_pose.pose.position.x, self.goal_pose.pose.position.y))
    """

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
        self.laser_min = data.range_min
        self.laser_max = data.range_min
        self.laser_ranges = data.ranges

    def move_ttbot(self, speed, heading):
        """ Function to move turtlebot passing directly a heading angle and the speed.
        @param  speed     Desired yaw angle.
        @param  heading   Desired speed.
        @return path      object containing the sequence of waypoints of the created path.
        """
        cmd_vel = Twist()

        # calculate current turtle bot orientation(yaw)
        bot_pose = self.ttbot_pose

        bot_quat = (bot_pose.pose.orientation.x, bot_pose.pose.orientation.y, bot_pose.pose.orientation.z,
                    bot_pose.pose.orientation.w)
        bot_euler = tf.transformations.euler_from_quaternion(bot_quat)
        bot_yaw = bot_euler[2]

        # calculate steering angle
        # steering_angle = 2 * (heading - bot_yaw)

        cmd_vel.linear.x = speed
        # cmd_vel.angular.z = steering_angle

        # TODO: using steering angle instead of heading but still named heading
        cmd_vel.angular.z = heading

        self.cmd_vel_pub.publish(cmd_vel)

    def follow_wall(self):
        speed = 0
        heading = 0

        return speed, heading

    def run(self):
        """ Main loop of node to run
        """
        while not rospy.is_shutdown():
            rospy.loginfo('Explorer node alive ...')

            #rospy.loginfo('laser min, max:{:.4f}'.format(self.laser_min, self.laser_max))

            #speed, heading = self.follow_wall()

            #self.move_ttbot(speed, heading)

            self.rate.sleep()
        # rospy.signal_shutdown("[{}] Finished Cleanly".format(self.name))


if __name__ == '__main__':
    ex = Explorer(node_name='explorer')
    ex.init_app()

    try:
        ex.run()
    except rospy.RROSInterruptException:
        print("program interrupted before completion", file=sys.stderr)