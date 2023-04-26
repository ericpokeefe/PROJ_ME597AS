#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import argparse
from operator import xor
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import rospy
import numpy as np

# Instantiate CvBridge
bridge_object = CvBridge()
cv2_img = None

def callback(value):
    pass

def image_callback(msg):
    global cv2_img
    print("Received an image!")
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge_object.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        print(e)

def setup_trackbars(range_filter):
    cv2.namedWindow("Trackbars", 0)
    for i in ["MIN", "MAX"]:
        v = 0 if i == "MIN" else 255
        for j in range_filter:
            cv2.createTrackbar("%s_%s" % (j, i), "Trackbars", v, 255, callback)

def get_trackbar_values(range_filter):
    values = []

    for i in ["MIN", "MAX"]:
        for j in range_filter:
            v = cv2.getTrackbarPos("%s_%s" % (j, i), "Trackbars")
            values.append(v)

    return values


#construct the argument parser to parse input static image
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False, help="path to the input image")
args = vars(ap.parse_args())
print(args)
image_path = args['image']


def main():
    global cv2_img
    if not image_path:
        rospy.init_node("hsv_detector", log_level=rospy.DEBUG)
        rospy.logwarn("Starting....")

        image_topic = "/camera/rgb/image_raw"
        rospy.Subscriber(image_topic, Image, image_callback)
        # sensors_obj = RobotRGBSensors("/camera/rgb/image_raw")
        # cv_image = cv2_img

    range_filter = "HSV"

    setup_trackbars(range_filter)

    # Load and convert static image once
    if image_path:
        image = cv2.imread(image_path)
        frame_to_thresh = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    while not rospy.is_shutdown():
        if not image_path:
            #image = sensors_obj.get_image()
            image = cv2_img
            if image is None:
                continue
            else:
                frame_to_thresh = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        v1_min, v2_min, v3_min, v1_max, v2_max, v3_max = get_trackbar_values(range_filter)

        thresh = cv2.inRange(frame_to_thresh, (v1_min, v2_min, v3_min), (v1_max, v2_max, v3_max))

        preview = cv2.bitwise_and(image, image, mask=thresh)
        # preview = cv.cvSet(preview, (0, 255, 0))
        # preview[np.where(preview == [1])] = (0, 255, 0)
        # preview[np.where(preview == [0])] = [255]   # for detect black

        cv2.imshow("Preview", preview)

        if cv2.waitKey(1) & 0xFF is ord('q'):
            break

    rospy.logwarn("Shutting down")

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
