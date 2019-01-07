#!/usr/bin/env python

#source devel/setup.bash
#./baxter.sh alan.local
#rosrun intera_interface enable_robot.py -e
#rosrun intera_interface joint_trajectory_action_server.py

#roslaunch sawyer_moveit_config sawyer_moveit.launch electric_gripper:=true

#rosrun drawing pizza.py

#rosrun tf tf_echo base right_gripper_tip

import argparse
import numpy as np
from numpy import ones,vstack
from numpy.linalg import lstsq
import cv2, time, sys
from cv_bridge import CvBridge, CvBridgeError
import rospy
import intera_interface
from numpy.linalg import *
import math
import sys
import draw

use = True

def inside_of(rect, obj):
    #Takes approximate contour of a rect and an object and decides if the object is in the rect 
    (tl, bl, br, tr) = rect
    left = min(tl[0][0], bl[0][0])
    right = max(br[0][0], tr[0][0])
    top = min(tl[0][1], tr[0][1])
    bot = max(bl[0][1], br[0][1])
    for i in obj:
        if i[0][0] < left or i[0][0] > right or i[0][1] < top or i[0][1] > bot:
            return False
    return True

def dist(a, b): 
    return np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

def find_midpoints(a, b, distance): 
    set_of_pts = [a]
    if dist(a, b) <= distance: 
        set_of_pts.append(b)
        return set_of_pts
    else: 
        c = ((a[0] + b[0])/2, (a[1] + b[1])/2)

        a_pts = find_midpoints(a, c, distance)
        b_pts = find_midpoints(c, b, distance)
        for i in range(1, len(a_pts)): 
            set_of_pts.append(a_pts[i])
        for i in range(1, len(b_pts)):
            set_of_pts.append(b_pts[i])
    return set_of_pts

def show_image_callback(img_data, (edge_detection, window_name)):
    """The callback function to show image by using CvBridge and cv"""
    global use
    if use:
        bagel.observe(200000) #200000
    bridge = CvBridge()
    board = None
    obj = None
    found = 0

    # try:
    #     cv_image = bridge.imgmsg_to_cv2(img_data, "bgr8")
    # except CvBridgeError, err:
    #     rospy.logerr(err)
    #     return
    # gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # get_edge = cv2.Canny(blurred, 30, 200)
    # image = np.hstack([get_edge])
    # _, contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cnt = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
    # for c in cnt:
    #     peri = 0.003*cv2.arcLength(c, True)
    #     approx = np.array(cv2.approxPolyDP(c, peri, True))
    #     if cv2.contourArea(c) > 160000 and cv2.contourArea(c) < 180000:
    #         board = approx #point order: TL, BL, BR, TR
    # print(cv2.contourArea(board))

    try:
        cv_image = bridge.imgmsg_to_cv2(img_data, "bgr8")
    except CvBridgeError, err:
        rospy.logerr(err)
        return

    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    get_edge = cv2.Canny(blurred, 30, 200)
    image = np.hstack([get_edge])
    _, contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
    for c in cnt:
        # approximate the contour
        peri = 0.003*cv2.arcLength(c, True)
        approx = np.array(cv2.approxPolyDP(c, peri, True))
        if found == 0 and cv2.contourArea(c) > 150000 and cv2.contourArea(c) < 180000 and len(approx) == 4:
            print('Found board')
            board = approx #point order: TL, BL, BR, TR
            found = 69
        elif found == 69 and inside_of(board, approx) and cv2.contourArea(approx) < cv2.contourArea(board) - 20000:
            print('Found shape')
            obj = approx
            found = 42
            break
    if found == 42:
        found = 0
        use = True
        start = obj[0]
        end = obj[len(obj)/2]
        rows,cols = cv_image.shape[:2]
        [vx,vy,x,y] = cv2.fitLine(obj, cv2.DIST_L2,0,0.01,0.01)
        lefty = int((-x*vy/vx) + y)
        righty = int(((cols-x)*vy/vx)+y)
        points = [(cols-1,righty), (0,lefty)]
        x_coords, y_coords = zip(*points)
        A = vstack([x_coords,ones(len(x_coords))]).T
        m, c = lstsq(A, y_coords)[0]
        more_pts = []
        for i in range(len(obj)):
            j = i + 1
            if i == len(obj) - 1:
                j = 0
            a = obj[i][0]
            b = obj[j][0]

            z = find_midpoints(a, b, 15)

            for k in z: 
                more_pts.append(k)
        try:
            cv_image = bridge.imgmsg_to_cv2(img_data, "bgr8")
        except CvBridgeError, err:
            rospy.logprint("****")
            return
        for i in more_pts:
            cv2.circle(cv_image, (i[0], i[1]), 5, (0,0,255), -1)
        cv2.drawContours(cv_image, [board], -1, (0, 0, 255), 5)
        #cv2.drawContours(cv_image, [obj], -1, (255, 0, 0), 4)
        cv2.line(cv_image, points[0], points[1],(0,255,0),2)
        cv2.imshow('Bordered', cv_image)
        cv2.waitKey(1)
        obj_top = []
        obj_bot = []
        if m <= 10 or m >= -10:
            for i in more_pts:
                y_line = m*i[0] + c
                if i[1] > y_line:
                    obj_bot.append(i)
                else: obj_top.append(i)
        else:
            for i in more_pts: 
                x_line = (i[1] - c) / m
                if i[0] > x_line: 
                    obj_top.append(i)
                else: 
                    obj_bot.append(i)
        final_top = np.array([obj_top[0][0], obj_top[0][1], 1])
        final_bot = np.array([obj_bot[0][0], obj_bot[0][1], 1])
        for i in range(1, len(obj_top)):
            final_top = np.vstack((final_top, np.array([obj_top[i][0], obj_top[i][1], 1])))
        for i in range(1, len(obj_bot)):
            final_bot = np.vstack((final_bot, np.array([obj_bot[i][0], obj_bot[i][1], 1])))
        #raw_input('Shape found. Press <Enter>')
        bagel.ready(20000)
        bagel.draw(final_top, final_bot, board)
    else: 
        use = False
        found = 0
        cv2.waitKey(1)
def main():
    rp = intera_interface.RobotParams()
    valid_cameras = rp.get_camera_names()
    if not valid_cameras:
        rp.log_message(("Cannot detect any camera_config"
            " parameters on this robot. Exiting."), "ERROR")
        return
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt,
                                     description=main.__doc__)
    parser.add_argument(
        '-c', '--camera', type=str, default="right_hand_camera",
        choices=valid_cameras, help='Setup Camera Name for Camera Display')
    parser.add_argument(
        '-r', '--raw', action='store_true', 
        help='Specify use of the raw image (unrectified) topic')
    parser.add_argument(
        '-e', '--edge', action='store_true',
        help='Streaming the Canny edge detection image')
    args = parser.parse_args()
    rospy.init_node('camera_display', anonymous=True)
    camera = intera_interface.Cameras()
    if not camera.verify_camera_exists(args.camera):
        rospy.logerr("Invalid camera name, exiting the example.")
        return
    camera.start_streaming(args.camera)
    rectify_image = not args.raw
    use_canny_edge = args.edge
    camera.set_callback(args.camera, show_image_callback,
        rectify_image=rectify_image, callback_args=(use_canny_edge, args.camera))
    def clean_shutdown():
        print("Shutting down camera_display node.")
        cv2.destroyAllWindows()
    rospy.on_shutdown(clean_shutdown)
    # rospy.loginfo("Camera_display node running. Ctrl-c to quit")
    rospy.spin()

if __name__ == '__main__':
    main()
    #/io/internal_camera/right_hand_camera/image_raw