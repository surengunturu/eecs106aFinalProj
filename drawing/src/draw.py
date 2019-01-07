#!/usr/bin/env python

import sys
import rospy
import numpy as np

from moveit_msgs.msg import OrientationConstraint
from geometry_msgs.msg import PoseStamped

from path_planner import PathPlanner
# from baxter_interface import Limb
from controller import Controller
from intera_interface import Limb
from intera_interface import gripper as robot_gripper
from kin_func_skeleton import *

def observe(time):
    joint_command = {}
    right = Limb('right')
    joint_command['right_j0'] = 0.75
    joint_command['right_j1'] = 0.22
    joint_command['right_j2'] = -1.72
    joint_command['right_j3'] = 1.17
    joint_command['right_j4'] = -1.37
    joint_command['right_j5'] = -2.97
    joint_command['right_j6'] = -1.54
    for _ in range(time):
        try: 
            right.set_joint_positions(joint_command)
        except KeyboardInterrupt : print('done')
    raw_input("Ready to find Shape. Press <Enter>")

def ready(time):
    joint_command = {}
    right = Limb('right')
    joint_command['right_j0'] = 0.5
    joint_command['right_j1'] = 0
    joint_command['right_j2'] = -1.4
    joint_command['right_j3'] = 1
    joint_command['right_j4'] = -1.7
    joint_command['right_j5'] = -1
    joint_command['right_j6'] = -1.7
    for _ in range(time):
        try: 
            right.set_joint_positions(joint_command)
        except KeyboardInterrupt : print('done')
    # right_gripper = robot_gripper.Gripper('right_gripper')
    # right_gripper.calibrate()
    # rospy.sleep(2.0)
    # right_gripper.close()
    # rospy.sleep(1.0)
    raw_input("Ready to draw. Press <Enter>")

def quadtomat(points):
    x = .828
    y = 0.550
    z = 0.028
    w = 0.107
    n = np.sqrt(x ** 2 + y ** 2 + z ** 2 + w ** 2)
    x = x/n
    y = y/n
    z = z/n
    w = w/n
    a = np.array([
        [1 - 2*(y**2) - 2*(z**2), (2*x*y) + (2*w*z), (2*x*z) - (2*w*y)], 
        [(2*x*y) - (2*w*z), 1 - 2*(x**2) - 2*(z**2), (2*y*z) + (2*w*x)], 
        [(2*x*z) + (2*w*y), (2*y*z) - (2*w*x), 1 - 2*(x**2) - 2*(y**2)]
        ])
    return np.dot(np.linalg.inv(a), points)

def homogenous(points, uv_board):
    tl = None
    tr = None
    bl = None
    br = None
    for i in uv_board:
        if i[0][0] < 300:
            if i[0][1] < 200:
                tl = i[0]
            else: bl = i[0]
        elif i[0][1] < 200:
            tr = i[0]
        else: br = i[0]

    #rosrun tf tf_echo base right_gripper_tip
    tl1 = np.array([0.340, -0.013])
    tr1 = np.array([0.554, 0.452])
    bl1 = np.array([0.694, -0.159])
    br1 = np.array([0.909, 0.311])


    a = np.array([
        [tl1[0], tl1[1], 1, 0, 0, 0, -tl[0] * tl1[0], -tl[0] * tl1[1]], 
        [0, 0, 0, tl1[0], tl1[1], 1, -tl[1] * tl1[0], -tl[1] * tl1[1]], 
        [bl1[0], bl1[1], 1, 0, 0, 0, -bl[0] * bl1[0], -bl[0] * bl1[1]], 
        [0, 0, 0, bl1[0], bl1[1], 1, -bl[1] * bl1[0], -bl[1] * bl1[1]], 
        [br1[0], br1[1], 1, 0, 0, 0, -br[0] * br1[0], -br[0] * br1[1]], 
        [0, 0, 0, br1[0], br1[1], 1, -br[1] * br1[0], -br[1] * br1[1]], 
        [tr1[0], tr1[1], 1, 0, 0, 0, -tr[0] * tr1[0], -tr[0] * tr1[1]], 
        [0, 0, 0, tr1[0], tr1[1], 1, -tr[1] * tr1[0], -tr[1] * tr1[1]]
        ])
    b = np.array([[tl[0]], [tl[1]], [bl[0]], [bl[1]], [br[0]], [br[1]], [tr[0]], [tr[1]]])
    x = np.dot(np.linalg.inv(a), b)
    H = np.array([
        [x[0][0], x[1][0], x[2][0]], 
        [x[3][0], x[4][0], x[5][0]], 
        [x[6][0], x[7][0], 1]
    ])
    Q = np.linalg.inv(H)
    return Q

def draw(top, bot, board):
    Q = homogenous(top.T, board)
    top = np.dot(Q, top.T).T
    bot = np.dot(Q, bot.T).T

    planner = PathPlanner('right_arm')

    Kp = 0.1 * np.array([0.3, 2, 1, 1.5, 2, 2, 3]) # Stolen from 106B Students
    Kd = 0.01 * np.array([2, 1, 2, 0.5, 0.5, 0.5, 0.5]) # Stolen from 106B Students
    Ki = 0.01 * np.array([1, 1, 1, 1, 1, 1, 1]) # Untuned
    Kw = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]) # Untuned

    controller = Controller(Kp, Kd, Ki, Kw, Limb('right'))

    orien_const = OrientationConstraint()
    orien_const.link_name = "right_gripper";
    orien_const.header.frame_id = "base";
    orien_const.orientation.y = -1.0;
    orien_const.absolute_x_axis_tolerance = 0.1;
    orien_const.absolute_y_axis_tolerance = 0.1;
    orien_const.absolute_z_axis_tolerance = 0.1;
    orien_const.weight = 1.0;

    ori = PoseStamped()
    ori.header.frame_id = 'base'
    ori.pose.orientation.x = 0.0
    ori.pose.orientation.y = -1.0
    ori.pose.orientation.z = 0.0
    ori.pose.orientation.w = 0.0
    goals = np.empty(len(top) + len(bot), dtype=object)
    while not rospy.is_shutdown():
        try:
            counter = 0
            count_top = 0
            count_bot = len(bot) - 1
            while count_top < len(top) or count_bot >= 0: 
                if count_top < len(top):
                    goal = PoseStamped()
                    goal.header.frame_id = "base"
                    goal.pose.position.x = top[count_top][0] / top[count_top][2] - 0.014 + .012
                    goal.pose.position.y = top[count_top][1] / top[count_top][2] + 0.005 - 0.04
                    goal.pose.position.z = -0.1849
                    goals[counter] = goal
                    counter += 1
                    count_top += 1
                if count_bot >= 0:
                    goal = PoseStamped()
                    goal.header.frame_id = "base"
                    goal.pose.position.x = bot[count_bot][0] / bot[count_bot][2] + 0.003
                    goal.pose.position.y = bot[count_bot][1] / bot[count_bot][2] - 0.01
                    goal.pose.position.z = -0.1861
                    goals[counter] = goal
                    counter += 1
                    count_bot -= 1
            for i in goals:
                i.pose.orientation = ori.pose.orientation
                plan = planner.plan_to_pose(i, list())
                if not controller.execute_path(plan):
                    raise Exception("Execution failed")
        except Exception as e:
            print e
        else:
            break

def move():
    planner = PathPlanner('right_arm')

    Kp = 0.1 * np.array([0.3, 2, 1, 1.5, 2, 2, 3]) # Stolen from 106B Students
    Kd = 0.01 * np.array([2, 1, 2, 0.5, 0.5, 0.5, 0.5]) # Stolen from 106B Students
    Ki = 0.01 * np.array([1, 1, 1, 1, 1, 1, 1]) # Untuned
    Kw = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]) # Untuned

    controller = Controller(Kp, Kd, Ki, Kw, Limb('right'))

    ori = PoseStamped()
    ori.header.frame_id = 'base'
    ori.pose.orientation.x = 0.0
    ori.pose.orientation.y = -1.0
    ori.pose.orientation.z = 0.0
    ori.pose.orientation.w = 0.0

    diff = 0

    observe(150000)
    ready(20000)

    #draw(planner, controller, ori)

    while not rospy.is_shutdown():
        for _ in range(10):
            while not rospy.is_shutdown():
                try:
                    #rosrun tf tf_echo base right_gripper_tip
                    goal_1 = PoseStamped()
                    goal_1.header.frame_id = "base"

                    #x, y, and z position
                    goal_1.pose.position.x = 0.65
                    goal_1.pose.position.y = 0.02 + diff
                    goal_1.pose.position.z = -0.227

                    #Orientation as a quaternion
                    goal_1.pose.orientation = ori.pose.orientation

                    plan1 = planner.plan_to_pose(goal_1, list())
                    if not controller.execute_path(plan1):
                        raise Exception("Execution failed")

                    goal_2 = PoseStamped()
                    goal_2.header.frame_id = "base"

                    #x, y, and z position
                    goal_2.pose.position.x = 0.80
                    goal_2.pose.position.y = -0.01 + diff
                    goal_2.pose.position.z = -0.227 #should be a little bit lower than initial point

                    #Orientation as a quaternion
                    goal_2.pose.orientation = ori.pose.orientation

                    plan2 = planner.plan_to_pose(goal_2, list())
                    diff += .004
                    
                    #raw_input("Press <Enter>")
                    if not controller.execute_path(plan2):
                        raise Exception("Execution failed")
                except Exception as e:
                    print e
                else:
                    break
        observe(50000)
        ready(20000)

if __name__ == '__main__':
    rospy.init_node('moveit_node')
    move()
