# -*- coding: utf-8 -*-
'''
@file utils
'''

from madgwick_filter.tools_ahrs import plot
from madgwick_filter.tools_ahrs import plot3
import ahrs
import mrob
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
# Seed random generator
GENERATOR = np.random.default_rng(42)

def calculate_quat_distances(quaternions1, quaternions2):
    '''
    Calculating distance between rotation quaternions:
    Provide the distance on the rotation in the tangent space of the ln(R * R_rhs^{-1})
    It is APE - Absolute Pose Error

    param: quaternions - array of quaternions [x, y, z, w]

    return: distance (float) between 2 SO3s from quaternions 1, 2
    '''
    R1 = np.array([mrob.quat_to_so3(quat) for quat in quaternions1])
    R2 = np.array([mrob.quat_to_so3(quat) for quat in quaternions2])
    zero_translation = np.array([0, 0, 0])
    SE3_1 = np.array([mrob.SE3(mrob.SO3(R), zero_translation) for R in R1])
    SE3_2 = np.array([mrob.SE3(mrob.SO3(R), zero_translation) for R in R2])

    distance = np.array([(SE3_1[i]).distance_rotation(SE3_2[i]) for i in range(len(SE3_1))])

    return distance

def normalize_quats(quats, w_id=0):
    signs = np.sign(quats[:, w_id])+(np.sign(quats[:, w_id])==0)                      # To multipy negative ones by -1, and zeros by 1 (other terms not 0)
    quats = quats * signs.reshape(-1, 1)
    return quats

def quaternion_multiply(quaternion1, quaternion0):
    '''
    Just quatenions multiplication formula
    '''
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

def calculate_g_distances(quaternions1, quaternions2):
    '''
    Calculating distance (L2 norm) between the gravity vectors orientations in 2 reference frames - 
    1) smartphone attitude according to smartphone data
    2) smartphone attitude according to mocap data
    The idea is, if Madgwick filter could accumulate rotation around vector g, resulting in growing error betweer
    quaternions, the directions of vectors g themselves should always align with same small error

    param: quaternions - two arrays of quaternions with same length, [w, x, y, z]

    return: distance (norm of (g1 - g2)/g) between two gravity vectors orientations
    '''
    R1 = np.array([mrob.quat_to_so3(quat) for quat in np.roll(quaternions1, -1, axis=1)])
    R2 = np.array([mrob.quat_to_so3(quat) for quat in np.roll(quaternions2, -1, axis=1)])

    g = np.array([0, 0, -9.81]).reshape(-1, 1) / 9.81                       # looking just for direction of g

    g_in_1 = np.array([np.linalg.inv(R) @ g for R in R1])
    g_in_2 = np.array([np.linalg.inv(R) @ g for R in R2])

    print("G vectors in smartphone and (mocap-observed-smartphone) reference frames")
    plot(g_in_1.reshape(-1, 3), g_in_2.reshape(-1, 3))
    print("Difference between G vectors in smartphone and (mocap-observed-smartphone) reference frames")
    plot(g_in_1.reshape(-1, 3) - g_in_2.reshape(-1, 3))

    g_in_1_SE3 = np.array([mrob.SE3(mrob.SO3(), g_in_1_i) for g_in_1_i in g_in_1])
    g_in_2_SE3 = np.array([mrob.SE3(mrob.SO3(), g_in_1_2) for g_in_1_2 in g_in_2])

    distance = np.array([mrob.SE3.distance_trans(g_in_1_SE3[i], g_in_2_SE3[i]) for i in range(len(g_in_1_SE3))])

    return distance

def RPE(quaternions1, quaternions2, increment=1):
    '''
    Relative Pose Error
    It shows how different are increments in quaternions1 from increments in quaternions2

    param: increment - number of steps between points in trajectory, which difference we are calculating
    '''
    R1 = np.array([mrob.quat_to_so3(quat) for quat in quaternions1])
    R2 = np.array([mrob.quat_to_so3(quat) for quat in quaternions2])

    zero_translation = np.array([0, 0, 0])

    SE3_1 = np.array([mrob.SE3(mrob.SO3(R), zero_translation) for R in R1])
    SE3_2 = np.array([mrob.SE3(mrob.SO3(R), zero_translation) for R in R2])

    change_in_1 = np.array([(SE3_1[i]).mul(SE3_1[i+increment].inv()) for i in range(len(SE3_1)-increment)])
    change_in_2 = np.array([(SE3_2[i]).mul(SE3_2[i+increment].inv()) for i in range(len(SE3_2)-increment)])

    distance = np.array([(change_in_1[i]).distance(change_in_2[i]) for i in range(len(change_in_1))])

    return distance

def RPE_g(quaternions1, quaternions2, increment=1):
    '''
    Relative Pose Error
    It shows how different are increments of vector g rotated by quaternions1 from increments 
    of vector g rotated by quaternions2

    param: increment - number of steps between points in trajectory, which difference we are calculating
    '''
    R1 = np.array([mrob.quat_to_so3(quat) for quat in quaternions1])
    R2 = np.array([mrob.quat_to_so3(quat) for quat in quaternions2])

    g = np.array([0, 0, -1]).reshape(-1, 1)                      # looking just for direction of g
    
    g_in_1 = np.array([np.linalg.inv(R) @ g for R in R1])
    g_in_2 = np.array([np.linalg.inv(R) @ g for R in R2])

    g_in_1_SE3 = np.array([mrob.SE3(mrob.SO3(), g_in_1_i) for g_in_1_i in g_in_1])
    g_in_2_SE3 = np.array([mrob.SE3(mrob.SO3(), g_in_1_2) for g_in_1_2 in g_in_2])

    change_in_1 = np.array([(g_in_1_SE3[i]).mul(g_in_1_SE3[i+increment].inv()) for i in range(len(g_in_1_SE3)-increment)])
    change_in_2 = np.array([(g_in_2_SE3[i]).mul(g_in_2_SE3[i+increment].inv()) for i in range(len(g_in_2_SE3)-increment)])

    distance = np.array([(change_in_1[i]).distance(change_in_2[i]) for i in range(len(change_in_1))])
    
    return distance





# DEPRECATED

def sum_between_timestamps(timestamps_data, data, timestamps_goal):
    '''
    DEPRECATED
    Summing all measurements between all neighboring pairs of timestamps_goal

    return: sums array
    '''
    data_goal = np.empty((len(timestamps_goal), data.shape[1]))
    t_goal_cur = 0
    sum_cur = 0

    for i in range(len(timestamps_data)):
        sum_cur += data[i]
        if(timestamps_data[i] >= timestamps_goal[t_goal_cur]):
            data_goal[t_goal_cur] = sum_cur
            t_goal_cur += 1
            sum_cur = 0

    return data_goal

def difference_between_timestamps(timestamps_data, data, timestamps_goal):
    '''
    Calculating differences of data between all neighboring pairs of timestamps_goal

    return: array of data_future - data_past (like integral of change)
    '''
    data_goal = np.empty((len(timestamps_goal), data.shape[1]))
    t_goal_cur = 0
    i_start = 0

    for i in range(len(timestamps_data)):
        if(timestamps_data[i] >= timestamps_goal[t_goal_cur]):
            data_goal[t_goal_cur] = data[i] - data[i_start]
            t_goal_cur += 1
            i_start = i+1

    return data_goal

def generate_filter_from_q0(q_init, data_gyr, data_acc, freq, data_magn=None):
    '''
    Generates madgwick filter with initial quaternion = q_init
    If data_magn is provided, generates MARG filter
    If not, generates IMU filter

    return: madgwick filter object
    '''
    madgwick_iterative = ahrs.filters.Madgwick(q0=q_init)
    Quaternions = [q_init]    # Allocate first quaternion in an array

    for index in range(1, len(data_gyr)+1):
        if(data_magn==None):
            new_quaternion = madgwick_iterative.updateIMU(q=Quaternions[index-1],
                                                gyr=data_gyr[index-1],
                                                acc=data_acc[index-1],
                                                dt=1/freq)   
        else:
            new_quaternion = madgwick_iterative.updateMARG(q=Quaternions[index-1],
                                                gyr=data_gyr[index-1],
                                                acc=data_acc[index-1],
                                                magn=data_magn[index-1],
                                                dt=1/freq)  
        Quaternions.append(new_quaternion)

    Quaternions = ahrs.QuaternionArray(Quaternions)
    madgwick_iterative.Q = Quaternions

    return madgwick_iterative

def RPRE(quaternions1, quaternions2):
    '''
    Root Product Relative Error
    '''
    R1 = np.array([mrob.quat_to_so3(quat) for quat in np.roll(quaternions1, -1, axis=1)])
    R2 = np.array([mrob.quat_to_so3(quat) for quat in np.roll(quaternions2, -1, axis=1)])

    g = np.array([0, 0, -1]).reshape(-1, 1)                       # looking just for direction of g

    g_in_1 = np.array([np.linalg.inv(R) @ g for R in R1])
    g_in_2 = np.array([np.linalg.inv(R) @ g for R in R2])
    
    g_in_1_SE3 = np.array([mrob.SE3(mrob.SO3(), g_in_1_i) for g_in_1_i in g_in_1])
    g_in_2_SE3 = np.array([mrob.SE3(mrob.SO3(), g_in_1_2) for g_in_1_2 in g_in_2])

    d12 = np.array([mrob.SE3.distance_trans(g_in_1_SE3[i], g_in_2_SE3[i]) for i in range(len(g_in_1_SE3))])
    d1 = np.array([mrob.SE3.distance_trans(g_in_1_SE3[0], g_in_1_SE3[i]) for i in range(len(g_in_1_SE3))])
    d2 = np.array([mrob.SE3.distance_trans(g_in_2_SE3[0], g_in_2_SE3[i]) for i in range(len(g_in_2_SE3))])

    rrse = np.array([d12[i] / np.sqrt(d1[i]*d2[i]) for i in range(1, len(d12))])

    return rrse
