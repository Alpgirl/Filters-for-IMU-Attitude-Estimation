# -*- coding: utf-8 -*-
'''
@file utils
'''

from tools.tools_ahrs import plot
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

def angle_rot(vec1, vec2):
    '''
    Calculate single angle of rotation between two 3D vectors
    '''
    dot_product = np.dot(vec1.flatten(), vec2.flatten())
    cross_product = np.cross(vec1.flatten(), vec2.flatten())
    sin_theta = np.linalg.norm(cross_product)
    cos_theta = dot_product
    return np.arctan2(sin_theta, cos_theta)

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

    distance = np.array([angle_rot(g_in_1[i], g_in_2[i]) for i in range(len(g_in_1))])

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

def calc_cov(noised, ground_truth):
    '''
    Calculating noise covariance of noised data
    Given that noised = ground_truth + noise, noise cov is:
    E{(noise)*(noise).T}
    '''
    noise = noised - ground_truth
    return np.cov(noise.T)

    
def RMSE(errors):
    '''
    Calculate Root Mean Squared of the given errors array
    '''
    return np.sqrt(np.mean(np.power(errors, 2), axis=0))

def APE_RPY(rpys1, rpys2, abs=True):
    '''
    Absolute Pose Error of roll pitch yaw angles

    param: abs - Return absolute values of errors or not. Default is yes
    abs=False is useful for STD calculation, where we need mean(error), not mean(abs(error))
    '''
    if abs:
        return np.abs(rpys1 - rpys2)
    else:
        return rpys1 - rpys2

def vectors_to_rpy(vec1, vec2):
    '''
    Compute roll, pitch, yaw angles (XYZ euler angles) between two unit vectors
    
    param: vec1, vec2 - 3D unit vectors (numpy arrays)
    
    return: Roll, Pitch, Yaw angles in radians (numpy array)
    '''
    # Ensure unit vectors
    v1 = vec1.flatten() / np.linalg.norm(vec1)
    v2 = vec2.flatten() / np.linalg.norm(vec2)

    # Compute rotation axis and angle
    axis = np.cross(v1, v2)
    angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))  # Clip for numerical stability

    if np.linalg.norm(axis) < 1e-6:  # If vectors are already aligned
        return np.array([0.0, 0.0, 0.0])  # No rotation needed

    axis = axis / np.linalg.norm(axis)  # Normalize axis

    return axis * angle

def APE_g_RPY(rpys1, rpys2, abs=True):
    '''
    Absolute Pose Error of gravity vector estimations in terms of roll pitch yaw angles

    param: abs - Return absolute values of errors or not. Default is yes
    abs=False is useful for STD calculation, where we need mean(error), not mean(abs(error))
    '''
    R1 = np.array([mrob.SO3(rpy).R() for rpy in rpys1])
    R2 = np.array([mrob.SO3(rpy).R() for rpy in rpys2])

    g = np.array([0, 0, -1]).reshape(-1, 1)                      # looking just for direction of g
    
    g_in_1 = np.array([np.linalg.inv(R) @ g for R in R1])
    g_in_2 = np.array([np.linalg.inv(R) @ g for R in R2])

    rpys_g1_to_g2 = np.array([vectors_to_rpy(g_in_1[i], g_in_2[i]) for i in range(len(g_in_1))])

    if abs:
        rpys_g1_to_g2 = np.abs(rpys_g1_to_g2)
    return rpys_g1_to_g2

def RPE_RPY(rpys1, rpys2, increment=1, abs=True):
    '''
    Relative Pose Error of roll pitch yaw angles
    It shows how different are increments in rpys1 from increments in rpys2

    param: increment - number of steps between points in trajectory, which difference we are calculating

    param: abs - Return absolute values of errors or not. Default is yes
    abs=False is useful for STD calculation, where we need mean(error), not mean(abs(error))
    '''

    change_in_1 = np.array([rpys1[i] - rpys1[i+increment] for i in range(len(rpys1)-increment)])
    change_in_2 = np.array([rpys2[i] - rpys2[i+increment] for i in range(len(rpys2)-increment)])

    rpe_rpys = np.array([change_in_1[i] / change_in_2[i] for i in range(len(change_in_1))])

    if abs:
        rpe_rpys = np.abs(rpe_rpys)
    return rpe_rpys

def RPE_g_RPY(rpys1, rpys2, increment=1, abs=True):
    '''
    Relative Pose Error of gravity vector estimations in terms of roll pitch yaw angles
    It shows how different are increments of gravity vector in rpys1 from increments in rpys2

    param: increment - number of steps between points in trajectory, which difference we are calculating

    param: abs - Return absolute values of errors or not. Default is yes
    abs=False is useful for STD calculation, where we need mean(error), not mean(abs(error))
    '''

    R1 = np.array([mrob.SO3(rpy).R() for rpy in rpys1])
    R2 = np.array([mrob.SO3(rpy).R() for rpy in rpys2])

    g = np.array([0, 0, -1]).reshape(-1, 1)                      # looking just for direction of g
    
    g_in_1 = np.array([np.linalg.inv(R) @ g for R in R1])
    g_in_2 = np.array([np.linalg.inv(R) @ g for R in R2])

    change_in_1 = np.array([vectors_to_rpy(g_in_1[i], g_in_1[i+increment]) for i in range(len(g_in_1)-increment)])
    change_in_2 = np.array([vectors_to_rpy(g_in_2[i], g_in_2[i+increment]) for i in range(len(g_in_2)-increment)])

    rp_g_rpys = np.array([change_in_1[i] / change_in_2[i] for i in range(len(change_in_1))])

    if abs:
        rp_g_rpys = np.abs(rp_g_rpys)
    return rp_g_rpys

# DEPRECATED

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

def RPE_angvels(angvels1, angvels2, increment=1):
    '''
    Relative Pose Error
    It shows how different are increments in angvels1 from increments in angvels2

    param: increment - number of steps between points in trajectory, which difference we are calculating
    '''

    change_in_1 = np.array([angvels1[i] - angvels1[i+increment] for i in range(len(angvels1)-increment)])
    change_in_2 = np.array([angvels2[i] - angvels2[i+increment] for i in range(len(angvels2)-increment)])

    distance = np.array([change_in_1[i] / change_in_2[i] for i in range(len(change_in_1))])

    return distance
