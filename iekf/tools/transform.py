# -*- coding: utf-8 -*-
'''
@file transform
'''

import mrob
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
# Seed random generator
GENERATOR = np.random.default_rng(42)

from tools import utils

def quat_to_angvel(t, t_next, quat, quat_next):
    '''
    Transforming quatenion to angular velocities
    First transform quaternions to SO3, then find delta_R and 
    use mrob.SO3.Ln to go from SO3 to R3 - angles
    and finally divide d_teta by d_time to get angular velocities
    '''
    # [w x y z] to [x y z w]
    quat = np.roll(quat, -1)
    quat_next = np.roll(quat_next, -1)

    R1 = mrob.quat_to_so3(quat)
    R2 = mrob.quat_to_so3(quat_next)

    delta_R = np.linalg.inv(R1) @ R2
    delta_teta = mrob.SO3.Ln(mrob.SO3(delta_R))
    delta_t = t_next - t

    ang_vel = delta_teta / delta_t

    return ang_vel

def quats_to_angvels(timestamps, quats):
    '''
    Transforming quatenions to angular velocities
    First transform quaternions to SO3, then find delta_R and 
    use mrob.SO3.Ln to go from SO3 to R3 - angles
    and finally divide d_teta by d_time to get angular velocities
    '''
    ang_vels = np.array([quat_to_angvel(timestamps[i], timestamps[i+1], quats[i], quats[i+1]) for i in range(len(timestamps) - 1)])

    return ang_vels

def angvel_to_quat(t, t_next, angvel, quat):
    '''
    Transforming angular velocities to quaternions
    First multiply angular velocities by d_t, then 
    use mrob.SO3 to go from R3 to SO3 with exponential mapping
    and finally get R_next = R @ delta_R and with mrob.so3_to_quat go to quaternion
    
    param: timestamps - array of timestamps for angvels
    param: angvel - array of angular velocitiy components [x, y, z]
    param: quat - initial attitude, [x, y, z, w]

    Return: quaternion of rotation with angvel in format [x, y, z, w]
    '''
    delta_t = t_next - t
    delta_teta = angvel * delta_t
    delta_R = mrob.SO3(delta_teta).R()

    R = mrob.quat_to_so3(quat)
    R_next = R @ delta_R

    quat_next = mrob.so3_to_quat(R_next)

    return quat_next

def angvels_to_quats(timestamps, angvels, quat0):
    '''
    Transforming angular velocities to quaternions
    First multiply angular velocities by d_t, then 
    use mrob.SO3 to go from R3 to SO3 with exponential mapping
    and finally get R_next = R @ delta_R and with mrob.so3_to_quat go to quaternion

    param: timestamps - array of timestamps for angvels
    param: angvels - array of angular velocities [x, y, z]
    param: quat0 - initial attitude, [w, x, y, z]
    
    Return: quaternions of rotations with angvels in format [w, x, y, z]
    '''
    quats = np.empty((len(timestamps), 4))
    quats[0] = np.roll(quat0, -1)           # transforming [w, x, y, z] -> [x, y, z, w]

    for i in range(1, len(timestamps)):
        quats[i] = angvel_to_quat(timestamps[i-1], timestamps[i], angvels[i-1], quats[i-1])
    
    return np.roll(quats, 1, axis=1)

def x_to_dx(t, t_next, x, x_next):
    '''
    Simple discrete dx/dt 

    return: dx
    '''
    delta_x = x_next - x
    delta_t = t_next - t
    dx = delta_x / delta_t

    return dx


def xs_to_dxs(timestamps, xs):
    '''
    Simple discrete dx/dt of an array

    return: dxs array
    '''
    dxs = np.array([x_to_dx(timestamps[i], timestamps[i+1], xs[i], xs[i+1]) for i in range(len(timestamps) - 1)])

    return dxs

def dx_to_x(t, t_next, dx, x):
    '''
    Simple discrete dx*dt 

    return: x_next
    '''
    delta_t = t_next - t
    delta_x = dx * delta_t
    x_next = x+delta_x

    return x_next


def dxs_to_xs(timestamps, dxs, x0):
    '''
    Simple discrete dx*dt of an array

    return: xs array
    '''
    xs = np.empty((len(timestamps), 3))
    xs[0] = x0

    for i in range(1, len(timestamps)):
        xs[i] = dx_to_x(timestamps[i-1], timestamps[i], dxs[i-1], xs[i-1])

    return xs

def quats_to_rpy(quats):
    '''
    Transforming quaternions [w, x, y, z] to roll pitch yaw angles

    Return: array of [roll, pitch, yaw]
    '''
    #rpys = np.empty((len(quats), 3))
    #for i in range(len(quats)):
    #    qw, qx, qy, qz = quats[i]
    #    roll = np.arctan2(2.0*(qy*qz - qw*qx), 2.0*qw*qw - 1 + 2.0*qz*qz)
    #    pitch = -np.arcsin(2.0*(qx*qz + qw*qy))
    #    yaw = np.arctan2(2.0*(qx*qy - qw*qz), 2.0*qw*qw - 1 + 2.0*qx*qx)
    #    rpys[i] = [roll, pitch, yaw]
    #return rpys
    #rotations = R.from_quat(np.roll(quats, -1, axis=1))          # transforming [w, x, y, z] -> [x, y, z, w]
    # [w x y z] to [x y z w]
    quats = np.roll(quats, -1, axis=1)

    R1 = np.array([mrob.quat_to_so3(quats[i]) for i in range(len(quats))])

    teta = np.array([mrob.SO3.Ln(mrob.SO3(R1[i])) for i in range(len(R1))])
    
    # Convert to RPY (XYZ Euler angles)
    #return rotations.as_euler('xyz')
    return teta

def rpy_to_quats(rpys):
    '''
    Transforming roll, pitch, yaw angles to quaternions [w, x, y, z]
    
    Parameters:
        rpys: Nx3 numpy array of [roll, pitch, yaw] angles in radians

    Returns:
        Nx4 numpy array of quaternions [w, x, y, z]
    '''
    # Create a Rotation object from RPY angles
    #rotations = R.from_euler('xyz', rpys)
    
    # Convert to quaternions (returns in [x, y, z, w] order)
    #quats = rotations.as_quat()
    
    # Reorder to [w, x, y, z]
    R = np.array([mrob.SO3(rpys[i]).R() for i in range(len(rpys))])

    quats = np.array([mrob.so3_to_quat(R[i]) for i in range(len(R))])
    quaternions = (np.roll(quats, 1, axis=1))      # transforming [x, y, z, w] -> [w, x, y, z] 
    
    return quaternions

# DEPRECATED

def data_to_wider_times(t_data, data, wider_times):
    '''
    Transforming data to smaller array with wider placed timestamps, keeping the data change between timestamps the same
    First integrating data, 
    then finding differences for every neighboring pair of wider_times, 
    and then derivating back, so the differences are concentrated in new time intervals

    return: array of data with the same change between timestamps
    '''
    data_int = dxs_to_xs(t_data, data, np.zeros_like(data[0]))
    data_int_summed = utils.difference_between_timestamps(t_data, data_int, wider_times)
    data_int_summed_deriv = np.append(data[0].reshape(1, -1), xs_to_dxs(wider_times, data_int_summed), axis = 0)

    return data_int_summed_deriv
