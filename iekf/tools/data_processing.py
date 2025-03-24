# -*- coding: utf-8 -*-
'''
@file data_processing
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.spatial.transform import Slerp, Rotation
from scipy.interpolate import CubicSpline
plt.rcParams['font.size'] = 14
# Seed random generator
GENERATOR = np.random.default_rng(42)

def import_tum_imu(filename, smoothing=True):
    '''
    Imports and smooths (savgol, cubic spline) data
    Importing t, 
    ang_vel, acc,
    data from 1, 2, 3, 4, ... columns of the .csv file 

    return: data_t - array of timestamps
            data_gyr_xyz - array of x, y, z angular velocities for each timestamp 
            data_acc_xyz - array of x, y, z accelerations for each timestamp 
    '''
    data_np = pd.read_csv(filename, header=1).to_numpy()
    data_t = data_np[:, 0] / 1e9                            # nanoseconds
    data_gyr_xyz = np.empty((len(data_t), 3))
    data_acc_xyz = np.empty((len(data_t), 3))
    data_gyr_xyz = data_np[:, 1:4]
    data_acc_xyz = data_np[:, 4:7]

    if smoothing:
        data_gyr_xyz = smooth_and_resample_imu(data_t, data_t, data_gyr_xyz)
        data_acc_xyz = smooth_and_resample_imu(data_t, data_t, data_acc_xyz)

    return data_t, data_gyr_xyz, data_acc_xyz

def import_tum_mocap(filename, smoothing=True):
    '''
    Importing and smoothing (savgol, SLERP) mocap data from .csv file with order: 
    time, translation, rotation quaternion [w, x, y, z]

    return: data_quat_t, data_quat_R [w, x, y, z], data_quat_T
    '''
    data_quat_np = pd.read_csv(filename, header=1).dropna().to_numpy()
    data_quat_t = data_quat_np[:, 0].astype(np.float64) / 1e9                           # nanoseconds
    data_quat_R = np.empty((len(data_quat_t), 4))
    data_quat_T = np.empty((len(data_quat_t), 3))
    data_quat_T[:, 0] = data_quat_np[:, 1]
    data_quat_T[:, 1] = data_quat_np[:, 2]
    data_quat_T[:, 2] = data_quat_np[:, 3]
    data_quat_R[:, 0] = data_quat_np[:, 4]
    data_quat_R[:, 1] = data_quat_np[:, 5]
    data_quat_R[:, 2] = data_quat_np[:, 6]
    data_quat_R[:, 3] = data_quat_np[:, 7]

    if smoothing:
        data_quat_R = smooth_and_resample_quats(data_quat_t, data_quat_t, data_quat_R)

    return data_quat_t, data_quat_R, data_quat_T

def import_combined_data(filename, smoothing=True):
    '''
    Imports and smooths (savgol, cubic spline) data
    Importing t, 
    ang_vel_uncal_x[rad/s], ang_vel_uncal_y[rad/s], ang_vel_uncal_z[rad/s],
    acc_uncal_x[m/s^2],     acc_uncal_y[m/s^2],     acc_uncal_z[m/s^2],
    mfield_uncal_x[uT],     mfield_uncal_y[uT],     mfield_uncal_z[uT],
    ang_vel_drift_x[rad/s], ang_vel_drift_y[rad/s], ang_vel_drift_z[rad/s],
    acc_bias_x[m/s^2],      acc_bias_y[m/s^2],      acc_bias_z[m/s^2],
    mfield_bias_x[uT],      mfield_bias_y[uT],      mfield_bias_z[uT]
    data from 1, 2, 3, 4, ... columns of the .csv file 

    return: data_t - array of timestamps
            data_gyr_xyz - array of x, y, z angular velocities for each timestamp 
            data_acc_xyz - array of x, y, z accelerations for each timestamp 
            data_magn_xyz - array of x, y, z of magnet field vector for each timestamp 
    '''
    data_np = pd.read_csv(filename, header=1).to_numpy()
    data_t = data_np[:, 1]
    # For some reason t is not in ms, but in 1e-4 * ms
    # was for old version, now ok
    #data_t = data_t/1e4
    data_gyr_uncal_xyz = np.empty((len(data_t), 3))
    data_acc_uncal_xyz = np.empty((len(data_t), 3))
    data_magn_uncal_xyz = np.empty((len(data_t), 3))
    drift_gyr_xyz = np.empty((len(data_t), 3))
    drift_acc_xyz = np.empty((len(data_t), 3))
    drift_magn_xyz = np.empty((len(data_t), 3))
    data_gyr_xyz = np.empty((len(data_t), 3))
    data_acc_xyz = np.empty((len(data_t), 3))
    data_magn_xyz = np.empty((len(data_t), 3))
    data_gyr_uncal_xyz = data_np[:, 2:5]
    data_acc_uncal_xyz = data_np[:, 5:8]
    data_magn_uncal_xyz = data_np[:, 8:11]
    drift_gyr_xyz = data_np[:, 11:14]
    drift_acc_xyz = data_np[:, 14:17]
    drift_magn_xyz = data_np[:, 17:20]

    # Removing drift
    data_gyr_xyz = data_gyr_uncal_xyz - drift_gyr_xyz
    data_acc_xyz = data_acc_uncal_xyz - drift_acc_xyz
    data_magn_xyz = data_magn_uncal_xyz - drift_magn_xyz

    if smoothing:
        data_gyr_xyz = smooth_and_resample_imu(data_t, data_t, data_gyr_xyz)
        data_acc_xyz = smooth_and_resample_imu(data_t, data_t, data_acc_xyz)
        data_magn_xyz = smooth_and_resample_imu(data_t, data_t, data_magn_xyz)

    return data_t, data_gyr_xyz, data_acc_xyz, data_magn_xyz

def interpolate_data(t_goal, t1, t2, data1, data2):
    '''
    Interpolating data between t1 and t2 to t_goal

    param: t_goal - time to interpolate data to
    
    return: data interpolated to t_goal
    '''
    return data1 + (data2 - data1) / (t2 - t1) * (t_goal - t1)

def sync_s2_to_s1(t_start, t_s1, t_s2, data_s2):
    '''
    Synchronizing s2 data (sensor2) to s1 timestamps, starting from t_start

    param: t_start - time to start interpolating from it
    param: t_s1 - array of timestamps from sensor 1
    param: t_s2 - array of timestamps from sensor 2
    param: data_s2 - array of data from sensor 2
    
    return: array of data_s2 interpolated to t_s1
    '''

    # Looking for t_start in t_s2
    i_start = 0
    for i in range(len(data_s2)):
        if t_s2[i] >= t_start:
            i_start = i
            break

    data_s2_sync = data_s2[i_start::].copy()
    t_s2_sync = t_s2[i_start::].copy()

    if(i_start > 0): data_s2_sync[0] = interpolate_data(t_s1[0], t_s2[i_start - 1], t_s2[i_start], data_s2[i_start - 1], data_s2[i_start])

    # interpolating data
    i = 1
    shift = 0
    while i < len(data_s2_sync):
        if(i+shift >= len(t_s2_sync) or i >= len(t_s1)): break
        if(t_s1[i] < t_s2_sync[i-1+shift]):                         # shifting t_s2 window to the left
            shift -= 1
            continue
        elif(t_s1[i] > t_s2_sync[i+shift]):                         # shifting t_s2 window to the right
            shift += 1
            continue
        
        data_s2_sync[i] = interpolate_data(t_s1[i], t_s2_sync[i - 1 + shift], t_s2_sync[i + shift], data_s2_sync[i - 1 + shift], data_s2_sync[i + shift])

        i += 1

    return data_s2_sync[:i]

def sync_data(t_base, t_gyr, d_gyr, t_acc, d_acc, t_magn, d_magn):
    '''
    Synchronizing gyroscope, accelerometer and magnetometer data to 
    the t_base timestamps

    param: t_base - time to interpolate to
    
    return: t_base, d_gyr_sync, d_acc_sync, d_magn_sync
     - timestamps and data of synchronised measurements
    '''
    # Looking for min common timestamp
    t_start = max(t_gyr[0], t_acc[0], t_magn[0])
    t_base_start = np.argmax(t_base >= t_start)
    t_base = t_base[t_base_start::]

    # Sunchronizing data
    d_gyr_sync = sync_s2_to_s1(t_start, t_base, t_gyr, d_gyr)
    d_acc_sync = sync_s2_to_s1(t_start, t_base, t_acc, d_acc)
    d_magn_sync = sync_s2_to_s1(t_start, t_base, t_magn, d_magn)

    # Cutting leftovers
    min_len = min(len(d_gyr_sync), len(d_acc_sync), len(d_magn_sync), len(t_base))
    
    t_base = t_base[:min_len:]
    d_gyr_sync = d_gyr_sync[:min_len:]
    d_acc_sync = d_acc_sync[:min_len:]
    d_magn_sync = d_magn_sync[:min_len:]

    return t_base, d_gyr_sync, d_acc_sync, d_magn_sync


def import_mocap_data(filename, smoothing=True):
    '''
    Importing and smoothing (savgol, SLERP) mocap data from .csv file with order: 
    time, rotation quaternion [x, y, z, w], translation, markers xyz

    return: data_quat_t, data_quat_R [w, x, y, z], data_quat_T, data_quat_Markers_xyz
    '''
    data_quat = pd.read_csv(filename, header=2).to_numpy()
    data_quat_headers = data_quat[0][2:] + " " + data_quat[2][2:] + " " + data_quat[3][2:]
    
    data_quat_np = pd.read_csv(filename, header=6).dropna().to_numpy()
    data_quat_t = data_quat_np[:, 1].astype(np.float64)
    data_quat_R = np.empty((len(data_quat_t), 4))
    data_quat_T = np.empty((len(data_quat_t), 3))
    data_quat_Markers_xyz = np.empty((len(data_quat_t), len(data_quat_headers) - 7))
    data_quat_R[:, 0] = data_quat_np[:, 5]
    data_quat_R[:, 1] = data_quat_np[:, 2]
    data_quat_R[:, 2] = data_quat_np[:, 3]
    data_quat_R[:, 3] = data_quat_np[:, 4]
    data_quat_T[:, 0] = data_quat_np[:, 6]
    data_quat_T[:, 1] = data_quat_np[:, 7]
    data_quat_T[:, 2] = data_quat_np[:, 8]

    for i in range(data_quat_Markers_xyz.shape[1]):
        data_quat_Markers_xyz[:, i] = data_quat_np[:, 9 + i]

    if smoothing:
        data_quat_R = smooth_and_resample_quats(data_quat_t, data_quat_t, data_quat_R)

    return data_quat_t, data_quat_R, data_quat_T, data_quat_Markers_xyz

def import_gamerotvec_data(filename, smoothing=True):
    '''
    Importing game rotation vector data from .csv file with order: time, rotation quaternion [w x y z]

    return: data_quat_t, data_quat_R
    '''
    data_quat_np = pd.read_csv(filename, header=1).dropna().to_numpy()
    data_quat_t = data_quat_np[:, 0].astype(np.float64)
    data_quat_R = np.empty((len(data_quat_t), 4))
    data_quat_R[:, 0] = data_quat_np[:, 5]
    data_quat_R[:, 1] = data_quat_np[:, 2]
    data_quat_R[:, 2] = data_quat_np[:, 3]
    data_quat_R[:, 3] = data_quat_np[:, 4]

    if smoothing:
        data_quat_R = smooth_and_resample_quats(data_quat_t, data_quat_t, data_quat_R)

    return data_quat_t, data_quat_R

def trim_to_same_time_interval(t_mocap, d_mocap, t_data_zeroed, d_gyr_sync, d_acc_sync, d_magn_sync):
    '''
    Cutting all data and time to be in the same interval of times from common t_min to common t_max

    return: t_mocap, d_mocap, t_data_zeroed, d_gyr_sync, d_acc_sync, d_magn_sync
    '''
    
    # Looking for min common timestamp
    min_common_timestamp = max(t_mocap[0], t_data_zeroed[0])
    min_common_timestamp_id_data = np.argmax(t_data_zeroed >= min_common_timestamp)
    min_common_timestamp_id_mocap = np.argmax(t_mocap >= min_common_timestamp)

    # Looking for max common timestamp
    max_common_timestamp = min(t_mocap[-1], t_data_zeroed[-1])
    max_common_timestamp_id_data = np.argmax(t_data_zeroed >= max_common_timestamp)
    max_common_timestamp_id_mocap = np.argmax(t_mocap >= max_common_timestamp)

    # Cutting time and data in t > max_common_timestamp
    t_data_zeroed = t_data_zeroed[min_common_timestamp_id_data:max_common_timestamp_id_data+1]          # including last, which is = max_common_timestamp
    d_gyr_sync = d_gyr_sync[min_common_timestamp_id_data:max_common_timestamp_id_data+1]
    d_acc_sync = d_acc_sync[min_common_timestamp_id_data:max_common_timestamp_id_data+1]
    d_magn_sync = d_magn_sync[min_common_timestamp_id_data:max_common_timestamp_id_data+1]
    t_mocap = t_mocap[min_common_timestamp_id_mocap:max_common_timestamp_id_mocap+1]
    d_mocap = d_mocap[min_common_timestamp_id_mocap:max_common_timestamp_id_mocap+1]

    return t_mocap, d_mocap, t_data_zeroed, d_gyr_sync, d_acc_sync, d_magn_sync

def trim_to_min_length(*arrays):
    """
    Trims all input arrays along the first dimension to match the minimal common length.
    
    param: *arrays - Arbitrary number of numpy arrays with different shapes.
    
    return: list of np.ndarray: List of trimmed arrays with the same first-dimension length.
    """
    if not arrays:
        raise ValueError("At least one array must be provided.")
    
    # Find the minimum length along the first dimension
    max_common_len = min(len(arr) for arr in arrays)
    
    # Trim all arrays to this length
    trimmed_arrays = [arr[:max_common_len] for arr in arrays]
    
    return trimmed_arrays

def arrays_from_i(i, *arrays):
    """
    Returns all input arrays starting from i.
    
    param: *arrays - Arbitrary number of numpy arrays with different shapes.
    para,: i - First index to be included in arrays
    
    return: list of np.ndarray: List of trimmed arrays starting from i.
    """
    if not arrays:
        raise ValueError("At least one array must be provided.")
    
    # Trim all arrays
    trimmed_arrays = [arr[i:] for arr in arrays]
    
    return trimmed_arrays

def downsample(i:int, *arrays):
    """
    Downsamples input arrays i times.
    
    param: *arrays - Arbitrary number of numpy arrays with different shapes.
    para,: i - downsample factor - each i sample will be included
    
    return: list of np.ndarray: List of i times downsampled arrays.
    """
    if not arrays:
        raise ValueError("At least one array must be provided.")
    
    # Trim all arrays
    downsampled_arrays = [arr[::i] for arr in arrays]
    
    return downsampled_arrays

def smooth_and_resample_imu(goal_timestamps, data_timestamps, data, window_length=15, polyorder=2):
    """
    Smooths 1D IMU data (angular velocities and accelerations) using the Savitzky-Golay filter
    and resamples it to goal_timestamps using cubic spline interpolation.
    
    param: goal_timestamps (array-like): Target timestamps for resampled data.
    param: data_timestamps (array-like): Original timestamps corresponding to data.
    param: data (array-like, shape nx3): Original IMU data (angular velocities or accelerations).
    param: window_length (int): The length of the filter window (must be odd and > polyorder).  - take between 5 and 15
    param: polyorder (int): The order of the polynomial to fit.
    
    return: np.ndarray: Smoothed and resampled IMU data (shape len(goal_timestamps) x 3).
    """
    # Ensure window_length is appropriate
    if window_length % 2 == 0:
        window_length += 1  # Make it odd
    window_length = min(window_length, len(data))  # Ensure it's not larger than data length
    
    if window_length <= polyorder:
        raise ValueError("window_length must be greater than polyorder")
    
    # Apply Savitzky-Golay smoothing to each IMU component separately
    smoothed_data = np.column_stack([
        savgol_filter(data[:, i], window_length, polyorder) for i in range(data.shape[1])
    ])
    
    # Use cubic spline interpolation
    resampled_data = np.column_stack([
        CubicSpline(data_timestamps, smoothed_data[:, i])(goal_timestamps) for i in range(data.shape[1])
    ])
    
    return resampled_data

def smooth_and_resample_quats(goal_timestamps, data_timestamps, data, window_length=5, polyorder=2):
    """
    Smooths 1D quaternion data using the Savitzky-Golay filter and resamples it to goal_timestamps
    using Spherical Linear Interpolation (SLERP) for accurate quaternion interpolation.
    
    param: goal_timestamps (array-like): Target timestamps for resampled data.
    param: data_timestamps (array-like): Original timestamps corresponding to data.
    param: data (array-like, shape nx4): Original quaternion data.
    param: window_length (int): The length of the filter window (must be odd and > polyorder).  - take between 5 and 15
    param: polyorder (int): The order of the polynomial to fit.
    
    return: np.ndarray: Smoothed and resampled quaternion data.
    """
    # Ensure window_length is appropriate
    if window_length % 2 == 0:
        window_length += 1  # Make it odd
    window_length = min(window_length, len(data))  # Ensure it's not larger than data length
    
    if window_length <= polyorder:
        raise ValueError("window_length must be greater than polyorder")
    
    # Apply Savitzky-Golay smoothing to each quaternion component separately
    smoothed_quats = np.column_stack([
        savgol_filter(data[:, i], window_length, polyorder) for i in range(4)
    ])
    
    # Normalize quaternions to maintain unit norm
    smoothed_quats /= np.linalg.norm(smoothed_quats, axis=1, keepdims=True)
    
    # Convert smoothed quaternions to Rotation objects
    smoothed_rotations = Rotation.from_quat(smoothed_quats)
    
    # goal_timestamps timestamps must be within data_timestamps with borders inclusive
    i_min = np.argmax(goal_timestamps >= data_timestamps[0])
    i_max = np.argmax(goal_timestamps > data_timestamps[-1])
    # If there are no elements > data_timestamps[-1]
    if i_max == 0 and goal_timestamps[0] <= data_timestamps[-1]:
        i_max = len(goal_timestamps)

    # Perform SLERP interpolation
    slerp = Slerp(data_timestamps, smoothed_rotations)
    resampled_rotations = slerp(goal_timestamps[i_min:i_max])
    
    return resampled_rotations.as_quat()


def sync_mocap_and_data(t_mocap, d_mocap, t_data_zeroed, d_gyr_sync, d_acc_sync, d_magn_sync):
    '''
    Synchronizing mocap and smartphone data to the common frequency - lowest of these two.
    Synchronization of smartphone data is done with Savitzky-Golay filter and Cubic Spline interpolation
    Synchronization of mocap data is done with Savitzky-Golay filter and Spherical Linear Interpolation (SLERP)
    '''
    t_mocap, d_mocap, t_data_zeroed, d_gyr_sync, d_acc_sync, d_magn_sync = trim_to_same_time_interval(
        t_mocap, d_mocap, t_data_zeroed, d_gyr_sync, d_acc_sync, d_magn_sync
    )

    # Synching smartphone to mocap
    if(len(t_mocap) < len(t_data_zeroed)):
        t_sync = t_mocap.copy()
        #d_gyr_sync_mocap = sync_s2_to_s1(t_start, t_sync, t_data_zeroed, d_gyr_sync)
        #d_acc_sync_mocap = sync_s2_to_s1(t_start, t_sync, t_data_zeroed, d_acc_sync)
        #d_magn_sync_mocap = sync_s2_to_s1(t_start, t_sync, t_data_zeroed, d_magn_sync)
        d_gyr_sync_mocap = smooth_and_resample_imu(t_sync, t_data_zeroed, d_gyr_sync)
        d_acc_sync_mocap = smooth_and_resample_imu(t_sync, t_data_zeroed, d_acc_sync)
        d_magn_sync_mocap = smooth_and_resample_imu(t_sync, t_data_zeroed, d_magn_sync)

        t_sync, d_gyr_sync_mocap, d_acc_sync_mocap, d_magn_sync_mocap, d_mocap_sync = trim_to_min_length(
            t_sync, d_gyr_sync_mocap, d_acc_sync_mocap, d_magn_sync_mocap, d_mocap
        )

    # Synching mocap to smartphone
    else:
        t_sync = t_data_zeroed.copy()
        #d_mocap_sync = sync_s2_to_s1(t_start, t_sync, t_mocap, d_mocap)
        d_mocap_sync = smooth_and_resample_quats(t_sync, t_mocap, d_mocap)

        t_sync, d_gyr_sync_mocap, d_acc_sync_mocap, d_magn_sync_mocap, d_mocap_sync = trim_to_min_length(
            t_sync, d_gyr_sync, d_acc_sync, d_magn_sync, d_mocap_sync
        )

    return t_sync, d_gyr_sync_mocap, d_acc_sync_mocap, d_magn_sync_mocap, d_mocap_sync


def apply_timeshift(data, i_shift, trim_end=False):
    '''
    Shifting data in time to apply time_offset, calculated by TwistnSync
    Positive i_shift belongs to later start of recording

    param: data - array of time or data
    param: i_shift - time_offset in terms of indices in times array
    param: trim_end - if True, trim end of data with i_shift < 0, to change arrays len the same
    '''

    if(i_shift >= 0):
        data_sync = data[i_shift:]         # need to delete tail of data moved to beginning
    else:
        if trim_end:
            data_sync = data[:len(data) - i_shift]
        else:
            data_sync = data
    return data_sync.copy()
