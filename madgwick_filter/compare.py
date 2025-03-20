# -*- coding: utf-8 -*-
'''
@file compare
'''

from madgwick_filter.tools_ahrs import plot
from madgwick_filter.tools_ahrs import plot3
import ahrs
import twistnsync as tns
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
# Seed random generator
GENERATOR = np.random.default_rng(42)

from madgwick_filter import utils, transform, data_processing


def plot_sm_and_mocap_angvel(sm_sensors_time, sm_sensors_data, mocap_time, mocap_data, plot_time=False, steady_state_samples=-1):
    '''
    Plot smartphone and mocap data

    param: plot_time - if False (default), x axis is just number of timestamps
                       if True, x axis shows time 
    param: steady_state_samples - number of samples to be taken as steady bias to remove bias
    '''
    plt.figure(figsize=(10,5))
    plots_bias = np.array([3.0, 0.0, -3.0])
    sm_color = 'b'
    mocap_color = 'purple'
    if not plot_time:
        sm_lines = plt.plot(sm_sensors_data + plots_bias, c=sm_color, linewidth=2)
        mcu_lines = plt.plot(mocap_data + plots_bias, c=mocap_color, linewidth=2)
    else:
        sm_lines = plt.plot(sm_sensors_time, sm_sensors_data + plots_bias, c=sm_color, linewidth=2)
        mcu_lines = plt.plot(mocap_time, mocap_data + plots_bias, c=mocap_color, linewidth=2)

    if steady_state_samples != -1:
        plt.axvline(steady_state_samples, c='g', linewidth=3)

    plt.text(0, plots_bias[0]+0.1, r'$w_x$-axis'); plt.text(0, plots_bias[1]+0.1, r'$w_y$-axis'); plt.text(0, plots_bias[2]+0.1, r'$w_z$-axis')
    plt.title('Smartphone and Mocap data')
    plt.xlabel(r'Sample, #')
    plt.ylabel(r'Angular velocities')
    plt.grid()
    plt.legend([sm_lines[0], mcu_lines[0]], ['Smartphone gyro data', 'Mocap gyro data'])
    plt.show()

def errors_estimation(quaterniions1, quaterniions2, source1="smartphone", source2="mocap"):
    '''
    Estimating APE, RPE of Attitudes in SO3 and of gravity vectors in these SO3
    '''
    # Checking if rotation quaternions align
    madgwick_distance = utils.calculate_quat_distances(quaterniions1, quaterniions2)
    print("Mean distance between " + source1 + " and " + source2 + " attitude estimations:")
    print(np.mean(madgwick_distance))
    print("Distance between " + source1 + " and " + source2 + " attitude estimations:")
    plot(madgwick_distance)
    print("Last distance between " + source1 + " and " + source2 + " attitude estimations:")
    print(madgwick_distance[-1])
    print("Norm of quaternions - shall be = 1")
    print(np.linalg.norm(quaterniions2, axis=1))
    plot(np.linalg.norm(quaterniions2, axis=1))

    # Checking if gravity vector, rotated by quaternions, align
    d_g = utils.calculate_g_distances(quaterniions1, quaterniions2)
    print("Mean distance between " + source1 + " and " + source2 + " vector g estimations:")
    print(np.mean(d_g))
    print("Distance between " + source1 + " and " + source2 + " vector g estimations:")
    plot(d_g)
    print("Last distance between " + source1 + " and " + source2 + " vector g estimations:")
    print(d_g[-1])

    print("Mean Relative Pose Error")
    rpe = utils.RPE(quaterniions1, quaterniions2)
    print(np.mean(rpe))
    print("Relative Pose Error")
    plot(rpe)
    print("Last RPE between " + source1 + " and " + source2 + " attitude estimations:")
    print(rpe[-1])

    print("Mean Relative Pose Error of vector g")
    rpe_g = utils.RPE_g(quaterniions1, quaterniions2)
    print(np.mean(rpe_g))
    print("Relative Pose Error of vector g")
    plot(rpe_g)
    print("Last RPE between " + source1 + " and " + source2 + " vector g estimations:")
    print(rpe_g[-1])
    return madgwick_distance, d_g, rpe, rpe_g

def compare_smartphone_to_mocap(smartphone_time, smartphone_quats, smartphone_gyros, mocap_time, mocap_quats, steady_state_samples, 
                                gyro=True, steady_end=False, source1="smartphone", source2="mocap"):
    '''
    Provides plots, bias, timeshift, translation matrix and errors
    Just a big summary of smartphone-to-mocap comparison, taking mocap as true trajectory
    '''
    
    if(gyro):                   # synching mocap with gyro data
        omega_sm = smartphone_gyros[:-1].copy()
    else:                       # synching mocap with filter (d/dt) quats data
        omega_sm = transform.quats_to_angvels(smartphone_time, smartphone_quats)
        omega_sm = data_processing.smooth_and_resample_imu(smartphone_time[:-1], smartphone_time[:-1], omega_sm)
    omega_mocap = transform.quats_to_angvels(mocap_time, mocap_quats)
    omega_mocap = data_processing.smooth_and_resample_imu(mocap_time[:-1], mocap_time[:-1], omega_mocap)

    # Plotting angular velocities from initial data
    print("angular velocities:")
    plot_sm_and_mocap_angvel(smartphone_time[:-1], omega_sm, mocap_time[:-1], omega_mocap)
    print("Assuming common time domain:")
    plot_sm_and_mocap_angvel(smartphone_time[:-1], omega_sm, mocap_time[:-1], omega_mocap, plot_time=True)
    print("Removing bias:")

    if not steady_end:
        plot_sm_and_mocap_angvel(smartphone_time[:-1], omega_sm, mocap_time[:-1], omega_mocap, plot_time=False, steady_state_samples=steady_state_samples)
        smartphone_bias = omega_sm[:steady_state_samples].mean(axis=0)
        mocap_bias = omega_mocap[:steady_state_samples].mean(axis=0)
    else:
        plot_sm_and_mocap_angvel(smartphone_time[:-1], omega_sm, mocap_time[:-1], omega_mocap, plot_time=False, steady_state_samples=len(omega_sm)-steady_state_samples)
        smartphone_bias = omega_sm[-steady_state_samples:].mean(axis=0)
        mocap_bias = omega_mocap[-steady_state_samples:].mean(axis=0)

    print('Smartphone bias:', smartphone_bias)
    print('Mocap bias:', mocap_bias)

    # Removing bias
    omega_sm_unbiased = omega_sm - smartphone_bias
    omega_mocap_unbiased = omega_mocap - mocap_bias

    # Synchronizing angular velocities in time and geometry
    time_sync = tns.TimeSync(
        omega_sm_unbiased,              # Unbiased smartphone gyro samples
        omega_mocap_unbiased,           # Unbiased Mocap samples
        smartphone_time[:-1],           # Smartphone gyro timestamps
        mocap_time[:-1]                 # Mocap timestamps
    )

    time_sync.resample(step=None)
    time_sync.obtain_delay()
    computed_offset = time_sync.time_delay
    time_offset = smartphone_time[0] - mocap_time[0] - computed_offset

    print('Time offset between smartphone and MCU clocks is', time_offset, 's')
    print("Time synchronized data:")

    # Calculating time_offset 
    # (smartphone_recording_start_time - mocap_recording_start_time) 
    # in terms of timestamps array indices
    if(computed_offset < 0): 
        i_shift = np.argmax(smartphone_time - smartphone_time[0] > -computed_offset)         # Zeroing time in case it starts not from 0. Then any time > time_offset
    else:
        i_shift = -np.argmax(smartphone_time - smartphone_time[0] > computed_offset)
    print("I shift = " + str(i_shift))

    # Plotting synchronized in time and geometry angular velocities 
    plot_sm_and_mocap_angvel(smartphone_time[:-1], omega_sm, mocap_time[:-1] + time_offset, omega_mocap, plot_time=True)
    omega_sm_aligned = (time_sync.M@omega_sm.T).T
    print("Transformation matrix M:")
    print(time_sync.M)
    print("Time and geometry synchronized data:")
    plot_sm_and_mocap_angvel(smartphone_time[:-1], omega_sm_aligned, mocap_time[:-1] + time_offset, omega_mocap, plot_time=True)

    smartphone_time_sync = data_processing.apply_timeshift(smartphone_time[:-1], i_shift)           # angvels are 1 sample shorter because they are d(x+1 - x)/d(t+1 - t)
    omega_sm_sync = data_processing.apply_timeshift(omega_sm_aligned, i_shift)
    mocap_time_sync = data_processing.apply_timeshift(mocap_time[:-1], -i_shift)
    omega_mocap_sync = data_processing.apply_timeshift(omega_mocap, -i_shift)
    smartphone_time_sync, omega_sm_sync, mocap_time_sync, omega_mocap_sync = data_processing.trim_to_min_length(smartphone_time_sync, omega_sm_sync, mocap_time_sync, omega_mocap_sync)

    print("Difference between time and geometry synchronized angular velocities:")
    print(np.mean(omega_sm_sync - omega_mocap_sync, axis=0))
    plot(omega_sm_sync - omega_mocap_sync)
    print("Time and geometry synchronized angular velocities:")
    plot(omega_sm_sync, omega_mocap_sync)

    # Integrating angular velocities back to quaternions
    quat_mocap_aligned = data_processing.apply_timeshift(mocap_quats, -i_shift)
    quat_sm_aligned = transform.angvels_to_quats(smartphone_time_sync, omega_sm_sync, quat_mocap_aligned[0])
    quat_sm_aligned, quat_mocap_aligned = data_processing.trim_to_min_length(quat_sm_aligned, quat_mocap_aligned)
    #quat_sm_aligned = smooth_and_resample_mocap(smartphone_time_sync, smartphone_time_sync, quat_sm_aligned)
    #quat_mocap_aligned = angvels_to_quats(mocap_time_sync, omega_mocap_sync, mocap_quats[0])
    print("Time and geometry synchronized quaternions, integrated from angular velocities:")
    plot(quat_sm_aligned, quat_mocap_aligned)

    errors_estimation(quat_sm_aligned, quat_mocap_aligned, source1, source2)
    
    return time_sync, quat_sm_aligned, quat_mocap_aligned

def compare_smartphone_to_gamerotvec(sm_time, sm_gyros, sm_accs, sm_magns, grv_time, grv_quats, freq=100, filter_IMU=True):
    '''
    Provides plots and difference between Madgwick filter from smartphone and game rotation vector
    Synchronizes smartphone and GRV in time, then calculates difference

    param: filter_IMU - if True, Magwick IMU is initialized. If false - Madgwick MARG

    return: t_data_sync, madgwick_filter, data_grv_quats
    '''

    # First we synchronize measurements in time
    t_data_sync, data_gyr, data_acc, data_magn, data_grv_quats = data_processing.sync_mocap_and_data(grv_time, grv_quats, sm_time, sm_gyros, sm_accs, sm_magns)
    
    if filter_IMU:
        madgwick= ahrs.filters.Madgwick(gyr=data_gyr,
                                 acc=data_acc,
                                 frequency=freq)
    else:
        madgwick = ahrs.filters.Madgwick(gyr=data_gyr,
                                 acc=data_acc,
                                 mag=data_magn,
                                 frequency=freq)
    
    print("Smartphone and Game Rotation Vector quaternions")
    plot(madgwick.Q, data_grv_quats)

    errors_estimation(madgwick.Q, data_grv_quats, source2="Game Rotation Vector")

    return t_data_sync, madgwick, data_grv_quats

def compare_gamerotvec_to_mocap(grv_time, grv_quats, mocap_time, mocap_quats, steady_state_samples=1000, steady_end=False, smoothing=False):
    '''
    Provides plots, bias, timeshift, translation matrix and errors
    Launches the same comparison as for smartphone_quats to mocap

    return: time_sync, quat_sm_aligned, quat_mocap_aligned
    '''
    if smoothing:
        grv_quats = data_processing.smooth_and_resample_quats(grv_time, grv_time, grv_quats)
        mocap_quats = data_processing.smooth_and_resample_quats(mocap_time, mocap_time, mocap_quats)
    
    time_sync, quat_sm_aligned, quat_mocap_aligned = compare_smartphone_to_mocap(
        grv_time, grv_quats, np.empty(1), mocap_time, mocap_quats, steady_state_samples, 
        gyro=False, steady_end=steady_end, source1="Game Rotation Vector"
    )

    return time_sync, quat_sm_aligned, quat_mocap_aligned


# DEPRECATED

def plot_sm_and_mocap_quat(sm_sensors_time, sm_sensors_data, mocap_time, mocap_data, plot_time=False, steady_state_samples=-1):
    '''
    Deprecated
    Plot smartphone and mocap data
    '''
    plt.figure(figsize=(10,5))
    plots_bias = np.array([3.0, 0.0, -3.0, -6.0])
    if(sm_sensors_data.shape[-1] == 3): plots_bias = np.array([3.0, 0.0, -3.0, -6.0])
    sm_color = 'b'
    mocap_color = 'purple'
    if not plot_time:
        sm_lines = plt.plot(sm_sensors_data + plots_bias, c=sm_color, linewidth=2)
        mcu_lines = plt.plot(mocap_data + plots_bias, c=mocap_color, linewidth=2)
    else:
        sm_lines = plt.plot(sm_sensors_time, sm_sensors_data + plots_bias, c=sm_color, linewidth=2)
        mcu_lines = plt.plot(mocap_time, mocap_data + plots_bias, c=mocap_color, linewidth=2)

    if steady_state_samples != -1:
        plt.axvline(steady_state_samples, c='g', linewidth=3)

    plt.text(0, plots_bias[0]+0.1, r'$W$-axis'); plt.text(0, plots_bias[1]+0.1, r'$I$-axis'); plt.text(0, plots_bias[2]+0.1, r'$J$-axis'); plt.text(0, plots_bias[3]+0.1, r'$K$-axis')
    plt.title('Smartphone and Mocap data')
    plt.xlabel(r'Sample, #')
    plt.ylabel(r'Rotation quaternion')
    plt.grid()
    plt.legend([sm_lines[0], mcu_lines[0]], ['Smartphone gyro+acc+magn data', 'Mocap data'])
    plt.show()

def compare_smartphone_to_mocap_quat(smartphone_time, smartphone_data, mocap_time, mocap_data, steady_state_samples):
    '''
    Deprecated
    !!! Incorrect from mathematical point of wiev - we can't rotate quaternion by SO3 @ Q
    Provides plots, bias, timeshift, translation matrix and mean error
    Just a big summary of smartphone-to-mocap comparison, taking mocap as true trajectory
    '''
    plot_sm_and_mocap_quat(smartphone_time, smartphone_data, mocap_time, mocap_data)
    print("Assuming common time domain:")
    plot_sm_and_mocap_quat(smartphone_time, smartphone_data, mocap_time, mocap_data, plot_time=True)
    print("Removing bias:")
    plot_sm_and_mocap_quat(smartphone_time, smartphone_data, mocap_time, mocap_data, plot_time=False, steady_state_samples=steady_state_samples)
    smartphone_bias = smartphone_data[:steady_state_samples].mean(axis=0)
    mocap_bias = mocap_data[:steady_state_samples].mean(axis=0)
    print('Smartphone gyro+acc+magn bias:', smartphone_bias)
    print('Mocap bias:', mocap_bias)
    smartphone_data_unbiased = smartphone_data - smartphone_bias
    mocap_data_unbiased = mocap_data - mocap_bias
    time_sync = tns.TimeSync(
        smartphone_data_unbiased,   # Unbiased smartphone gyro+acc+magn samples
        mocap_data_unbiased,        # Unbiased Mocap samples
        smartphone_time,            # Smartphone gyro+acc+magn timestamps
        mocap_time                  # Mocap timestamps
    )
    time_sync.resample(step=None)
    time_sync.obtain_delay()
    computed_offset = time_sync.time_delay
    time_offset = smartphone_time[0] - mocap_time[0] - computed_offset
    print('Time offset between smartphone and MCU clocks is', time_offset, 's')
    print("Time synchronized data:")
    plot_sm_and_mocap_quat(smartphone_time, smartphone_data, mocap_time + time_offset, mocap_data, plot_time=True)
    smartphone_data_aligned = (time_sync.M@smartphone_data.T).T
    print("Transformation matrix M:")
    print(time_sync.M)
    print("Time and geometry synchronized data:")
    plot_sm_and_mocap_quat(smartphone_time, smartphone_data_aligned, mocap_time + time_offset, mocap_data, plot_time=True)
    i_shift = np.argmax(smartphone_time > time_offset)
    madgwick_distance = utils.calculate_quat_distances(smartphone_data_aligned[i_shift::], mocap_data[:len(mocap_data)-i_shift:])
    print("Mean distance between smartphone and mocap attitude estimations:")
    print(np.mean(madgwick_distance))
    print("Smartphone and mocap attitude estimations:")
    plot(madgwick_distance)
    sm_quat_unbias, mocap_quat_unbias = data_processing.unbias_quat_array(smartphone_data_aligned, mocap_data, i_shift)
    madgwick_distance_unbias = utils.calculate_quat_distances(sm_quat_unbias, mocap_quat_unbias)
    print("Mean distance between UNBIASED smartphone and mocap attitude estimations:")
    print(np.mean(madgwick_distance_unbias))
    print("Smartphone and mocap UNBIASED attitude estimations:")
    plot(madgwick_distance_unbias)
    return time_sync

def trying_to_align_accelerations(t_accs_from_data, accs_from_data, time_sync_M):
    '''
    Unsuccessfull attempt to geometry-synchronize accelerations
    When we use Twistnsync for gyroscope data, to align smartphone and mocap data
    in time and geometry, we find time_offset and rotation matrix time_sync_M.
    But if we want to construct Madgwick filter from the synchronized data, we need to 
    time-geometry-synchronize at least the accelerometer.

    To do this, I've tried rotating acceleration vectors with the same time_sync_M matrix
    Strange behaviour of filter and incorrect result

    Also I've tried to rotate da/dt vectors, like the angular velocity of gravity vector rotation
    It gives the same problems, but additionally norms of accelerometer measurements become corrupted

    Substituting time_sync_M by time_sync_M.T (inverse) does not eliminate these problems
    '''
    print("Trying to synchronize accelerations directly, time_sync.M @ acceleration")
    accs_from_data_aligned = (time_sync_M@accs_from_data.T).T
    print("Initial vs synchronized accelerations")
    plot(accs_from_data, accs_from_data_aligned)
    print("Norm of initial and synchronized accelerations directly:")
    plot(np.linalg.norm(accs_from_data, axis=1), np.linalg.norm(accs_from_data_aligned, axis=1))
    print("--------------------------------------------------------------------------------------------")
    print("Initial accelerations:")
    plot(accs_from_data)
    accs_normalised = np.array([accs_from_data[i]/np.linalg.norm(accs_from_data[i]) for i in range(len(accs_from_data))])
    d_accs_vectors = transform.xs_to_dxs(t_accs_from_data, accs_normalised)
    print("First dericative of acceleration vectors:")
    plot(d_accs_vectors)
    d_accs_vectors_aligned = (time_sync_M@d_accs_vectors.T).T
    print("Synchronized first derivatives = (time_sync.M) @ (First dericative of acceleration vectors):")
    plot(d_accs_vectors_aligned)
    accs_aligned_vectors = transform.dxs_to_xs(t_accs_from_data, d_accs_vectors_aligned, accs_normalised[0])
    accs_aligned = np.array([accs_aligned_vectors[i]*np.linalg.norm(accs_from_data[i]) for i in range(len(accs_aligned_vectors))])
    #print("Synchronized accelerations (from integrating derivatives vectors):")
    #plot(accs_aligned)
    print("Norm of initial and synchronized accelerations from derivatives:")
    plot(np.linalg.norm(accs_from_data, axis=1), np.linalg.norm(accs_aligned, axis=1))
    print("Initial vs synchronized accelerations from derivatives")
    plot(accs_from_data, accs_aligned)

    return accs_from_data_aligned, accs_aligned