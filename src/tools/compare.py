# -*- coding: utf-8 -*-
'''
@file compare
'''

from tools.tools_ahrs import plot
from tools.tools_ahrs import plot3
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
# Seed random generator
GENERATOR = np.random.default_rng(42)

from tools import utils


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

def errors_estimation(quaternions1, quaternions2, source1="smartphone", source2="mocap"):
    '''
    Estimating APE, RPE of Attitudes in SO3 and of gravity vectors in these SO3
    '''
    # Checking if rotation quaternions align
    madgwick_distance = utils.calculate_quat_distances(quaternions1, quaternions2)
    print("Mean distance between " + source1 + " and " + source2 + " attitude estimations:")
    madgwick_distance_mean = np.mean(madgwick_distance)
    print(madgwick_distance_mean)
    print("STD of distance between " + source1 + " and " + source2 + " attitude estimations:")
    madgwick_distance_std = np.std(madgwick_distance, ddof=1)
    print(madgwick_distance_std)                                        # ddof = 1 for unbiased std
    print("Distance between " + source1 + " and " + source2 + " attitude estimations:")
    plot(madgwick_distance)
    print("Last distance between " + source1 + " and " + source2 + " attitude estimations:")
    print(madgwick_distance[-1])
    print("RMSE of APE between " + source1 + " and " + source2 + " attitude estimations:")
    madgwick_distance_rmse = utils.RMSE(madgwick_distance)
    print(madgwick_distance_rmse)
    print("\n")
    #print("Norm of quaternions - shall be = 1")
    #print(np.linalg.norm(quaternions2, axis=1))
    #plot(np.linalg.norm(quaternions2, axis=1))

    # Checking if gravity vector, rotated by quaternions, align
    d_g = utils.calculate_g_distances(quaternions1, quaternions2)
    print("Mean distance between " + source1 + " and " + source2 + " vector g estimations:")
    d_g_mean = np.mean(d_g)
    print(d_g_mean)
    print("STD of distance between " + source1 + " and " + source2 + " vector g estimations:")
    d_g_std = np.std(d_g, ddof=1)
    print(d_g_std)                                        # ddof = 1 for unbiased std
    print("Distance between " + source1 + " and " + source2 + " vector g estimations:")
    plot(d_g)
    print("Last distance between " + source1 + " and " + source2 + " vector g estimations:")
    print(d_g[-1])
    print("RMSE of APE between " + source1 + " and " + source2 + " vector g estimations:")
    d_g_rmse = utils.RMSE(d_g)
    print(d_g_rmse)
    print("\n")

    print("Mean Relative Pose Error")
    rpe = utils.RPE(quaternions1, quaternions2)
    print("Mean distance between " + source1 + " and " + source2 + " attitude estimations:")
    rpe_mean = np.mean(rpe)
    print(rpe_mean)
    print("STD of distance between " + source1 + " and " + source2 + " attitude estimations:")
    rpe_std = np.std(rpe, ddof=1)
    print(rpe_std)                                        # ddof = 1 for unbiased std
    print("Relative Pose Error")
    plot(rpe)
    print("Last RPE between " + source1 + " and " + source2 + " attitude estimations:")
    print(rpe[-1])
    print("RMSE of RPE between " + source1 + " and " + source2 + " attitude estimations:")
    rpe_rmse = utils.RMSE(rpe)
    print(rpe_rmse)
    print("\n")

    print("Mean Relative Pose Error of vector g")
    rpe_g = utils.RPE_g(quaternions1, quaternions2)
    print("Mean distance between " + source1 + " and " + source2 + " vector g estimations:")
    rpe_g_mean = np.mean(rpe_g)
    print(rpe_g_mean)
    print("STD of distance between " + source1 + " and " + source2 + " vector g estimations:")
    rpe_g_std = np.std(rpe_g, ddof=1)
    print(rpe_g_std)                                        # ddof = 1 for unbiased std
    print("Relative Pose Error of vector g")
    plot(rpe_g)
    print("Last RPE between " + source1 + " and " + source2 + " vector g estimations:")
    print(rpe_g[-1])
    print("RMSE of RPE between " + source1 + " and " + source2 + " vector g estimations:")
    rpe_g_rmse = utils.RMSE(rpe_g)
    print(rpe_g_rmse)
    print("\n")

    return (madgwick_distance_mean, madgwick_distance_std, madgwick_distance_rmse, 
            d_g_mean, d_g_std, d_g_rmse, 
            rpe_mean, rpe_std, rpe_rmse, 
            rpe_g_mean, rpe_g_std, rpe_g_rmse
    )

def errors_estimation_rpy(rpys1, rpys2, source1="smartphone", source2="mocap"):
    '''
    Estimating APE, RPE of Attitudes in SO3 and of gravity vectors in these SO3
    '''
    # Checking if rotation quaternions align
    ape = utils.APE_RPY(rpys1, rpys2)
    ape_signed = utils.APE_RPY(rpys1, rpys2, abs=False)         # for STD
    print("Mean distance between " + source1 + " and " + source2 + " RPY attitude estimations:")
    ape_mean = np.mean(ape, axis=0)
    print(ape_mean)
    print("STD of distance between " + source1 + " and " + source2 + " RPY attitude estimations:")
    ape_std = np.std(ape_signed, ddof=1, axis=0)
    print(ape_std)                                              # ddof = 1 for unbiased std
                                                                # we use ape_signed because we need mean of error, not it's abs
    print("Distance between " + source1 + " and " + source2 + " RPY attitude estimations:")
    plot(ape)
    print("Last distance between " + source1 + " and " + source2 + " RPY attitude estimations:")
    print(ape[-1, :])
    print("RMSE of APE between " + source1 + " and " + source2 + " RPY attitude estimations:")
    ape_rmse = utils.RMSE(ape)
    print(ape_rmse)
    print("\n")
    #print("Norm of quaternions - shall be = 1")
    #print(np.linalg.norm(quaterniions2, axis=1))
    #plot(np.linalg.norm(quaterniions2, axis=1))

    # Checking if gravity vector, rotated by quaternions, align
    ape_g = utils.APE_g_RPY(rpys1, rpys2)
    ape_g_signed = utils.APE_g_RPY(rpys1, rpys2, abs=False)     # for STD
    print("Mean distance between " + source1 + " and " + source2 + " RPY vector g estimations:")
    ape_g_mean = np.mean(ape_g, axis=0)
    print(ape_g_mean)
    print("STD of distance between " + source1 + " and " + source2 + " RPY vector g estimations:")
    ape_g_std = np.std(ape_g_signed, ddof=1, axis=0)
    print(ape_g_std)                                            # ddof = 1 for unbiased std
                                                                # we use ape_signed because we need mean of error, not it's abs
    print("Distance between " + source1 + " and " + source2 + " RPY vector g estimations:")
    plot(ape_g)
    print("Last distance between " + source1 + " and " + source2 + " RPY vector g estimations:")
    print(ape_g[-1, :])
    print("RMSE of APE between " + source1 + " and " + source2 + " RPY vector g estimations:")
    ape_g_rmse = utils.RMSE(ape_g)
    print(ape_g_rmse)
    print("\n")

    print("Mean Relative Pose Error")
    rpe = utils.RPE_RPY(rpys1, rpys2)
    rpe_signed = utils.RPE_RPY(rpys1, rpys2, abs=False)         # for STD
    print("Mean distance between " + source1 + " and " + source2 + " RPY attitude estimations:")
    rpe_mean = np.mean(rpe, axis=0)
    print(rpe_mean)
    print("STD of distance between " + source1 + " and " + source2 + " RPY attitude estimations:")
    rpe_std = np.std(rpe_signed, ddof=1, axis=0)
    print(rpe_std)                                              # ddof = 1 for unbiased std
                                                                # we use ape_signed because we need mean of error, not it's abs
    print("Relative Pose Error RPY")
    plot(rpe)
    print("Last RPE between " + source1 + " and " + source2 + " RPY attitude estimations:")
    print(rpe[-1, :])
    print("RMSE of RPE between " + source1 + " and " + source2 + " RPY attitude estimations:")
    rpe_rmse = utils.RMSE(rpe)
    print(rpe_rmse)
    print("\n")

    print("Mean Relative Pose Error of vector g")
    rpe_g = utils.RPE_g_RPY(rpys1, rpys2)
    rpe_g_signed = utils.RPE_g_RPY(rpys1, rpys2, abs=False)     # for STD
    print("Mean distance between " + source1 + " and " + source2 + " RPY vector g estimations:")
    rpe_g_mean = np.mean(rpe_g, axis=0)
    print(rpe_g_mean)
    print("STD of distance between " + source1 + " and " + source2 + " RPY vector g estimations:")
    rpe_g_std = np.std(rpe_g_signed, ddof=1, axis=0)
    print(rpe_g_std)                                            # ddof = 1 for unbiased std
                                                                # we use ape_signed because we need mean of error, not it's abs
    print("Relative Pose Error RPY of vector g")
    plot(rpe_g)
    print("Last RPE between " + source1 + " and " + source2 + " RPY vector g estimations:")
    print(rpe_g[-1, :])
    print("RMSE of RPE between " + source1 + " and " + source2 + " RPY vector g estimations:")
    rpe_g_rmse = utils.RMSE(rpe_g)
    print(rpe_g_rmse)
    print("\n")

    return (ape_mean, ape_std, ape_rmse,
            ape_g_mean, ape_g_std, ape_g_rmse,
            rpe_mean, rpe_std, rpe_rmse,
            rpe_g_mean, rpe_g_std, rpe_g_rmse
            )

def ciplot(t, mu, minus_sigma, plus_sigma, x_real, color=None):
    """
    Plots a shaded region on a graph between specified lower and upper confidence intervals (L and U).

    :param t: The time series corresponding to the state.
    :param mu: The predicted state of the variable.
    :param minus_sigma: THe lower bound of the confidence interval.
    :param plus_sigma: The upper bound of the confidence interval.
    :param x_real: The real value of the state variable.
    :param color: Color of the fill inside the lower and upper bound curves (optional).
    :return handle: The handle to the plot of the state variable.
    """

    assert minus_sigma.shape[0] == plus_sigma.shape[0]
    assert t.shape[0] == mu.shape[0]

    plt.fill_between(t, minus_sigma, plus_sigma, color=color, alpha=0.5)
    x_pred, = plt.plot(t, mu)
    x_real, = plt.plot(t, x_real)

    return x_pred, x_real

def plot_covs(states, covs, state_i, gt):
    sigma = np.sqrt(covs[:, state_i, state_i])
    minus_sigma = states[:, state_i] - sigma
    plus_sigma = states[:, state_i] + sigma
    plt.figure(figsize=(14,8))
    t = np.array(range(len(states)))
    handles = ciplot(t, states[:, state_i], minus_sigma, plus_sigma, gt[:, state_i])
    if state_i == 0:
        angle_name = "roll"
    elif state_i == 1:
        angle_name = "pitch"
    elif state_i == 2:
        angle_name = "yaw"

    plt.title('IEKF estimations of ' + angle_name)
    plt.legend(handles, ['Estimated angle', 'Ground Truth'])
    plt.xlabel('Timestamp')
    plt.ylabel('Angle, rad')
    plt.show()