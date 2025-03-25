import numpy as np
from tools import compare, data_processing, utils
import pandas as pd


class SLAMData(object):
    """
    Abstract class for loading SLAM data, particularly IMU (inertial measurement unit) measurements.
    Stores accelerometer and gyroscope data in separate numpy arrays in shape (N, 3)
    """
    def __init__(self):
        super().__init__()
        self.csv_imu_path = ''
        self.csv_mocap_path = ''

        self.accl_data = [] # numpy array
        self.gyro_data = [] # numpy array
        self.timestamps = [] # numpy array

        self.quat_timestamps = [] # numpy array
        self.quat_T = [] # numpy array
        self.quat_R = [] # numpy array
    
    def import_imu_data(self, smoothing=False, downsample=False, downscale=1):
        '''
        Imports and smooths (savgol, cubic spline) data
        Importing t, 
        ang_vel, acc,
        data from 1, 2, 3, 4, ... columns of the .csv file 

        return: timestamps - array of timestamps
                gyro_data - array of x, y, z angular velocities for each timestamp 
                accl_data - array of x, y, z accelerations for each timestamp 
        '''
        data_np = pd.read_csv(self.csv_imu_path, header=1).to_numpy()
        self.timestamps = data_np[:, 0] / 1e9
        self.gyro_data = np.empty((len(self.timestamps), 3))
        self.accl_data = np.empty((len(self.timestamps), 3))
        self.gyro_data = data_np[:, 1:4]
        self.accl_data = data_np[:, 4:7]

        if smoothing:
            self.gyro_data = data_processing.smooth_and_resample_imu(self.timestamps, self.timestamps, self.gyro_data)
            self.accl_data = data_processing.smooth_and_resample_imu(self.timestamps, self.timestamps, self.accl_data)

        if downsample:
            self.timestamps, self.gyro_data, self.accl_data = data_processing.downsample(downscale, self.timestamps, self.gyro_data, self.accl_data)
            self.freq_field /= downscale

        return self.timestamps, self.gyro_data, self.accl_data
    
    def import_mocap_data(self, smoothing=False, downsample=False, downscale=1):
        '''
        Importing and smoothing (savgol, SLERP) mocap data from .csv file with order: 
        time, translation, rotation quaternion [w, x, y, z]

        return: quat_timestamps, quat_R [w, x, y, z], quat_T
        '''
        data_quat_np = pd.read_csv(self.csv_mocap_path, header=1).dropna().to_numpy()
        self.quat_timestamps = data_quat_np[:, 0].astype(np.float64) / 1e9
        self.quat_R = np.empty((len(self.quat_timestamps), 4))
        self.quat_T = np.empty((len(self.quat_timestamps), 3))
        self.quat_T[:, 0] = data_quat_np[:, 1]
        self.quat_T[:, 1] = data_quat_np[:, 2]
        self.quat_T[:, 2] = data_quat_np[:, 3]
        self.quat_R[:, 0] = data_quat_np[:, 4]
        self.quat_R[:, 1] = data_quat_np[:, 5]
        self.quat_R[:, 2] = data_quat_np[:, 6]
        self.quat_R[:, 3] = data_quat_np[:, 7]

        if smoothing:
            self.quat_R = data_processing.smooth_and_resample_quats(self.quat_timestamps, self.quat_timestamps, self.quat_R)

        if downsample:
            self.quat_timestamps, self.quat_R, self.quat_T = data_processing.downsample(downscale, self.quat_timestamps, self.quat_R, self.quat_T)

        return self.quat_timestamps, self.quat_R, self.quat_T
    

class TUMData(SLAMData):
    """
    Loads TUM (Technical University of Munich) data given the path to the csv file.
    https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset
    """
    def __init__(self, imu_path, mocap_path):
        super().__init__()
        self.csv_imu_path = imu_path
        self.csv_mocap_path = mocap_path

        self.freq_imu_field = 200 # Hz, from TUM website
        self.freq_mocap_field = 120 # Hz, from TUM website

        # self.gyro_data = np.stack((dict['ang_vel_uncal_x[rad/s]'], dict['ang_vel_uncal_y[rad/s]'], dict['ang_vel_uncal_z[rad/s]']), axis=1)
        # self.accl_data = np.stack((dict['acc_uncal_x[m/s^2]'], dict['acc_uncal_y[m/s^2]'], dict['acc_uncal_z[m/s^2]']), axis=1)
        # self.timestamps = np.array(dict['timestamp_global[ms]'])
        # assert self.gyro_data.shape[0] == self.accl_data.shape[0]

    # def load_imu_data(self):
    #     """
    #     Loads IMU data from the csv file and stores in dictionary with specific keys defined in abstract class.
    #     """
    #     assert self.csv_imu_path.split('.')[1] == 'csv'
    #     with open(self.csv_path) as f:
    #         reader = csv.DictReader(f)

    #         # create a dict with keys from base class 
    #         dict = {key: [] for key in self.keys}

    #         # fill dict
    #         for row in reader:
    #             for i in range(len(self.keys)):
    #                 # extract raw keys and traget keys from base class
    #                 key_raw, key_target = list(row.keys())[i], self.keys[i]
    #                 if key_raw == "#timestamp [ns]":
    #                     dict[key_target].append(float(row[key_raw]) * 1e-9) # convert to ms
    #                 else:
    #                     dict[key_target].append(float(row[key_raw]))

    #     return dict   

# class CustomData(SLAMData):
    