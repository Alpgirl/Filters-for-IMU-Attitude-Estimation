from abc import ABC, abstractmethod
import csv
import numpy as np


class SLAMData(object):
    def __init__(self):
        super().__init__()
        self.keys = ['timestamp_global[ms]', 'ang_vel_uncal_x[rad/s]', 'ang_vel_uncal_y[rad/s]', 'ang_vel_uncal_z[rad/s]', \
                     'acc_uncal_x[m/s^2]', 'acc_uncal_y[m/s^2]', 'acc_uncal_z[m/s^2]']
        self.accl_data = [] # numpy array
        self.gyro_data = [] # numpy array

    
    @abstractmethod
    def load_data(self):
        pass

    def get_size(self):
        return self.accl_data.shape[0]

    def get_gyro_dataIdx(self, idx):
        return self.gyro_data[idx]
    
    def get_accl_dataIdx(self, idx):
        return self.accl_data[idx]
    

class TUMData(SLAMData):
    def __init__(self, path):
        super().__init__()
        self.csv_path = path
        dict = self.load_data()
        self.gyro_data = np.stack((dict['ang_vel_uncal_x[rad/s]'], dict['ang_vel_uncal_y[rad/s]'], dict['ang_vel_uncal_z[rad/s]']), axis=1)
        self.accl_data = np.stack((dict['acc_uncal_x[m/s^2]'], dict['acc_uncal_y[m/s^2]'], dict['acc_uncal_z[m/s^2]']), axis=1)
        assert self.gyro_data.shape[0] == self.accl_data.shape[0]

    def load_data(self):
        assert self.csv_path.split('.')[1] == 'csv'
        with open(self.csv_path) as f:
            reader = csv.DictReader(f)

            # create a dict with keys from base class 
            dict = {key: [] for key in self.keys}

            # fill dict
            for row in reader:
                for i in range(len(self.keys)):
                    # extract raw keys and traget keys from base class
                    key_raw, key_target = list(row.keys())[i], self.keys[i]
                    if key_raw == "#timestamp [ns]":
                        dict[key_target].append(str(row[key_raw]))
                    else:
                        dict[key_target].append(float(row[key_raw]))
                
        return dict

# class CustomData(SLAMData):
    