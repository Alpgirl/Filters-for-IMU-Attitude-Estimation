"""
This file implements the Extended Kalman Filter.
"""

import numpy as np
import mrob
import ahrs

from filters.localization_filter import LocalizationFilter

def Exp(vec):
    '''
    Matrix exponent from Lie algebra: R3 -> SO3

    param: vec - [fi_x, fi_y, fi_z] vector

    return: mrob.SO3 object
    '''
    return mrob.geometry.SO3(vec)

class Madgwick():
    def __init__(self):
        super().__init__()

    def apply(self, data_gyr, data_acc, freq):
        madgwick_IMU = ahrs.filters.Madgwick(gyr=data_gyr,
                                 acc=data_acc,
                                 frequency=freq)
        return madgwick_IMU

class IEKF(LocalizationFilter):
    '''
    Left-Invariant EKF 
    '''
    def V_t(self, dt):
        '''
        Motion jacobian 

        We propagate motion as d_R = w*dt, so it's just d_t
        '''

        V = np.eye(self.motion_dim) * dt
        return V
    
    def H_t(self, a):
        '''
        Observation jacobian 

        We observe deviation of acceleration vector from gravity vector
        h(t) = R_t @ a - g

        So we build Jacobian out of Lie generators
        '''
        state = self.mu_bar
        assert isinstance(state, np.ndarray)
        assert isinstance(a, np.ndarray)

        assert state.shape == (self.state_dim,)
        assert a.shape == (self.obs_dim,)

        G1 = np.array([[0, 0, 0], 
                       [0, 0, -1], 
                       [0, 1, 0]])
        G2 = np.array([[0, 0, 1], 
                       [0, 0, 0], 
                       [-1, 0, 0]])
        G3 = np.array([[0, -1, 0], 
                       [1, 0, 0], 
                       [0, 0, 0]])
        
        R_bar = Exp(state).R()

        d_fi1 = G1 @ R_bar @ a.reshape(-1, 1) 
        d_fi2 = G2 @ R_bar @ a.reshape(-1, 1)  
        d_fi3 = G3 @ R_bar @ a.reshape(-1, 1)  
        
        H = np.hstack((d_fi1, d_fi2, d_fi3))
        return H

    def predict(self, u, dt):
        '''
        Propagate state with action u during dt

        if bias self.b is set, we substract this bias from u

        u_unbiased = u - bias
        '''
        d_R = Exp(u*dt)
        u_unbias = u - self.b
        d_R_unbias = Exp(u_unbias*dt)
        unbias_rot = Exp(d_R_unbias.R() @ self.b * dt).inv()
        R_bar = Exp(self.mu).mul(d_R)
        R_bar_unbias = R_bar.mul(unbias_rot)

        V = self.V_t(dt)
        M = self.M

        Sigma_bar = d_R.R() @ self.Sigma @ d_R.R().T + V @ M @ V.T
        self._state_bar.mu = R_bar_unbias.Ln().reshape(-1, 1)
        self._state_bar.Sigma = Sigma_bar

    def update(self, a):
        '''
        Update state with observation of acceleration vector a
        '''
        Q = self.Q
        H = self.H_t(a)
        K = self.Sigma_bar @ H.T @ np.linalg.inv(H @ self.Sigma_bar @ H.T + Q)

        R_bar = Exp(self.mu_bar)
        innocation_vec = R_bar.R() @ a - self.g
        R = Exp(K @ innocation_vec).mul(R_bar)

        self._state.mu = R.Ln().reshape(-1, 1)
        self._state.Sigma = (np.eye(self.state_dim) - K @ H) @ self.Sigma_bar

        return K, innocation_vec

    def update_fake(self):
        '''
        Update state without observation - simply keep propagating
        '''
        R_bar = Exp(self.mu_bar)
        self._state.mu = R_bar.Ln().reshape(-1, 1)
        self._state.Sigma = self.Sigma_bar
