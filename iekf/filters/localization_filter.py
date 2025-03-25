"""
An abstract base class to implement the various localization filters in the task: EKF or  PF.
"""

from abc import ABC
from abc import abstractmethod

import numpy as np

from tools.objects import Gaussian
import copy 

class LocalizationFilter(ABC):
    def __init__(self, initial_state, M, Q, g, b):
        """
        Initializes the filter parameters.

        :param initial_state: The Gaussian distribution representing the robot prior.
        :param M: A 3x3 np-array matrix of motion noise parameters (format: [rad/s]).
        :param Q: A 3x3 np-array matrix of the measurement noise covariance (format: rad).
        """

        assert isinstance(initial_state, Gaussian)
        assert initial_state.Sigma.shape == (3, 3)

        if not isinstance(initial_state, Gaussian):
            raise TypeError('The initial_state must be of type `Gaussian`. (see tools/objects.py)')

        if initial_state.mu.ndim < 1:
            raise ValueError('The initial mean must be a 1D numpy ndarray of size 3.')
        elif initial_state.mu.shape == (3, ):
            # This transforms the 1D initial state mean into a 2D vector of size 3x1.
            initial_state.mu = initial_state.mu[np.newaxis].T
        elif initial_state.mu.shape != (3, 1):
            raise ValueError('The initial state mean must be a vector of size 3x1')

        self.state_dim = 3   # [fi_x, fi_y, fi_z]
        self.motion_dim = 3  # [dfi_x, dfi_y, dfi_z]
        self.obs_dim = 3     # [a_x, a_y, a_z]

        self._state = copy.deepcopy(initial_state)
        self._state_bar = copy.deepcopy(initial_state)

        # Motion noise variance
        self.M = M
        # Measurement noise variance
        self.Q = Q
        # gravity vector, usually [0, 0, -9.81]
        self.g = g
        # Motion bias, to be substracted from each motion
        self.b = b

    @abstractmethod
    def predict(self, u):
        """
        Updates mu_bar and Sigma_bar after taking a single prediction step after incorporating the control.

        :param u: The control for prediction (format: [omega_x, omega_y, omega_z]).
        """
        raise NotImplementedError('Must implement a prediction step for the filter.')

    @abstractmethod
    def update(self, z):
        """
        Updates mu and Sigma after incorporating the observation z.

        :param z: Observation measurement (format: [a_x, a_y, a_z]).
        """
        raise NotImplementedError('Must implement an update step for the filter.')

    @property
    def mu_bar(self):
        """
        :return: The state mean after the prediction step (format: 1D array for easy indexing).
        """
        return self._state_bar.mu.T[0]

    @property
    def Sigma_bar(self):
        """
        :return: The state covariance after the prediction step (shape: 3x3).
        """
        return self._state_bar.Sigma

    @property
    def mu(self):
        """
        :return: The state mean after the update step (format: 1D array for easy indexing).
        """
        return self._state.mu.T[0]

    @property
    def Sigma(self):
        """
        :return: The state covariance after the update step (shape: 3x3).
        """
        return self._state.Sigma
