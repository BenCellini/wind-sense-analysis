
import numpy as np
from numpy.linalg import inv

class circKF:
    def __init__(self, A, B, C, D, Q, R):
        # Set the kalman filter properties. All inputs must be numpy arrays.
        # Assume the following structure:
        #   n = number of states
        #   r = number of inputs
        #   m = number of measurements (outputs)
        #
        #   x = state vector [n x 1]
        #   u = input vector [r x 1]
        #   y = measurement (output) vector [m x 1]
        #

        self.A = A  # process matrix [n x n]
        self.B = B  # input matrix [n x r]
        self.C = C  # output matrix [m x n]
        self.D = D  # feedforward matrix [m x r]
        self.Q = Q  # process noise [n x n] or scalar
        self.R = R  # measurement noise [m x 1] or scalar

        self.m, self.n = np.shape(self.C)  # number of outputs & states
        _, self.r = np.shape(self.B)  # number of inputs

        self.P = []  # state covariance
        self.S = []  # S
        self.K = np.zeros((self.n, 1))  # kalman gain

        self.x = np.zeros((self.n, 1))  # states
        self.u = np.zeros((self.r, 1))  # inputs
        self.y = np.zeros((self.m, 1))  # measurements

    def calKalmanGain(self, P, Q=None, R=None):
        # calKalmanGain: calculate the Kalman gain K & new covariance of the state P
        #   P: previous state covariance
        #   Q: process noise (optional)
        #   R: measurement noise (optional)
        #

        # Set Q & R matrices if not given
        if Q is not None:
            self.Q = Q

        if R is not None:
            self.R = R

        # State covariance
        self.P = self.A.dot(P).dot(np.transpose(self.A)) + self.Q

        # S
        self.S = self.C.dot(self.P).dot(np.transpose(self.C)) + self.R

        # Kalman gain
        self.K = self.P.dot(np.transpose(self.C)).dot(inv(self.S))

        # State covariance for next iteration
        self.P = self.P - self.K.dot(self.C).dot(np.transpose(self.P))

    def estimateState(self, xk_1, yk, uk, circ=True):
        # calKalmanGain: calculate the Kalman gain K & new covariance of the state P
        #   xk_1: previous state
        #   yk: current measurement
        #   uk: current input
        #   circ: (boolean) if true then use circular filter for first state
        #

        # Wrap measurements
        if circ:
            if np.abs(xk_1[0, 0] - yk[0, 0]) > np.pi:
                yk[0, 0] = yk[0, 0] + 2*np.pi*np.sign(xk_1[0, 0] - yk[0, 0])

        self.y = yk
        self.u = uk

        # Estimate state
        pred_error = self.y - self.C.dot(self.A).dot(xk_1)  # measurement - prediction from model
        self.x = self.A.dot(xk_1) + self.B.dot(self.u) + self.K.dot(pred_error)  # new estimated state

        # Wrap state
        if circ:
            self.x[0, 0] = self.x[0, 0] % (2*np.pi)

    def runFilter(self, y_record, u_record, P0, x0=None, circ=True):
        # runFilter: calculate the Kalman gain K & new covariance of the state P
        #   y_record: set of prerecorded measurements
        #   u_record: set of prerecorded inputs
        #   P0: initial noise covariance
        #   x0: initial state estimate
        #   circ: (boolean) if true then use circular filter for first state
        #

        n_point = y_record.size  # number of data points

        # Allow consistent 2D indexing
        y_record = np.atleast_2d(y_record)
        u_record = np.atleast_2d(u_record)

        # Preallocate storage for state estimates (X), kalman gains (K), & noise covariance (P)
        X = np.empty((n_point, self.n))
        K = np.empty((n_point, self.n))
        P = np.empty((n_point, self.n))

        X[:] = np.NaN
        K[:] = np.NaN

        # Initial state estimate
        if x0 is None:
            xf = np.array(y_record[:, 0])  # 1st state
            self.x = np.vstack((xf, np.zeros((self.n-1, 1))))  # rest of states are 0
        else:
            self.x = x0

        X[0, :] = x0  # store initial state estimate

        # Initial state covariance estimate
        self.P = P0
        # P[0, :] = self.P  # store state covariance

        # Loop through data & filter
        for k in range(1, n_point):
            # Get current measurements
            self.y = np.atleast_2d(y_record[:, k])

            # Get current input
            self.u =  np.atleast_2d(u_record[:, k])

            # Kalman iteration
            self.calKalmanGain(self.P)
            self.estimateState(self.x, self.y, self.u, circ)

            # Store state estimates & kalman gain
            X[k, :] = np.transpose(self.x)
            K[k, :] = np.transpose(self.K)
            # P[k, :] = self.P

        return X, K


def circKF_pos_vel(dt, sigma_a=1, sigma_y=1, sigma_p=1):
    A = np.array([[1, dt], [0, 1]])
    B = np.array([[(1 / 2) * (dt ** 2)], [dt]])
    C = np.transpose(np.array([[1], [0]]))
    D = np.array([0])
    m, n = np.shape(C)

    Q = sigma_a * np.eye(n, n)
    R = sigma_y ** 2
    P0 = (sigma_p ** 2) * np.eye(n, n)

    kf = circKF(A, B, C, D, Q, R)

    return kf, P0
