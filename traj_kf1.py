from matplotlib.pyplot import xcorr
import numpy as np
from test import gaussian

class KF:
   
    def __init__(self):
        self.x = 0

    def update(self, prior, measurement):
        x, P = prior        # mean and variance of prior
        z, R = measurement  # mean and variance of measurement
        
        y = z - x        # residual
        K = P / (P + R)  # Kalman gain

        x = x + K*y      # posterior
        P = (1 - K) * P  # posterior variance
        return gaussian(x, P)

    def predict(self, posterior, movement):
        x, P = posterior # mean and variance of posterior
        dx, Q = movement # mean and variance of movement
        x = x + dx
        P = P + Q
        return gaussian(x, P)

    def filter(self, measured_x_pose, state_vector):
        v, theta = state_vector
        


        return self.x 
