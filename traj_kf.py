from matplotlib.pyplot import xcorr
import numpy as np
from test import gaussian

class KF:
   
    def __init__(self):
        self.x = 0
        self.P = 500

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

    def filter(self, sensor_measurment, sensor_var, velocity, dt):
        dx = velocity*dt 
        prior = self.predict(gaussian(self.x, self.P), gaussian(dx, 0.0001))
        
        self.x, self.P = self.update(prior, gaussian(sensor_measurment, sensor_var))

        return self.x 
