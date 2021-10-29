from __future__ import division
import math
import matplotlib.pyplot as plt
from numpy.random import randn
import numpy as np
from numpy.testing._private.utils import measure
from traj_kf1 import KF
np.random.seed(23)

def a_drag (vel, altitude):
    """ returns the drag coefficient of a baseball at a given velocity (m/s)
    and altitude (m).
    """
    
    v_d = 35
    delta = 5
    y_0 = 1.0e4
    
    val = 0.0039 + 0.0058 / (1 + math.exp((vel - v_d)/delta))
    val *= math.exp(-altitude / y_0)
    
    return val

def noise_func(input, min=0.0, max=1.0):
    sd = 0.01*input**3
    if sd > max: 
        sd = max
    elif sd < min: 
        sd = min

    noise = np.random.normal(0, sd)
    return noise

def find_theta(dx, dy):
    """
    Feed in difference in x and difference in y, float
    """
    if dy != 0:
        theta = np.arctan(abs(dy) / dx)
        if dy < 0:
            theta = -theta
    else: 
        theta = 0.0
    
    return np.degrees(theta)


def compute_trajectory(v_0_mph, theta, v_wind_mph=0, alt_ft = 0.0, dt = 0.01):
    filter = KF()
    g = 9.8
    
    ### comput
    theta = math.radians(theta)
    # initial velocity in direction of travel
    v_0 = v_0_mph * 0.447 # mph to m/s
    
    # velocity components in x and y
    v_x = v_0 * math.cos(theta)
    v_y = v_0 * math.sin(theta)
   
    v_wind = v_wind_mph * 0.447 # mph to m/s
    altitude = alt_ft / 3.28   # to m/s
    
    ground_level = altitude
    
    x_pred = [0.0]
    x_measured = [0.0]
    x_true = [0.0]
    y = [altitude]

    v_arr = [v_0]
    angles = [np.degrees(theta)]
    
    i = 0
    while y[i] >= ground_level:
        
        v = math.sqrt((v_x - v_wind) **2+ v_y**2)
        v_arr.append(v)

        state_vector = [v+noise_func(v, max=v//15), angles[i]+noise_func(abs(theta))]
        
        x_measured.append(x_measured[i] + (v_x + v_x*noise_func(v_x)) * dt)
        x_true.append(x_true[i] + v_x * dt)
        #x_pred.append(filter.filter(x_measured[i+1], state_vector))
        y.append(y[i] + v_y * dt)     
        
        v_x = v_x - a_drag(v, altitude) * v * (v_x - v_wind) * dt
        v_y = v_y - a_drag(v, altitude) * v * v_y * dt - g * dt

        angles.append(find_theta(x_true[i+1] - x_true[i], y[i+1] - y[i]))

        i += 1
        
    # meters to yards
    np.multiply (x_measured, 1.09361)
    np.multiply (x_true, 1.09361)
    np.multiply (x_pred, 1.09361)
    np.multiply (y, 1.09361)
    
    return x_true, x_measured, np.ones_like(x_true), y, v_arr, angles


if __name__ == "__main__":
    x_true, x_measured, x_pred, y, v, angles = compute_trajectory(v_0_mph = 110., theta=35., v_wind_mph = 0., alt_ft=0.)

    plt.plot(x_true, y)
    plt.plot(x_measured, y)
    plt.plot(x_pred, y)
    plt.show()

    plt.plot(np.arange(0, len(v), 1), v)
    plt.show()

    plt.plot(np.arange(0, len(angles), 1), np.zeros_like(angles))
    plt.plot(np.arange(0, len(angles), 1), angles)
    plt.show()