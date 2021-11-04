from __future__ import division
import math
import matplotlib.pyplot as plt
from numpy.random import randn
import numpy as np
from numpy.testing._private.utils import measure
from traj_kf import KF
np.random.seed(23)

MEASURMENT_SD = 0.01    # 1 cm  
PROCESS_SD = 0.20       # 20 cm 

MEASURMENT_VAR = MEASURMENT_SD**2 
PROCESS_VAR =  PROCESS_SD**2

#  ellipsis in python
PROCESS_BIAS = np.random.normal(0, PROCESS_SD, 1149)
MEASURMENT_BIAS = np.random.normal(0, MEASURMENT_SD, 1149) 

def add_noise(v_x, dt, pos, selector=1):
    if selector == 0:
        return (v_x) * dt + PROCESS_BIAS[pos]
    else:
        return (v_x) * dt + MEASURMENT_BIAS[pos]

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

def compute_trajectory(v_0_mph, theta, v_wind_mph=0, alt_ft = 0.0, dt = 0.005):
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
    x_filter = [0.0]

    y = [altitude]
    
    i = 0
    while y[i] >= ground_level:
        
        v = math.sqrt((v_x - v_wind) **2+ v_y**2)
        
        x_measured.append(x_measured[i] + add_noise(v_x, dt, i+1))
        x_true.append(x_true[i] + v_x * dt)
        process_measurment = add_noise(v_x, dt, i+1, selector=0)
        pred, filtered_val = filter.filter(x_measured[i+1], MEASURMENT_VAR, process_measurment, PROCESS_VAR)
        x_pred.append(pred)
        x_filter.append(filtered_val)

        print([x_true[i+1], x_measured[i+1], x_pred[i+1], x_filter[i+1]])
        y.append(y[i] + v_y * dt)        
        
        v_x = v_x - a_drag(v, altitude) * v * (v_x - v_wind) * dt
        v_y = v_y - a_drag(v, altitude) * v * v_y * dt - g * dt
        i += 1
    

    print(*PROCESS_BIAS[:10], sep=', ')
    print(*MEASURMENT_BIAS[:10], sep=', ')
    # meters to yards
    np.multiply (x_measured, 1.09361)
    np.multiply (x_true, 1.09361)
    np.multiply (x_pred, 1.09361)
    np.multiply (x_filter, 1.09361)
    np.multiply (y, 1.09361)
    
    return x_true, x_measured, x_pred, x_filter, y


if __name__ == "__main__":
    x_true, x_measured, x_pred, x_filter, y = compute_trajectory(v_0_mph = 110., theta=35., v_wind_mph = 0., alt_ft=0.)

    # plt.plot(x_true, y, 'k')
    plt.plot(x_measured, y, 'y')
    plt.plot(x_pred, y, 'r')
    plt.plot(x_filter, y, 'b')

    plt.show()