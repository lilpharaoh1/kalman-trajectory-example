from collections import namedtuple

# gaussian = namedtuple('gaussian', ['pos_x', 'pos_y'])

gaussian = namedtuple('Gaussian', ['mean', 'var'])

x = gaussian(0.0, 20.0**2)
process_model = gaussian(1.0, 1**2)

# velocity * dt  velocity = 1 and dt = 1 

# def predict(pos, movement):
#     return gaussian(pos.mean+movement.mean, pos.var+movement.var)

# # def update(prior, likelihood):
# #     pass


def update(prior, measurement):
    x, P = prior        # mean and variance of prior
    z, R = measurement  # mean and variance of measurement
    
    y = z - x        # residual
    K = P / (P + R)  # Kalman gain

    x = x + K*y      # posterior
    P = (1 - K) * P  # posterior variance
    return gaussian(x, P)

def predict(posterior, movement):
    x, P = posterior # mean and variance of posterior
    dx, Q = movement # mean and variance of movement
    x = x + dx
    P = P + Q
    return gaussian(x, P)


