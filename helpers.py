import numpy as np

def standardize(gamma, upper_val = 0.3, lower_val = 0.1):

    s = (gamma-np.min(gamma))/(np.max(gamma)-np.min(gamma))
    out = s * (upper_val - lower_val) + lower_val

    return out