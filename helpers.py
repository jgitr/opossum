import numpy as np

def sequence(upper_bound):
    res = []
    diff = 1
    x = 1
    while x <= upper_bound:
        res.append(x)
        x += diff
        #diff = 3 if diff == 1 else 1
    return res  #', '.join(res)

def standardize(gamma, upper_val = 0.3, lower_val = 0.1):

    s = (gamma-np.min(gamma))/(np.max(gamma)-np.min(gamma))
    out = s * (upper_val - lower_val) + lower_val

    return out