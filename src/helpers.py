import numpy as np

def standardize(gamma, upper_val = 0.3, lower_val = 0.1):

    s = (gamma-np.min(gamma))/(np.max(gamma)-np.min(gamma))
    out = s * (upper_val - lower_val) + lower_val

    return out

def is_pos_def(x):
            return np.all(np.linalg.eigvals(x) > 0)


def adjusting_assignment_level(level):
    
    adjustment_dict = {'low' : -0.545, 'medium' : 0, 'high' : 0.545}
    
    return adjustment_dict[level]

def revert_string_prob(string):
    
    reverse_dict = {'low': 0.35, 'medium': 0.5, 'high': 0.65} 
    
    return reverse_dict[string]