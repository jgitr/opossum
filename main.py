#from SimulateData import SimData
#import numpy as np # necessary if predefined_idx is used in s.generate_treatment_effect
from SimulateData import UserInterface
from plots import *

# Todo: Make this user friendly!
"""
User Input: 
N, k, options / effect type, assignment type (random...)

Output:
Y, X, Treatment Effect = theta_0 * D
Correlation Matrix
if specified use get function for g_0(x), D

"""


if __name__ == "__main__":
    u = UserInterface(100,10, seed=19)
    u.generate_treatment(random_assignment=False)
    y, X, assignment, treatment = u.output_data()
#    u.plot_covariates_correlation()    
#
#    u.plot_distribution(y, treatment)

    histogram(u.s.propensity_score)
 



