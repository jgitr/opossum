#from SimulateData import SimData
#import numpy as np # necessary if predefined_idx is used in s.generate_treatment_effect
from SimulateData import UserInterface


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
    u = UserInterface(10000,10)
    u.generate_treatment(random_assignment=True, constant=False, heterogeneous=True)
    y, X, assignment, treatment = u.output_data()
#    u.plot_covariates_correlation()    










