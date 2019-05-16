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
    u = UserInterface(500,10)
    u.generate_treatment()
    y, X, assignment, treatment = u.output_data()
    u.plot_covariates_correlation()    

    u.plot_distribution(y, treatment)

#    s = SimData(5000,10)
#    s.generate_covariates()
#    s.generate_treatment_assignment(False) # returns treatment assigment vector [0,1,...]
#    s.generate_treatment_effect()
#    correlation_heatmap = s.visualize_correlation()
#
#    # Output Triple
#    out = s.generate_outcome_variable()
    print('y: ', y)
    print('X:', X)
    print('treatment: ', treatment)




