from SimulateData import SimData
import numpy as np # necessary if predefined_idx is used in s.generate_treatment_effect


# Todo: Make this user friendly!
"""
User Input: 
N, k, options / effect type, assignment type (random...)

Output:
Y, X, Treatment Effect = theta_0 * D
Correlation Matrix
if specified use get function for g_0(x), D

"""
# Todo: Create helper class, inherit all variables from global one, reduce amount of functions for the user
# Todo: Test Model, try Powers Paper


if __name__ == "__main__":
    s = SimData(5000,10)
    s.generate_covariates()
    s.generate_treatment_assignment(False) # returns treatment assigment vector [0,1,...]
    s.generate_treatment_effect()
    correlation_heatmap = s.visualize_correlation()

    # Output Triple
    out = s.generate_outcome_variable()


