from SimulateData import UserInterface


"""
User Input: 
N, k, options / effect type, assignment type (random...)

Output:
Y, X, Treatment Effect = theta_0 * D
Correlation Matrix
if specified use get function for g_0(x), D

"""


if __name__ == "__main__":
    
    ##### propensity score plot
    u = UserInterface(10000,10, seed=5, skewed_covariates = True)
    u.generate_treatment(random_assignment=False, treatment_option_weights = [1, 0, 0, 0])
    y, X, assignment, treatment = u.output_data()

    


