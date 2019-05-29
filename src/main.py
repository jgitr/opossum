
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
    u = UserInterface(10000,10, seed=5)
    u.generate_treatment(random_assignment=True,constant_pos=False, heterogeneous_pos=False, heterogeneous_neg=True, treatment_option_weights = None)
    y, X, assignment, treatment = u.output_data()





#import matplotlib.pyplot as plt
#
#fig = plt.figure()
#
#plt.hist(u.s.theta_option2, bins = 100)
#
#
#fig = plt.figure()
#
#plt.hist(u.s.gamma, bins=100)




