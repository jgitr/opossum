### double machine learning plot
from opossum import UserInterface
import seaborn as sns
import matplotlib.pyplot as plt
from dml import dml_single_run
import numpy as np
import time

start_time = time.time()
mc_iterations = 500
# initilizing empty arrays
avg_treatment_effects = np.zeros(mc_iterations)
theta_estimations = np.zeros((mc_iterations,3))

# iterating to simulate and estimate mc_iterations times
for i in range(mc_iterations):
    if i%10 ==0:
        print(round(i/mc_iterations,2 ))
    # generate data
    u = UserInterface(1000,50)
    u.generate_treatment(random_assignment=True, constant_pos=True, 
                         heterogeneous_pos=False)
    Y, X, assignment, treatment = u.output_data(binary=False, 
                                                x_y_relation='partial_nonlinear_simple')
    # save true treatment effects
    avg_treatment_effects[i] = np.mean(treatment[assignment==1])
    # save estimations 
    theta_estimations[i,:] = dml_single_run(Y,X,assignment)
# extract estimations of each method
theta_ols = theta_estimations[:,0]
theta_naive_dml= theta_estimations[:,1]
theta_cross_dml = theta_estimations[:,2]
 
duration = time.time() - start_time
print('Duration: ' + str(round(duration,2)) + ' sec')

mc_iterations = 500
# initilizing empty arrays
avg_treatment_effects_notrandom = np.zeros(mc_iterations)
theta_estimations = np.zeros((mc_iterations,3))

# iterating to simulate and estimate mc_iterations times
for i in range(mc_iterations):
    if i%10 ==0:
        print(round(i/mc_iterations,2 ))
    # generate data
    u = UserInterface(1000,50)
    u.generate_treatment(random_assignment=False, constant_pos=True, 
                         heterogeneous_pos=False)
    Y, X, assignment, treatment = u.output_data(binary=False, 
                                                x_y_relation='partial_nonlinear_simple')
    # save true treatment effects
    avg_treatment_effects_notrandom[i] = np.mean(treatment[assignment==1])
    # save estimations 
    theta_estimations[i,:] = dml_single_run(Y,X,assignment)
# extract estimations of each method
theta_ols_notrandom = theta_estimations[:,0]
theta_naive_dml_notrandom = theta_estimations[:,1]
theta_cross_dml_notrandom = theta_estimations[:,2]


# create plot
fig, axes = plt.subplots(1,2,figsize=(12,4))

axes[0].set_title('Random treatment assignment')
axes[1].set_title('Non-random treatment assignment')

axes[0].set_xlabel('average treatment effect estimates')
axes[1].set_xlabel('average treatment effect estimates')

axes[0].set_ylabel('Density')
axes[1].set_ylabel('Density')

axes[0].set_xlim((0.5,3))
axes[1].set_xlim((0.5,3))

axes[0].set_ylim((0,4))
axes[1].set_ylim((0,4))

sns.distplot(theta_ols, ax=axes[0], bins = mc_iterations, hist=False, rug=True, label='OLS')
sns.distplot(theta_naive_dml, ax=axes[0], bins = mc_iterations, hist=False, rug=True, label='DML naive')
sns.distplot(theta_cross_dml, ax=axes[0], bins = mc_iterations, hist=False, rug=True, label='DML cross')

sns.distplot(theta_ols_notrandom, ax=axes[1], bins = mc_iterations, hist=False, rug=True, label='OLS')
sns.distplot(theta_naive_dml_notrandom, ax=axes[1], bins = mc_iterations, hist=False, rug=True, label='DML naive')
sns.distplot(theta_cross_dml_notrandom, ax=axes[1], bins = mc_iterations, hist=False, rug=True, label='DML cross')

axes[0].axvline(np.mean(avg_treatment_effects), color='r', label='real treat')
axes[1].axvline(np.mean(avg_treatment_effects_notrandom), color='r', label='real treat')

axes[0].legend()
axes[1].legend()

plt.savefig('dml_estimator_distribution.png')


