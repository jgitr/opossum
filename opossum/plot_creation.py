import os
os.chdir(r'/home/tobias//github_repositories/predictiveanalytics/src')
from opossum import UserInterface
from plot_functions import propensity_score_plt, all_treatment_effect_plt, \
                            single_treatment_effect_plt, output_difference_plt, \
                            avg_treatment_effect_plt, scatter_plot_y_x, \
                            scatter_plot_y_x_treatment_difference, \
                            pos_neg_heterogeneous_effect, scatter_transformations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

u = UserInterface(10000,20, seed=5)
u.generate_treatment(random_assignment=True, treatment_option_weights = [0, 0, 0.7, 0.3, 0, 0], intensity=5)
y, X, assignment, treatment = u.output_data()

pos_neg_heterogeneous_effect(treatment,assignment)

plt.savefig('pos_neg_heterogeneous_treatment_effect.png')



import os

os.getcwd()

#### New heterogeneous effect
u = UserInterface(100000,10, seed=5)
u.generate_treatment(random_assignment=True, treatment_option_weights = [0, 0, 0, 0, 0, 1], intensity=10)
y, X, assignment, treatment = u.output_data()


single_treatment_effect_plt(treatment,assignment,'Just Positive heterogeneous')

u = UserInterface(100000,10, seed=5)
u.generate_treatment(random_assignment=True, treatment_option_weights = [0, 0, 0.5, 0.5, 0, 0], intensity=10)
y, X, assignment, treatment = u.output_data()


single_treatment_effect_plt(treatment,assignment,'50:50 pos/neg heterogeneous')

u = UserInterface(100000,10, seed=5)
u.generate_treatment(random_assignment=True, treatment_option_weights = [0.2, 0, 0.4, 0.3, 0.1, 0], intensity=10)
y, X, assignment, treatment = u.output_data()

single_treatment_effect_plt(treatment,assignment,'mix')

##### scatter plot y~X with 3 different transformations


u = UserInterface(10000,30, seed=71, categorical_covariates = None)
u.generate_treatment(random_assignment=True, 
                     treatment_option_weights = [0, 0, 0, 0, 1, 0])

y_linear, X, assignment, treatment = u.output_data(x_y_relation = 'linear_simple')
y_partial_nonlinear, X, assignment, treatment = u.output_data(x_y_relation = 'partial_nonlinear_simple')
y_nonlinear, X, assignment, treatment = u.output_data(x_y_relation = 'nonlinear_simple')

y_list = [y_linear, y_partial_nonlinear, y_nonlinear]

scatter_transformations(y_list, X, u.get_weigths_covariates_to_outputs())

#plt.savefig('y_transformations_plot.png')

##### scatter plot y~X without treatment
u = UserInterface(10000,50, seed=8, categorical_covariates = None)
u.generate_treatment(random_assignment=True, 
                     treatment_option_weights = [0, 0, 0, 0, 1, 0])
y, X, assignment, treatment = u.output_data(x_y_relation = 'partial_nonlinear_simple')

scatter_plot_y_x(np.dot(X, u.get_weigths_covariates_to_outputs()),y)

##### scatter plot y~X with treatment
u = UserInterface(10000,100, seed=8, categorical_covariates = None)
u.generate_treatment(random_assignment=True, 
                     treatment_option_weights = [0, 0, 1, 0, 0, 0],
                     intensity = 10)
y, X, assignment, treatment = u.output_data(x_y_relation = 'nonlinear_simple')

scatter_plot_y_x_treatment_difference(np.dot(X, u.get_weigths_covariates_to_outputs()),
                                      y, assignment)

scatter_plot_y_x(X[:,7],y)
   
u.get_weigths_covariates_to_outputs() 

u.plot_covariates_correlation()

##### propensity score plot
u = UserInterface(10000,10, seed=5)
u.generate_treatment(random_assignment=False,non_linear_assignment=True, 
                     assignment_prob = 'medium', treatment_option_weights = [1, 0, 0, 0, 0, 0])
y, X, assignment, treatment = u.output_data()

prop_score_conditioned = u.get_propensity_scores()

print('Average of propensity scores:' + str(np.mean(u.get_propensity_scores())))

u = UserInterface(10000,10, seed=5)
u.generate_treatment(random_assignment=True, assignment_prob = 'high',  treatment_option_weights = [1, 0, 0, 0, 0, 0])
y, X, assignment, treatment = u.output_data()

prop_score_random = u.get_propensity_scores()
propensity_score_plt(prop_score_predictions_rf,prop_score_random)




###### treatment effects plots
#
### Each option alone 
#treatment_list = []
#assignment_list = []
#
#for i in range(4):
#    treatment_option_weights = np.zeros(4)
#    treatment_option_weights[i] = 1
#    
#    u = UserInterface(10000,10, seed=123)
#    u.generate_treatment(random_assignment=True, treatment_option_weights = treatment_option_weights)
#    y, X, assignment, treatment = u.output_data(binary=False)
#    
#    treatment_list.append(treatment)
#    assignment_list.append(assignment)
#
#all_treatment_effect_plt(treatment_list, assignment_list)
#
#### All options at once 
#
## equally distributed
#u = UserInterface(10000,10, seed=23)
#u.generate_treatment(constant=True, heterogeneous=True, negative=True, no_treatment=True)
#y, X, assignment, treatment = u.output_data(binary=False)
#    
#single_treatment_effect_plt(treatment, assignment, title = 'All 4 options at once equally distributed')
#
## more realistic case
#
#u = UserInterface(10000,10, seed=23)
#u.generate_treatment(treatment_option_weights = [0, 0, 0.7, 0.1, 0.2])
#y, X, assignment, treatment = u.output_data(binary=False)
#    
#single_treatment_effect_plt(treatment, assignment, 
#                            title = 'More realistic case with [0, 0, 0.7, 0.1, 0.2] distribution')
#
#
#### Output differences treated/not_treated plots

### continous 
u = UserInterface(10000,10, seed=7, categorical_covariates = None)
u.generate_treatment(random_assignment=True, treatment_option_weights = [0, 0, 1, 0, 0, 0], 
                     intensity = 5)
y, X, assignment, treatment = u.output_data(False, x_y_relation = 'nonlinear_simple')


y_treated = y[assignment==1]
y_not_treated = y[assignment==0]

output_difference_plt(y_not_treated, y_treated)

### binary
u = UserInterface(10000,10, seed=15)
u.generate_treatment(random_assignment=True, treatment_option_weights = [0, 0, 1, 0, 0, 0], 
                     intensity=5)
y, X, assignment, treatment = u.output_data(True)

y_treated = y[assignment==1]
y_not_treated = y[assignment==0]

output_difference_plt(y_not_treated, y_treated, binary = True)



##### Inverse probability weighting
u = UserInterface(10000,20, seed=12)
u.generate_treatment(random_assignment=False, treatment_option_weights = [1, 0, 0, 0, 0, 0])
y, X, assignment, treatment = u.output_data(False)


# setting up logistic regression    
lr = LogisticRegression()
# fit model to data
lr.fit(X, assignment)
# predict propensity scores D ~ X
prop_score_predictions = lr.predict_proba(X)[:,1]

rf = RandomForestClassifier(100)

treatment_transformed = treatment[assignment==1].copy()

treatment_transformed[treatment_transformed==0.5] = 0
treatment_transformed[treatment_transformed==1] = 1

rf.fit(X[assignment==1],treatment_transformed)
treatment_predictions = rf.predict(X[assignment==1])

np.sum(treatment_transformed==treatment_predictions)/len(treatment_predictions)

index = np.random.choice(np.arange(len(treatment_transformed)),size=len(treatment_transformed),
                         replace=False )

index[:len(treatment_transformed)/2]


#    propensity_score_plt(prop_score_predictions, u.s.propensity_score )

def inverse_probability_weighting(prop_score_predictions, assignment, y):
    '''
    
    ATE = mu_1 - mu_0
    ATE = E[ (D_i*Y_i) / e(x_i) ]   -   E[ ((1-D_i)*Y_i) / (1-e(x_i)) ] 
    
    '''
    mu_1 = np.sum(y[assignment == 1] / prop_score_predictions[assignment == 1])/len(y)

    mu_0 = np.sum(y[assignment == 0] / (1-prop_score_predictions[assignment == 0]))/len(y)

    tau = mu_1 - mu_0
    
    return tau

ipw_ate_estimate = inverse_probability_weighting(prop_score_predictions, assignment, y)

simple_ate_estimate = np.mean(y[assignment==1]) - np.mean(y[assignment==0])

true_ate = np.mean(treatment[assignment==1])

