########## Creating plots for README ##########

#import os
#os.chdir(r'/home/tobias/github_repositories/predictiveanalytics/src')

from SimulateData import UserInterface

from plot_functions import propensity_score_plt, all_treatment_effect_plt, single_treatment_effect_plt, output_difference_plt, avg_treatment_effect_plt 



##### propensity scores
u = UserInterface(10000,10, seed=5)
u.generate_treatment(random_assignment=False, treatment_option_weights = [1, 0, 0, 0, 0])
y, X, assignment, treatment = u.output_data()

prop_score_conditioned = u.get_propensity_scores()

u = UserInterface(10000,10, seed=5)
u.generate_treatment(random_assignment=True, assignment_prob = 0.5,  treatment_option_weights = [1, 0, 0, 0, 0])
y, X, assignment, treatment = u.output_data()

prop_score_random = u.get_propensity_scores()

propensity_score_plt(prop_score_conditioned, prop_score_random, file_name = 'propensity_scores.png', save = True)


##### treatment effects
# options
treatment_combinations = [[1,0,0,0,0], [0,1,0,0,0], [0,0,0,0,1], [0,0,1,0,0], [0,0,0,1,0], [0,0,0.5,0.5,0]]
treatment_list = []

for combination in treatment_combinations:
    
    u = UserInterface(10000,10, seed=15)
    u.generate_treatment(random_assignment=True, treatment_option_weights = combination)
    y, X, assignment, treatment = u.output_data(binary=False)
    
    treatment_list.append(treatment)
    

all_treatment_effect_plt(treatment_list, assignment, file_name = 'treatment_effect_options.png', save = True)

# realistic 
u = UserInterface(10000,10, seed=23)
u.generate_treatment(treatment_option_weights = [0, 0, 0.7, 0.1, 0.2])
y, X, assignment, treatment = u.output_data(binary=False)

single_treatment_effect_plt(treatment, assignment, 
                            title = 'Example with [0, 0, 0.7, 0.1, 0.2] treatment distribution', 
                            file_name = 'realistic_treatment_effect.png', save = True)


##### output distribution
# continuous
u = UserInterface(10000,10, seed=7)
u.generate_treatment(random_assignment=True, treatment_option_weights = [0, 0, 1, 0, 0])
y, X, assignment, treatment = u.output_data(False)

y_treated = y[assignment==1]
y_not_treated = y[assignment==0]

output_difference_plt(y_not_treated, y_treated, file_name = 'continuous_output.png', save = True)


# binary 
u = UserInterface(10000,10, seed=7)
u.generate_treatment(random_assignment=True, treatment_option_weights = [0, 0, 1, 0, 0])
y, X, assignment, treatment = u.output_data(True)

y_treated = y[assignment==1]
y_not_treated = y[assignment==0]

output_difference_plt(y_not_treated, y_treated, binary = True, file_name = 'binary_output.png', save = True)

















