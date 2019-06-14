from opossum import UserInterface



if __name__ == "__main__":
    
    ##### propensity score plot
    u = UserInterface(10000, 7, seed=5)
    u.generate_treatment(random_assignment = True, 
                         assignment_prob = 0.5, 
                         constant_pos = False, 
                         constant_neg = False,
                         heterogeneous_pos = False, 
                         heterogeneous_neg = False, 
                         no_treatment = False, 
                         discrete_heterogeneous = True,
                         treatment_option_weights = None, # [constant_pos, constant_neg, heterogeneous_pos, heterogeneous_neg, no_treatment] percentage of each
                         intensity = 5)
    y, X, assignment, treatment = u.output_data()



