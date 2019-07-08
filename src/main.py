from opossum import UserInterface



if __name__ == "__main__":
    
    ##### propensity score plot
    u = UserInterface(10000, 50, seed=5, categorical_covariates = None)
                      # replace None with either: int, list of 2 ints, list of 
                      # 1 int and 1 list of ints, where [num of covariates, num
                      # of categories]
    u.generate_treatment(random_assignment = True, 
                         assignment_prob = 0.5, 
                         constant_pos = False, 
                         constant_neg = False,
                         heterogeneous_pos = True, 
                         heterogeneous_neg = False, 
                         no_treatment = False, 
                         discrete_heterogeneous = False,
                         treatment_option_weights = None, 
                         # replace None with vector [constant_pos, constant_neg, 
                         # heterogeneous_pos, heterogeneous_neg, no_treatment, 
                         # discrete_heterogeneous] percentage of each
                         intensity = 5)
    y, X, assignment, treatment = u.output_data(binary=False)


