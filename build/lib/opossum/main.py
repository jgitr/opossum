from opossum import UserInterface



if __name__ == "__main__":
    u = UserInterface(10000, 50, seed=None, categorical_covariates = None)
                      # If wanted, replace None with either: int, list of 2 
                      # ints, list of 1 int and 1 list of ints, where 
                      # [num of covariates, num of categories]
    u.generate_treatment(random_assignment = True, 
                         assignment_prob = 0.5, 
                         constant_pos = True, 
                         constant_neg = False,
                         heterogeneous_pos = False, 
                         heterogeneous_neg = False, 
                         no_treatment = False, 
                         discrete_heterogeneous = False,
                         treatment_option_weights = None, 
                         # If wanted replace None with vector [constant_pos, 
                         # constant_neg, heterogeneous_pos, heterogeneous_neg, 
                         # no_treatment, discrete_heterogeneous] percentage of 
                         # each
                         intensity = 5)
    y, X, assignment, treatment = u.output_data(binary=False, 
                                                x_y_relation = 
                                                'partial_nonlinear_simple')


