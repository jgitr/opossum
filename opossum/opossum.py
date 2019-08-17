from scipy import random, stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from helpers import standardize, is_pos_def, adjusting_assignment_level, \
                    revert_string_prob, relation_fct

class SimData:
    """
    Main class that package is built on
    """

    def __init__(self, N, k, seed):
        '''
        Parameters:
            N (int): Number of observations
            k (int): Number of covariates
        Attributes:
            weights_treatment_assignment (numpy array): Weight vector, drawn 
                from a uniform distribution U(0,1), of length k. It is used to 
                weight covariates when assigning treatment non-randomly and 
                when creating heterogeneous treatment effects.
            weights_covariates_to_outputs (numpy array) Weight vector, drawn 
                from a beta distribution Beta(1,5), of length k. It is used to 
                weight covariate importance when creating output y from X.
            z_set_size_assignment (int): Number of covariates in subset Z of X
                that are used to assign treatment non-randomly.
            z_set_size_treatment (int): Number of covariates in subset Z of X 
                that are used to create heterogeneous treatment effects.
            interaction_num (int): Number of interaction terms that are 
                randomly created if chosen in output creation. 
        '''
        if seed is not None:
            random.seed(seed) # For debugging
        self.N = N # Natural, number of observations
        self.k = k # Natural, number of covariates
        
        # initilizing weight vector for treatment assignment 
        # using random weights from U[0,1]
        self.weights_treatment_assignment = np.random.uniform(0,1,self.k)
        # doing the same for relation of X and y with 
        # beta distribution (alpha=1, beta=5)
        self.weights_covariates_to_outputs =  np.random.beta(1,5,self.k) 
        # set size of subset Z of X for heterogeneous treatment creation
        self.z_set_size_treatment = np.int(self.k/2)
        # set size of subset Z of X for non-random treatment assignment        
        self.z_set_size_assignment = np.int(self.k/2)
        # set number of covariates used for creating interaction terms of X
        self.interaction_num = int(np.sqrt(self.k))


    def generate_covariates(self, categorical_covariates):

        """
        Generates the covariates matrix 
        
        Parameters:
            categorical_covariates (int or list): Either an int, indicating the
                number of categories that all covariates are made of; a list 
                with 2 ints, the first int indicating the number of covariates 
                and the second the number of categories; or a list with one int
                and a list of ints, where the list of ints includes the 
                different number of categories wanted.        
        ...
        
        Returns:
            None
        """

        A = random.rand(self.k, self.k) 
        # To allow for negative correlations
        overlay_matrix = np.random.randint(2, size=(self.k, self.k))
        overlay_matrix[overlay_matrix == 0] = -1
        # correcting for number of covariates
        A = (10/(self.k)) * A * overlay_matrix
        # Assuring positive definitness         
        sigma = np.dot(A, A.transpose())  
        
        # Positive Definite Check
        if not is_pos_def(sigma):
            raise ValueError('sigma is not positive definite!')
        
        # Expected values
        mu = np.repeat(0, self.k)
        # Final covariates
        X = np.random.multivariate_normal(mu, sigma, self.N)
        
        ### Categorical variables ###
        
        if categorical_covariates == None:
            self.X = X
            return None
        
        # Single integer: all covariates become categorical with int categories
        if type(categorical_covariates) == int:
            # Standardizing column wise to [0,1]
            X = (X - np.min(X, axis=0))/(np.max(X, axis=0)-np.min(X, axis=0))
            
            X_categorical = np.zeros(X.shape)
            # Creating categorical variables with chosen number of categories
            for c in range(categorical_covariates-1):
                X_categorical += np.random.binomial(1, X)
            
            X = X_categorical
            
        elif type(categorical_covariates) == list and len(categorical_covariates) == 2:
            num_cat_covariates = categorical_covariates[0]
            if num_cat_covariates > self.k:
                raise Warning("Number of catigorical variables ({}) is greater "
                              "than number of covariates ({}). \nAll {} "
                              "covariates are made categorical."
                              .format(num_cat_covariates, self.k, self.k))
            
            X_cat_part = X[:,:num_cat_covariates]
            # Standardizing column wise to [0,1]
            X_cat_part = (X_cat_part - np.min(X_cat_part, axis=0)) \
                         /(np.max(X_cat_part, axis=0)-np.min(X_cat_part, axis=0))
            X_categorical = np.zeros(X_cat_part.shape)
            
            # List with 2 ints: Chosen number of covarites become categorical
            # with chosen number of categories
            if type(categorical_covariates[1]) ==  int:
                num_categories = categorical_covariates[1]
                # Creating categorical variables with chosen number of categories
                for c in range(num_categories-1):
                    X_categorical += np.random.binomial(1, X_cat_part)
                
            # List with int and list of ints: Chosen number of covariates become 
            # categorical according to chosen list of number of categories
            elif type(categorical_covariates[1]) == list:
                
                num_categories_list = categorical_covariates[1]
                
                # Creating vector with wanted category numbers 
                category_type_vector = np.array((num_cat_covariates // 
                                                 len(num_categories_list) + 1) 
                                                * num_categories_list)
                # Making sure it has the wanted length, which is the number of
                # wanted categorical covariates
                category_type_vector = category_type_vector[:num_cat_covariates]
                
                start = 0 
                end = 0
                # Selecting one by one wanted category numbers
                for num_categories in num_categories_list:
                    
                    end += np.sum(category_type_vector == num_categories)
                    # Adding up bernouli outcomes to get categorical variables
                    for c in range(num_categories-1):
                        X_categorical[:, start:end] += np.random.binomial(1, 
                                                       X_cat_part[:, start:end])
                    
                    start = end
                    
            else:
                raise ValueError("categorical_covariates needs to be either an "
                                 "int, a list of 2 ints, or a list of one int "
                                 "and a list of ints. \nMake sure that the "
                                 "second item of the list is an int or a list "
                                 "of ints")
            
            X[:,:num_cat_covariates] = X_categorical                
            

        else:
            raise ValueError("categorical_covariates needs to be either an int, "
                             "a list of 2 ints, or a list of one int and a list "
                             "of ints. \nMake sure that it is a list of length "
                             "2 or a single int" )
                

        self.X = X

        return None



    def generate_treatment_assignment(self, random, assignment_prob):
        
        """
        Generates treatment assignment vector
        
        Parameters:
            random (boolean): If True, treatment is assigned randomly 
                according to assignment_prob parameter. If False, treatment 
                assignment is determined depending on covariates. 
                (default is True)
            assignment_prob (float or string): The probability with which 
                treatment is assigned. In the case of random assignment, it can
                be a float with 0 < prob < 1. If assignment is not random it
                should be one of the following strings: 'low', 'medium', 'high'.
                The strings stand for the values 0.35, 0.5, 0.65 respectively
                and can also be used in the non-random case.
                (default is 0.5)
            
        ...
            
        Returns:
            None
                
        """       
        # random treatment assignment
        if random:
            m_0 = assignment_prob  # probability
            
            # reverting strings like 'low' to float prob like 0.35, if necessary
            if isinstance(m_0, str):
                m_0 = revert_string_prob(m_0)                    
            
            # propensity scores for each observation 
            self.propensity_scores = np.repeat(m_0,self.N)
                                
                
        else:
            # Creating index for selection of covariates for assignment
            a_idx = np.concatenate((np.zeros(self.k - self.z_set_size_assignment)
                                    , np.ones(self.z_set_size_assignment)))
            np.random.shuffle(a_idx)
            # Selecting covariates 
            X_a = self.X[:, a_idx == 1].copy()
            
            noise = np.random.uniform(0,0.25, self.N)
                        
            a = np.dot(X_a, self.weights_treatment_assignment[a_idx == 1]) \
                + noise  
            
            try:
                # get value that adjusts z and thus propensity scores 
                z_adjustment = adjusting_assignment_level(assignment_prob)
                
            except KeyError:
                z_adjustment = 0
                
                # making sure default of 0.5 does not give warning
                if assignment_prob != 0.5:                    
                    raise Warning('When assignment is not random, expected '
                                  'assignment_prob can only be \'low\': 0.35, '
                                  '\'medium\': 0.5, or \'high\': 0.65. Now '
                                  'medium is chosen ')
            
            # Using empirical mean, sd
            a_mean = np.mean(a) 
            a_sigma = np.std(a)
            # normalizing 'a' vector and adjust if chosen
            z = (a - a_mean) / a_sigma + z_adjustment    
            
            # using normalized vector z to get probabilities from normal cdf
            # to later assign treatment with binomial in D
            m_0 = stats.norm.cdf(z)
            
            # propensity scores for each observation 
            self.propensity_scores = m_0

        # creating array out of binomial distribution that assigns treatment 
        # according to probability m_0
        self.D = np.random.binomial(1, m_0, self.N)

        
        return None



    def generate_treatment_effect(self, treatment_option_weights, constant_pos, 
                                  constant_neg, heterogeneity_pos, 
                                  heterogeneity_neg, no_treatment, 
                                  discrete_heterogeneity, intensity):

        """
        Generates chosen kinds of treatment effects
        
        Parameters:
            constant_pos (boolean): If True, the treatment effect is a positive
                constant.
                (default is True)
            constant_neg (boolean): If True, the treatment effect is a negative
                constant.
                (default is False)
            heterogeneous_pos (boolean): If True, the treatment effect is 
                positive and heterogeneous, i.e. it depends on a number of 
                covariates and varries in size. 
                (default is False)
            heterogeneous_neg (boolean): If True, the treatment effect is 
                negative and heterogeneous, i.e. it depends on a number of 
                covariates and varries in size.
                (default is False)
            no_treatment (boolean): If True, then there is no treatment effect.
                (default is False)
            discrete_heterogeneous (boolean): If True, then the treatment 
                effect consists of 2 values of different size. The size 
                is determined by a subset of covariates.
                (default is False)
            treatment_option_weights (list): List of length 6 with weights of 
                wanted treatment effects in the following order:
                [const_pos, const_neg, heterogeneous_pos, heterogeneous_neg, 
                no_treatment, discrete_heterogeneous]. Its values need to sum 
                up to 1. If chosen, the values overwrite the boolean parameters 
                for each treatment effect.
                (default is None)
            intensity (int or float): Value affects the size of the treatment 
                effect. Needs to be between 1 and 10. Formula for the actual
                magnitude of the treatment effects are: 
                const: intensity*0.2, heterogeneous: [0, intensity*0.4]
                discrete_heterogeneous: {intensity*0.1, intensity*0.2}
                (default is 5)
                
        When creating the treatment effect, there are two ways to choose which 
        kind of effects are used. Either one sets all treatment effect booleans
        that are wanted to True and then they are equally weighted created, or
        one gives a list of length 6 with weights of the wanted distribution of 
        effects to the parameter treatment_option_weights, which overwrites 
        whatever booleans where chosen before. 
        
        Returns:
            None
        """

        # length of treatment_option_weights vector/
        # number of treatment effect options
        tow_length = 6
        
        if intensity > 10 or intensity < 1:
            raise ValueError("intensity needs to be an int or float value of " 
                             "[1,10]")
        
        if treatment_option_weights is not None:            
            # make sure it's a numpy array
            treatment_option_weights = np.array(treatment_option_weights)
            if np.around(np.sum(treatment_option_weights),3) !=1:
                raise ValueError('Values in treatment_option_weights-vector '
                                 'must sum up to 1')
            if len(treatment_option_weights) !=tow_length:
                raise ValueError('Treatment_option_weights-vector must be of '
                                 'length {}'.format(tow_length))
            
            
            # take times N to get absolute number of each option
            absolute_ratio = (self.N*treatment_option_weights).astype(int)
            
            # adjusting possible rounding errors by increasing highest value 
            if sum(absolute_ratio) < self.N:
                index_max = np.argmax(treatment_option_weights)
                absolute_ratio[index_max] = absolute_ratio[index_max] \
                                            + (self.N-sum(absolute_ratio))
            
            # fill up index-array with options 1-6 according to the weights
            weight_ratio_index = np.zeros((self.N,))
            counter = 0
            for i in range(len(absolute_ratio)):
                weight_ratio_index[counter:counter+absolute_ratio[i],] = i+1
                counter += absolute_ratio[i]
            # shuffle 
            np.random.shuffle(weight_ratio_index)
            
            n_idx = weight_ratio_index
            
            # overwriting booleans according to given treatment_option_weights             
            options_boolean = treatment_option_weights > 0
            
            constant_pos, constant_neg, heterogeneity_pos, heterogeneity_neg, \
            no_treatment, discrete_heterogeneity  = tuple(options_boolean)
            
        # Process options
        options = []
        
        options_boolean = np.array([constant_pos, constant_neg, heterogeneity_pos, 
                                    heterogeneity_neg, no_treatment, 
                                    discrete_heterogeneity])

        # selecting wanted treatment options into list        
        for i in range(len(options_boolean)):
            if options_boolean[i]:
                options.append(i+1)
        
        if treatment_option_weights is None:
            if options ==[]:
                raise ValueError("At least one treatment effect option must be"
                                 "True")
            # assigning which individual gets which kind of treatment effect 
            treatment_option_weights = np.zeros(len(options_boolean))
            treatment_option_weights[options_boolean] = 1/np.sum(options_boolean)
            # from options 1-6
            n_idx = np.random.choice(options, self.N, True)
            
            
            
        # array to fill up with theta values         
        theta_combined = np.zeros(self.N)
        
        if constant_pos:
            # Option 1
            con = 0.2*intensity 
            
            theta_combined[n_idx == 1] = con

        if constant_neg:
            # Option 2
            con = -0.2*intensity 
            
            theta_combined[n_idx == 2] = con


        if heterogeneity_pos or heterogeneity_neg:
            # Options 3 & 4
            
            # creating index vector that assigns which covariates are part of Z
            h_idx = np.concatenate((np.zeros(self.k - self.z_set_size_treatment),
                                    np.ones(self.z_set_size_treatment)))
            np.random.shuffle(h_idx)
            
            X_h = self.X[:,h_idx == 1].copy()

            w = np.random.normal(0,0.25,self.N)

            weight_vector_adj = self.weights_treatment_assignment[h_idx == 1]
            
            gamma = np.sin(np.dot(X_h, weight_vector_adj)) + w  
            
            # Standardize on [0,g(intensity)], g(): some function e.g. g(x)=0.2x
            theta_option2 = standardize(gamma, intensity*0.4, 0)
            # calculating percentage quantile of negative treatment effect weights 
            percentage_neg = treatment_option_weights[3] \
                            / (treatment_option_weights[2]+ 
                               treatment_option_weights[3])
            # get quantile value that splits distribution into two groups
            quantile_value = np.quantile(theta_option2, percentage_neg)
            # move distribution into negative range by the amount of quantile 
            # value
            theta_option2 = theta_option2 - quantile_value

            theta_combined[(n_idx == 3) | (n_idx == 4)] \
            = theta_option2[(n_idx == 3) | (n_idx == 4)]
            

        if no_treatment:
            # Option 5
            theta_combined[n_idx == 5] = 0 
        
        if discrete_heterogeneity:
            # Option 6
            # assigning randomly which covariates affect treatment effect
            # creating index vector
            dh_idx = np.concatenate((np.zeros(self.k - self.z_set_size_treatment),
                                     np.ones(self.z_set_size_treatment)))
            np.random.shuffle(dh_idx)
            
            # choosing covariates in Z
            X_dh = self.X[:,dh_idx == 1].copy()
            # adjusting weight vector to length of Z 
            weight_vector_adj = self.weights_treatment_assignment[dh_idx == 1]
            
            a = np.sin(np.dot(X_dh,weight_vector_adj))

            a = standardize(a, 1,0) 
            theta_dh = np.random.binomial(1,a).astype(float) * -1
#            # normalizing 'a' vector
#            a_mean = np.mean(a)
#            a_sigma = np.std(a)
#            z = (a - a_mean) / a_sigma           
#            # create probabilities
#            dh_effect_prob = stats.norm.cdf(z)
#            
#            # assigning low and high treatment outcome 
#            theta_dh = np.random.binomial(1,dh_effect_prob).astype(float) * -1
            
            low_effect = 0.1 * intensity 
            
            high_effect = 0.2 * intensity
                
            theta_dh[theta_dh == 0] = low_effect
            
            theta_dh[theta_dh == -1] = high_effect
            
            theta_combined[n_idx == 6] = theta_dh[n_idx == 6]
            
            
        # Assign identifier 0 for each observation that did not get assigned to 
        # treatment        
        n_idx[self.D == 0] = 0
        
        # create vector that shows 0 for not assigned observations and 
        # treatment-type (1-6) for assigned ones 
        self.treatment_effect_type = n_idx
        # vector that includes sizes of treatment effects for each observation
        self.treatment_effect = theta_combined

        return None
    
    def generate_realized_treatment_effect(self):
        """
        Model-wise: Theta_0 * D
        :return:  Extract Treatment Effect where Treatment has been assigned
        """

        return self.get_treatment_effect() * self.get_treatment_assignment()

    def generate_noise(self):
        """
        model-wise: U or V
        Restriction: Expectation must be zero conditional on X, D.
        However, the expectation is independent anyways.
        :return: One-dim. array of normally distributed rv with 0 and 1
        """
        return np.random.normal(0, 1, self.N)



    def generate_outcome_variable(self, binary, x_y_relation):
        """
        Generates g_0(X), output variable y and returns simulated variables
        
        Parameters:
            binary (boolean): If True output is going to be binary, otherwise
                continuous. 
            x_y_relation (string): Chooses the simulated relationship between 
                X and y. Possible values are: 
                'linear_simple', 'linear_interaction', 
                'partial_nonlinear_simple', 'partial_nonlinear_interaction',
                'nonlinear_simple', 'nonlinear_interaction'  

        ...
        
        Returns:
            tuple
        """
        # Creating random interaction terms of covariates
        interaction_idx_1 = np.random.choice(np.arange(self.k), 
                                             self.interaction_num)
        interaction_idx_2 = np.random.choice(np.arange(self.k), 
                                             self.interaction_num)
        
        self.X_interaction \
        = self.X[:,interaction_idx_1] * self.X[:,interaction_idx_2]
        
        self.weights_interaction \
        = self.weights_covariates_to_outputs[interaction_idx_1]
        
        try:
            self.g_0_X = relation_fct(self, x_y_relation)
        except TypeError:
            raise ValueError('x_y_relation needs to be one of the following ' 
                             'strings:\n"linear_simple", "linear_interaction", ' 
                             '"partial_nonlinear_simple", ' 
                             '"partial_nonlinear_interaction", '
                             '"nonlinear_simple", "nonlinear_interaction"')
                
        if not binary:
            # Theta_0 * D
            realized_treatment_effect = self.generate_realized_treatment_effect()
            # + g_0(x) + U
            y = realized_treatment_effect + self.g_0_X + self.generate_noise()  
        
        if binary:
            # generating y as probability between 0.1 and 0.9
            y = self.g_0_X #+ self.generate_noise()
            y_probs = standardize(y, 0.1, 0.9)
            # generate treatment effect as probability
            realized_treatment_effect = self.generate_realized_treatment_effect()/10 
            # max. range of treatment effect is [-4,4] (with intensity 10 and 
            # only choosing pos. or neg. effect) thus dividing by 10 assures 
            # that additional probability is at most 0.4
            
            
            y_probs += realized_treatment_effect
            y_probs = np.clip(y_probs, 0, 1)
            y = np.random.binomial(1, y_probs, self.N)
            
              
        return y, self.X, self.D, realized_treatment_effect



    def visualize_correlation(self):
        """ Generates Correlation Matrix of the Covariates """

        corr = np.corrcoef(self.X, rowvar = False)
        sns.heatmap(corr, annot = True)
        plt.show()
        return None

    def __str__(self):
        return "N = " + str(self.N) + ", k = " + str(self.k)
    
    def get_N(self):
        return self.N

    def get_k(self):
        return self.k
    
    def set_N(self, new_N):
        self.N = new_N

    def set_k(self, new_k):
        self.k = new_k
        
    def get_X(self):
        return self.X

    def get_g_0_X(self):
        return self.g_0_X

    def get_treatment_assignment(self):
        return self.D
    
    def get_treatment_effect(self):
        return self.treatment_effect



##### New class that includes SimData class by initizilaizing it internally and 
##### only displays a few simple functions to user


class UserInterface:
    '''
    Class to wrap up all functionalities and give user just the functions that 
    are necessary to create the wanted variables y, X, D, and treatment.
    '''
    def __init__(self, N, k, seed = None, categorical_covariates = None):
        '''
        Initilizes needed Classes and generates covariates
        
        Parameters:
            N (int): Number of observations
            k (int): Number of covariates
            seed (int): Random seed to allow reproducing of results
                (default is None)
            categorical_covariates (int or list): Either an int, indicating the
                number of categories that all covariates are made of; a list 
                with 2 ints, the first int indicating the number of covariates 
                and the second the number of categories; or a list with one int
                and a list of ints, where the list of ints includes the 
                different number of categories wanted.
                
        Attributes:
            weights_treatment_assignment (numpy array): Weight vector, drawn 
                from a uniform distribution U(0,1), of length k. It is used to 
                weight covariates when assigning treatment non-randomly and 
                when creating heterogeneous treatment effects.
            weights_covariates_to_outputs (numpy array) Weight vector, drawn 
                from a beta distribution Beta(1,5), of length k. It is used to 
                weight covariate importance when creating output y from X.
            z_set_size_assignment (int): Number of covariates in subset Z of X
                that are used to assign treatment non-randomly.
            z_set_size_treatment (int): Number of covariates in subset Z of X 
            that are used to create heterogeneous treatment effects. 
        '''

        self.backend = SimData(N, k, seed)
        self.backend.generate_covariates(categorical_covariates 
                                         = categorical_covariates)
        
        
    def generate_treatment(self, random_assignment = True, 
                           assignment_prob = 0.5,
                           constant_pos = True, 
                           constant_neg = False,
                           heterogeneous_pos = False, 
                           heterogeneous_neg = False, 
                           no_treatment = False, 
                           discrete_heterogeneous = False,
                           treatment_option_weights = None,
                           intensity = 5):
        '''
        Assigns and generates treatment effect        
        
        Parameters:
            random_assignment (boolean): If True, treatment is assigned randomly 
                according to assignment_prob parameter. If False, treatment 
                assignment is determined depending on covariates. 
                (default is True)
            assignment_prob (float or string): The probability with which 
                treatment is assigned. In the case of random assignment, it can
                be a float with 0 < prob < 1. If assignment is not random it
                should be one of the following strings: 'low', 'medium', 'high'.
                The strings stand for the values 0.35, 0.5, 0.65 respectively
                and can also be used in the non-random case.
                (default is 0.5)
            constant_pos (boolean): If True, the treatment effect is a positive
                constant.
                (default is True)
            constant_neg (boolean): If True, the treatment effect is a negative
                constant.
                (default is False)
            heterogeneous_pos (boolean): If True, the treatment effect is 
                positive and heterogeneous, i.e. it depends on a number of 
                covariates and varries in size. 
                (default is False)
            heterogeneous_neg (boolean): If True, the treatment effect is 
                negative and heterogeneous, i.e. it depends on a number of 
                covariates and varries in size.
                (default is False)
            no_treatment (boolean): If True, then there is no treatment effect.
                (default is False)
            discrete_heterogeneous (boolean): If True, then the treatment 
                effect consists of 2 values of different size. The size 
                is determined by a subset of covariates.
                (default is False)
            treatment_option_weights (list): List of length 6 with weights of 
                wanted treatment effects in the following order:
                [const_pos, const_neg, heterogeneous_pos, heterogeneous_neg, 
                no_treatment, discrete_heterogeneous]. Its values need to sum 
                up to 1. If chosen, the values overwrite the boolean parameters 
                for each treatment effect.
                (default is None)
            intensity (int or float): Value affects the size of the treatment 
                effect. Needs to be between 1 and 10. Formula for the actual
                magnitude of the treatment effects are: 
                const: intensity*0.1, heterogeneous: [0, intensity*0.2]
                discrete_heterogeneous: {intensity*0.05, intensity*0.1}
                (default is 5)
        
        Treatment assignment can be done randomly or determined by a subset Z of 
        covariates. The assignment probability can be freely chosen between 0 
        and 1 in the random case and from 3 levels ('low','medium','high') in 
        the non-random case.
        When creating the treatment effect, there are two ways to choose which 
        kind of effects are used. Either one sets all treatment effect booleans
        that are wanted to True and then they are equally weighted created, or
        one gives a list of length 6 with weights of the wanted distribution of 
        effects to the parameter treatment_option_weights, which overwrites 
        whatever booleans where chosen before. 
        
        Returns:
            None

'''
        self.backend.generate_treatment_assignment(random_assignment, 
                                                   assignment_prob)
        self.backend.generate_treatment_effect(treatment_option_weights, 
                                               constant_pos,
                                               constant_neg, heterogeneous_pos, 
                                               heterogeneous_neg, no_treatment, 
                                               discrete_heterogeneous, 
                                               intensity)

        return None

    def output_data(self, binary = False, 
                    x_y_relation = 'partial_nonlinear_simple'):
        '''
        Generates g_0(X), output variable y and returns simulated variables
        
        Parameters:
            binary (boolean): If True output is going to be binary, otherwise
                continuous. 
                (default is False)
            x_y_relation (string): Chooses the simulated relationship between 
                X and y. Possible values are: 
                'linear_simple', 'linear_interaction', 
                'partial_nonlinear_simple', 'partial_nonlinear_interaction',
                'nonlinear_simple', 'nonlinear_interaction'  
                (default is 'partial_nonlinear_simple')
                
        When simulating a dataset, there are different options to transform X
        into y. It can be linear or non-linear in different ways. The options
        that can be chosen for x_y_relation correspond to the following 
        functions:
        linear -> y ~ X
        partial non-linear -> y ~ 2.5*cos(X)^3 + 0.5*X
        non-linear -> y ~ 3*cos(X)^3 + 0.6*X
        Depending on the addition "simple" or "interaction" X consists only of
        single covariates x_i or additionally on random interaction terms of 
        some of the covariates x_i*x_j; i,j in{1,...,k}
        Generates output array "y" the following way: 
        Y = Theta_0 * D + g_0(X) + U,
        where Theta_O is the treatment effect of each observation, D the dummy 
        vector for assigning treatment, g_0() the transformation function, and 
        U a normal-distributed noise-/error term
        
        Returns:
            tuple: A tuple with variables y, X, assignment_vector, 
                    treatment_vector
        '''                

        return self.backend.generate_outcome_variable(binary, x_y_relation)
    
    def plot_covariates_correlation(self):
        '''
        Shows a correlation heatmap of the covariates 
        '''
        self.backend.visualize_correlation()
        return None
    
    def get_propensity_scores(self):
        return self.backend.propensity_scores
    
    def get_weights_treatment_assignment(self):
        return self.backend.weights_treatment_assignment
    
    def get_weigths_covariates_to_outputs(self):
        return self.backend.weights_covariates_to_outputs
    
    def get_treatment_effect_type(self):
        '''
        Gives a vector with the type of treatment effect of each observation
        
        Treatment types are accordingly:
            0 No treatment assigned
            1 positive constant effect
            2 negative constant effect
            3 positive heterogeneous effect
            4 negative heterogeneous effect
            5 no treatment effect (but assigned)
            6 discrete heterogeneous effect
            
        Returns:
            numpy array: n*1 array with treatment type of each observation
        '''
        return self.backend.treatment_effect_type
        
    def set_weights_treatment_assignment(self, new_weight_vector):
        if len(new_weight_vector) is not self.backend.get_k():
            raise ValueError('New weight vector needs to be of dimension k')
            
        self.backend.weights_treatment_assignment = np.array(new_weight_vector)
            
    def set_weights_covariates_to_outputs(self, new_weight_vector):
        if len(new_weight_vector) is not self.backend.get_k():
            raise ValueError('New weight vector needs to be of dimension k')
        
        self.backend.weights_covariates_to_outputs= np.array(new_weight_vector)
        
    def set_subset_z_size_treatment(self, new_size):
        '''
        Adjusts number of covariates used to create heterogeneous treatment
        
        Parameters:
            new_size (int): Wanted number of covariates to determine 
                heterogeneous treatment effects. Needs to be in set {1,...,k}.
        
        For the heterogeneous treatment effects, the resulting effects depend
        on values of covariates in a subset Z of X. This method adjusts how 
        many covariates are randomly chosen to be in Z. Apply before using 
        generate_treatment().
        '''
        
        if new_size < 1 or new_size > self.backend.get_k():
            raise ValueError('Size of subset Z needs to be within [1,k]')
            
        self.backend.z_set_size_treatment = new_size

    def set_subset_z_size_assignment(self, new_size):
        '''
        Adjusts number of covariates used to non-randomly assign treatment 
        
        Parameters:
            new_size (int): Wanted number of covariates to determine 
                treatment assignment. Needs to be in set {1,...,k}.
        
        Non-random treatment assignment depends on values of covariates in a 
        subset Z of X. This method adjusts how many covariates are randomly
        chosen to be in Z. Apply before using generate_treatment()
        '''
        
        if new_size < 1 or new_size > self.backend.get_k():
            raise ValueError('Size of subset Z needs to be within [1,k]')
            
        self.backend.z_set_size_assignment = new_size

    def set_interaction_num(self, new_num):
        '''
        Adjust number of interaction terms used in output creation
        
        Parameters:
            new_num (int): Wanted number of interaction terms x_i*x_j that 
                should be added to single covariates.
            
        When choosing one of the '..._interaction' options for x_y_relation
        in output_data(), interaction_num of interaction terms are randomly 
        created and added. The internal default value for that attribute is 
        sqrt(k). This method changes the default value to the chosen integer.
        Use between initilizing the class and using output_data().
        
        Return:
            None
        '''
        if not isinstance(new_num, int):
            raise ValueError('new_num needs to be of type int')
        
        self.interaction_num = new_num
        
    def __str__(self):
        return "N = " + str(self.backend.get_N()) + ", k = " + \
                str(self.backend.get_k())






