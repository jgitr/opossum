from scipy import random, stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from helpers import standardize, is_pos_def, adjusting_assignment_level, revert_string_prob

class SimData:
    """
    Model: Refer to Paper in Readme

    # Y = Outcome (dependend variable). Either continuous or binary.
    # N = Number of observations (real number)
    # k = Number of covariates (real number). At least 10
    # random_d = treatment assignment: (Either T for random assignment or F for confounding on X)
    # theta = treatment effect: (Either real number for only one theta, or "binary" {0.1,0.3}, "con" for continuous values (0.1,0.3) or "big" for {1,0.4})
    # var = Size of the variance (Noise-level)

    """

    def __init__(self, N, k, seed):
        if seed is not None:
            random.seed(seed) # For debugging
        self.N = N # Natural, number of observations
        self.k = k # Natural, number of covariates
        
        # initilizing weight vector for treatment assignment 
        # using random weights from U[0,1]
        self.weights_treatment_assignment = np.random.uniform(0,1,self.k)
        # doing the same for relation of X and y with beta distribution (alpha=1, beta=5)
        self.weights_covariates_to_outputs =  np.random.beta(1,5,self.k) #np.random.uniform(0,1,self.k)
        
        # set size of subset Z of X for heterogeneous treatment creation
        self.z_set_size = np.int(self.k/2)
        
    def generate_outcome_variable(self, binary):
        """
        Model-wise Y
        options: binary, multilevel(discrete), continuous

        Y = Theta_0 * D + g_0(X) + U
        D = m_0(X) + V
        Theta_0 = t_0(Z) + W

        """
        
        if not binary:
            realized_treatment_effect = self.generate_realized_treatment_effect() # Theta_0 * D
            y = realized_treatment_effect + self.g_0_X + self.generate_noise()  # * g_0(x) + U
        
        if binary:
            # generating y as probability between 0.1 and 0.9
            y = self.g_0_X + self.generate_noise()
            y_probs = standardize(y, 0.1, 0.9)
            # generate treatment effect as probability
            
            realized_treatment_effect = self.generate_realized_treatment_effect()/5 
            # max. range of treatment effect is [-2,2] (with intensity 10 and only choosing pos. or neg. effect)
            # thus dividing by 5 assures that additional probability is at most 0.4
            
            
            y_probs += realized_treatment_effect
            y_probs = np.clip(y_probs, 0, 1)
            y = np.random.binomial(1, y_probs, self.N)
            
              
        return y, self.X, self.D, realized_treatment_effect

    def generate_covariates(self, categorical_covariates, nonlinear = True, skew = True):

        """
        Generates the covariates matrix and its non-linear transformation
        
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

        # 1)
        # Sigma
        A = random.rand(self.k, self.k) # drawn from uniform distribution in [0,1]

        # Make sure negative covariance exists, too.
        overlay_matrix = np.random.randint(2, size=(self.k, self.k))  # set -1 where 0
        overlay_matrix[overlay_matrix == 0] = -1
        A = (10/(self.k)) * A * overlay_matrix
        
        self.A = A
        
        sigma = np.dot(A, A.transpose())  # a matrix multiplied with its transposed is aaaalways positive definite
        
        # Positive Definite Check
        if not is_pos_def(sigma):
            raise ValueError('sigma is not positive definite!')


        # 2)
        mu = np.repeat(0, self.k)


        X = np.random.multivariate_normal(mu, sigma, self.N)
        
        
        if skew:
            def skew_data(x):
                x[x < 1] = x[x < 1] - 1
                x[x >= 1] = np.log(x[x >= 1])
                return x
            X = skew_data(X)


        if nonlinear:
            # transforming by cos(X*weights)²
            self.g_0_X = np.cos(np.dot(X,self.weights_covariates_to_outputs))**3 + 0.2*np.dot(X,self.weights_covariates_to_outputs)
        else:
            # If not nonlinear, then g_0(X) is just the identity 
            self.g_0_X = np.dot(X,np.repeat(1/self.k,self.k))  # dim(X) = n * k -> need to vector multiply with vector shaped [k,1]
            
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
                raise Warning('Number of catigorical variables ({}) is greater than number of covariates ({}). \nAll {} covariates are made categorical.'.format(num_cat_covariates, self.k, self.k))
            
            X_cat_part = X[:,:num_cat_covariates]
            # Standardizing column wise to [0,1]
            X_cat_part = (X_cat_part - np.min(X_cat_part, axis=0))/(np.max(X_cat_part, axis=0)-np.min(X_cat_part, axis=0))
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
                category_type_vector = np.array((num_cat_covariates // len(num_categories_list) + 1) * num_categories_list)
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
                        X_categorical[:, start:end] += np.random.binomial(1, X_cat_part[:, start:end])
                    
                    start = end
                    
            else:
                raise ValueError("categorical_covariates needs to be either an int, a list of 2 ints, or a list of one int and a list of ints. \nMake sure that the second item of the list is an int or a list of ints")
            
            X[:,:num_cat_covariates] = X_categorical                
            

        else:
            raise ValueError("categorical_covariates needs to be either an int, a list of 2 ints, or a list of one int and a list of ints. \nMake sure that it is a list of length 2 or a single int" )
                
        
#        if categorical_covariates >= 1:
#            """
#            Categorical realizations of X.
#            The idea is to use some standardize some x element of X in [0,1] and then use that as p parameter for a bernoulli random variable.
#            Repeatedly sum the realization and treat as category.
#            Seed allows for different bernoulli realizations :)
#            So far transforms all components of X.
#            """
#
#            for i in range(len(X)):
#                category = 0                                                                     # reset for every iteration
#                z = standardize(X[i], 1, 0)                                                      # vector
#            
#            for j in range()
#            
#            for j in range(categorical_covariates):
#                    category += np.random.binomial(1, z)                                         # bernoulli rv is binomial rv with n = 1
#                X[i] = category

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
            a = np.dot(self.X, self.weights_treatment_assignment)    # X*weights -> a (Nx1 vector)
            
            try:
                # get value that adjusts z and thus propensity scores 
                z_adjustment = adjusting_assignment_level(assignment_prob)
                
            except KeyError:
                z_adjustment = 0
                
                # making sure default of 0.5 does not give warning
                if assignment_prob != 0.5:                    
                    raise Warning('When assignment is not random, expected assignment_prob' 
                                  + ' can only be \'low\': 0.35, \'medium\': 0.5,' +
                                  'or \'high\': 0.65. Now medium is chosen ')
            
            # Using empirical mean, sd
            a_mean = np.mean(a) 
            a_sigma = np.std(a)
            # normalizing 'a' vector and adjust if chosen
            z = (a - a_mean) / a_sigma + z_adjustment    
            
            # using normalized vector z to get probabilities from normal pdf
            # to later assign treatment with binomial in D
            m_0 = stats.norm.cdf(z)
            
            # propensity scores for each observation 
            self.propensity_scores = m_0

        # creating array out of binomial distribution that assigns treatment according to probability m_0
        self.D = np.random.binomial(1, m_0, self.N)

        
        return None

    def visualize_correlation(self):
        """
        Generates Correlation Matrix of the Covariates
        :return:
        """

        corr = np.corrcoef(self.X, rowvar = False)
        sns.heatmap(corr, annot = True)
        plt.show()
        return None


    def generate_treatment_effect(self, treatment_option_weights, constant_pos, 
                                  constant_neg, heterogeneity_pos, 
                                  heterogeneity_neg, no_treatment, 
                                  discrete_heterogeneity, intensity):

        """
        """

        # ratio list: [constant, heterogeneity, negative, no_treatment] (treatment_option_weights)
        # e.g. [0, 0.5, 0.1, 0.4], needs to sum up to 1 
        
        # length of treatment_option_weights vector/number of treatment effect options
        tow_length = 6
        
        if intensity > 10 or intensity < 1:
            raise ValueError("intensity needs to be an int or float value of [1,10]")
        
        if treatment_option_weights is not None:            
            # make sure it's a numpy array
            treatment_option_weights = np.array(treatment_option_weights)
            if np.around(np.sum(treatment_option_weights),3) !=1:
                raise ValueError('Values in treatment_option_weights-vector must sum up to 1')
            if len(treatment_option_weights) !=tow_length:
                raise ValueError('Treatment_option_weights-vector must be of length {}'.format(tow_length))
            
            
            # take times N to get absolute number of each option
            absolute_ratio = (self.N*treatment_option_weights).astype(int)
            
            # adjusting possible rounding errors by increasing highest value 
            if sum(absolute_ratio) < self.N:
                index_max = np.argmax(treatment_option_weights)
                absolute_ratio[index_max] = absolute_ratio[index_max] + (self.N-sum(absolute_ratio))
            
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
            
            constant_pos, constant_neg, heterogeneity_pos, heterogeneity_neg, no_treatment, discrete_heterogeneity  = tuple(options_boolean)
            
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
                raise ValueError("At least one treatment effect option must be True")
            # assigning which individual gets which kind of treatment effect 
            treatment_option_weights = np.zeros(len(options_boolean))
            treatment_option_weights[options_boolean] = 1/np.sum(options_boolean)
            # from options 1-6
            n_idx = np.random.choice(options, self.N, True)
            
            
            
        # array to fill up with theta values         
        theta_combined = np.zeros(self.N)
        
        if constant_pos:
            # Option 1
            # Rules: Theta_0 = constant  (c) with c = 0.2
            # constant is independent from covariates 
            con = 0.1*intensity #  constant value for treatment effect
            
            theta_combined[n_idx == 1] = con

        if constant_neg:
            # Option 2
            # Rules: Theta_0 = negative constant  (c) with c = -0.2
            # constant is independent from covariates 
            con = -0.1*intensity #  constant value for treatment effect
            
            theta_combined[n_idx == 2] = con


        if heterogeneity_pos or heterogeneity_neg:
            # Option 3
            # Rules:
            # (1) Apply trigonometric function
            # (2) Standardize the treatment effet within the interval [0,intensity*0.2]
            
            # creating index vector that assigns which covariates are part of Z
            h_idx = np.concatenate((np.zeros(self.k - self.z_set_size),np.zeros(self.z_set_size)))
            np.random.shuffle(h_idx)
            
            X_h = self.X[:,h_idx == 1].copy()

            w = np.random.normal(0,0.25,self.N)

            weight_vector_adj = self.weights_treatment_assignment[h_idx == 1]
            
            gamma = np.sin(np.dot(X_h, weight_vector_adj)) + w  # one gamma for each observation in n
            
            # (2) Standardize on [0,g(intensity)], g(): some function e.g. g(x)=0.2x
            theta_option2 = standardize(gamma, intensity*0.2, 0)
            # calculating percentage quantile of negative treatment effect weights 
            quantile_neg = treatment_option_weights[3]/(treatment_option_weights[2]+ treatment_option_weights[3])
            # get quantile value that splits distribution into two groups
            quantile_value = np.quantile(theta_option2, quantile_neg)
            # move distribution into negative range by the amount of quantile value
            theta_option2 = theta_option2 - quantile_value

            theta_combined[(n_idx == 3) | (n_idx == 4)] = theta_option2[(n_idx == 3) | (n_idx == 4)]
            

        if no_treatment:
            theta_combined[n_idx == 5] = 0 # not really necessary since vector was full of 0 
        
        if discrete_heterogeneity:
            ### assigning randomly which covariates affect treatment effect
            # creating index vector
            dh_idx = np.concatenate((np.zeros(self.k - self.z_set_size),np.ones(self.z_set_size)))
            np.random.shuffle(dh_idx)
            
            # choosing covariates in Z
            X_dh = self.X[:,dh_idx == 1].copy()
            # adjusting weight vector to length of Z 
            weight_vector_adj = self.weights_treatment_assignment[dh_idx == 1]
            
            a = np.dot(X_dh,weight_vector_adj)
            
            # normalizing 'a' vector
            a_mean = np.mean(a)
            a_sigma = np.std(a)
            z = (a - a_mean) / a_sigma          
            
            # create probabilities
            dh_effect_prob = stats.norm.cdf(z)
            
            # assigning low and high treatment outcome 
            theta_dh = np.random.binomial(1,dh_effect_prob).astype(float)
            
            low_effect = 0.05 * intensity
            
            high_effect = 0.1 * intensity
                
            theta_dh[theta_dh == 0] = low_effect
            
            theta_dh[theta_dh == 1] = high_effect
            
            theta_combined[n_idx == 6] = theta_dh[n_idx == 6]
            
            
        # Assign identifier 0 for each observation that did not get assigned to treatment        
        n_idx[self.D == 0] = 0
        
        # create vector that shows 0 for not assigned observations and treatment-type (1-5) for assigned ones 
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
    
#    def visualize_distribution(self, y, treatment):
#        """
#        input: outcome variable y_treated, y_not_treated, treatment
#        :return: Depict
#        # Add histogram data
#    
#        x1 = y[self.D == 0]
#        x2 = y[self.D == 1]
#        #x2 = treatment
#        s Output Distribution
#        """
#
#        # Group data together
#        hist_data = [x1, x2]
#
#        group_labels = ['No Treatment Assignment', 'Treatment Assigment']
#
#        # Create distplot with custom bin_size
#        bin_s = list(np.arange(-50, 50)/10)
#        fig = ff.create_distplot(hist_data, group_labels)#, bin_size = bin_s)
#
#        # Plot!
#        # Adjust title, legend
#        plt.interactive(False)
#        return plotly.offline.plot(fig, filename='Distplot with Multiple Bin Sizes')




##### New class that includes SimData class by initizilaizing it internally and 
##### only displays a few simple functions to user


class UserInterface:
    '''
    Class to wrap up all functionalities and give user just the functions that are 
    necessary to create the wanted variables y, X, and treatment
    '''
    def __init__(self, N, k, seed = None, skewed_covariates = False, categorical_covariates = None):
        '''
        Input:  N, Int with number of observations
                k, Int with number of covariates 
        
        Initilizes UserInterface class with number of observations N and number of covariates k.
        Generates Nxk matrix "X" with values for each covariate for all observations and saves 
        it in object s
        Additional options for covariates X:
            - Skewness
            - Categories
        '''

        self.backend = SimData(N, k, seed)
        self.backend.generate_covariates(skew = skewed_covariates, categorical_covariates = categorical_covariates)
        #self.backend.plot_covariates()
        
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
        Input:  random_assignment, Boolean to indicate if treatment assignment should be random 
                or dependent on covariates
                
                constant, Boolean allow for constant treatment effect 
                
                heterogeneous, Boolean allow for heterogeneous treatment effects that 
                depend on covariates
                
                negetive, Boolean allow treatment effects to be negetive, drawn from U[-1,0]
                
                no_treatment, Boolean allow treatment effect to be zero when assigned
                
                treatment_option_weights, List-like object of length 4, with corresponding weights 
                summing up to 1 [constant, heterogeneous, negative, no_treatment], e.g. [0, 0.5, 0.1, 0.4]
                
        
        Generates treatment assignment array "D" and treatment effect array "treatment_effect"
        ans saves them as self. internal variables in s 
        
        return: None
        '''
        self.backend.generate_treatment_assignment(random_assignment, assignment_prob)
        self.backend.generate_treatment_effect(treatment_option_weights, constant_pos,
                                               constant_neg, heterogeneous_pos, 
                                               heterogeneous_neg, no_treatment, 
                                               discrete_heterogeneous, intensity)

        return None

    def output_data(self, binary = False):
        '''
        Generates output array "y" the following way: Y = Theta_0 * D + g_0(X) + U,
        where Theta_O is the treatment effect of each observation, D the dummy vector
        for assigning treatment, g_0() the non_linear transformation function, and U
        a normal-distributed noise-/error term
        
        return: y, X, treatment_effect
         '''                
        return self.backend.generate_outcome_variable(binary)
    
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
        return self.backend.treatment_effect_type
        
    def set_weights_treatment_assignment(self, new_weight_vector):
        if len(new_weight_vector) is not self.backend.get_k():
            raise ValueError('New weight vector needs to be of dimension k')
            
        self.backend.weights_treatment_assignment = np.array(new_weight_vector)
            
    def set_weights_covariates_to_outputs(self, new_weight_vector):
        if len(new_weight_vector) is not self.backend.get_k():
            raise ValueError('New weight vector needs to be of dimension k')
        
        self.backend.weights_covariates_to_outputs= np.array(new_weight_vector)
        
    def set_subset_z_size(self, new_size):
        if new_size < 1 or new_size > self.backend.get_k():
            raise ValueError('Size of subset Z needs to be within [1,k]')
            
        self.backend.z_set_size = new_size
        
    def __str__(self):
        return "N = " + str(self.backend.get_N()) + ", k = " + str(self.backend.get_k())






