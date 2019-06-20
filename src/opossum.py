from scipy import random, stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from helpers import standardize, is_pos_def

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

    def generate_covariates(self, nonlinear = True, skew = True):

        """
        Model-wise: g_0(X)

        Algorithm for Covariates

        1) Generate a random positive definite covariance matrix Sigma
        based on a uniform distribution over the space k*k of the correlation matrix

        2) Generate random normal distributed variables X_{N*k} with mean = 0 and variance = sigma

        Create low- and high-dimensional dataset.
         Correlation between the covariates isrealistic.
         options:
            – continuous with some known distribution (e.g.  Normal()).
            – discrete (dummy and multilevel).
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

        self.X = X

        if nonlinear:
            # transforming by cos(X*weights)²
            self.g_0_X = np.cos(np.dot(X,self.weights_covariates_to_outputs))**3 + 0.2*np.dot(X,self.weights_covariates_to_outputs)
        else:
            # If not nonlinear, then g_0(X) is just the identity 
            self.g_0_X = np.dot(X,np.repeat(1/self.k,self.k))  # dim(X) = n * k -> need to vector multiply with vector shaped [k,1]

        return None


    def generate_treatment_assignment(self, random, assignment_prob):
        
        """
        Treatment assignment
        binary and multilevel (discrete).The generation should be
        –random (randomized control trial) with possible imbalanced assignment (e.g.  75% treatedand 25% control.

        :return:
        """       
        # random treatment assignment
        if random:
            m_0 = assignment_prob  # probability
            
            # propensity scores for each observation 
            self.propensity_scores = np.repeat(m_0,self.N)
        else:
            a = np.dot(self.X, self.weights_treatment_assignment)    # X*weights -> a (Nx1 vector)

            # Using empirical mean, sd
            a_mean = np.mean(a)
            a_sigma = np.std(a)
            z = (a - a_mean) / a_sigma          # normalizing 'a' vector

            # using normalized vector z to get probabilities from normal pdf
            # to later assign treatment with binomial in D
            m_0 = stats.norm.cdf(z)
            
            # propensity scores for each observation 
            self.propensity_scores = m_0

        # creating array out of binomial distribution that assigns treatment according to probability m_0
        self.D = np.random.binomial(1, m_0, self.N)

        
        return None

##??? Removed plotting function from covariates function and created extra plotting function
## BUT: Seems useless right now, either create useful plot or remove function
    def plot_covariates(self):
        '''

        '''
#        plt.interactive(False)
        plt.hist(self.X, bins=10)
        plt.ylabel('Test')
        plt.show()#block=True)

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
                raise ValueError('Treatment_option_weights-vector must be of length 5')
            
            
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
            
            # old assigning of Z (depended on number of choosen treatment effects)
#            r_idx = np.random.choice(options, size = self.k, replace = True)
#            
#            #(1) Trigonometric
#            X_option2 = self.X[:,(r_idx == 3) | (r_idx == 4)].copy()
#            
#            w = np.random.normal(0,0.25,self.N)
#            # Need to adjust weights dimension such that it complies with the alternated k 
#            weight_vector_adj = self.weights_treatment_assignment[(r_idx == 3) | (r_idx == 4)]
#            
#            gamma = np.sin(np.dot(X_option2, weight_vector_adj)) + w  # one gamma for each observation in n

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
    def __init__(self, N, k, seed = None, skewed_covariates = False):
        '''
        Input:  N, Int with number of observations
                k, Int with number of covariates 
        
        Initilizes UserInterface class with number of observations N and number of covariates k.
        Generates Nxk matrix "X" with values for each covariate for all observations and saves 
        it in object s
        '''

        self.backend = SimData(N, k, seed)
        self.backend.generate_covariates(skew = skewed_covariates)
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






