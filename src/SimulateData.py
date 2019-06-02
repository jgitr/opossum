
from scipy import random, stats
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly
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
#???        maybe standardize as well? Right now not necessary since all values are [0,1] anyway
            realized_treatment_effect = self.generate_realized_treatment_effect() # needs to be a probability
            
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
        A = A * overlay_matrix

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
#            b = 1/np.arange(1,self.k+1) # diminishing weight vector
            b = np.random.uniform(0,1,self.k) # random weight vector drawn from U[0,1]
            self.g_0_X = np.cos(np.dot(X,b))**2  # overwrite with nonlinear covariates
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

        #weight_vector = 1/np.arange(1,self.k+1)             # diminishing weights
        weight_vector_alt = np.random.uniform(0,1,self.k)   # random weights from U[0,1]
        
        # random treatment assignment
        if random:
            m_0 = assignment_prob  # probability
            
            # propensity scores for each observation 
            self.propensity_score = np.repeat(m_0,self.N)
        else:
            a = np.dot(self.X, weight_vector_alt)    # X*weights -> a (Nx1 vector)

            # Using empirical mean, sd
            a_mean = np.mean(a)
            a_sigma = np.std(a)
            z = (a - a_mean) / a_sigma          # normalizing 'a' vector
            
            # using normalized vector z to get probabilities from normal pdf
            # to later assign treatment with binomial in D
            m_0 = stats.norm.cdf(z)
            
            # propensity scores for each observation 
            self.propensity_score = m_0

        # creating array out of binomial distribution that assigns treatment according to probability m_0
        self.D = np.random.binomial(1, m_0, self.N)
        self.weight_vector = weight_vector_alt
        
        return None

##??? Removed plotting function from covariates function and created extra plotting function
## BUT: Seems useless right now, either create useful plot or remove function
    def plot_covariates(self):
        '''

        '''
        plt.interactive(False)
        plt.hist(self.X, bins=10)
        plt.ylabel('Test')
        plt.show(block=True)

    def visualize_correlation(self):
        """
        Generates Correlation Matrix of the Covariates
        :return:
        """

        df = pd.DataFrame(self.X)  
        
        corr = df.corr()
        corr.style.background_gradient(cmap='coolwarm') # requires HTML backend
        sns.heatmap(corr, annot = True)
        plt.show()
        return None


    def generate_treatment_effect(self, treatment_option_weights, constant, heterogeneity,
                                  negative, no_treatment):

        """
        options Theta(X), where X are covariates:
        –No treatment effect(for all or for some people).
        –Constant ( for all or for some people).
        –heterogeneity (discrete and continuous).
        –Even negative values seem realistic ( for some people).

        -predefined_idx:
        Instead of randomly assigning (drawing from a uniform distribution) the k covariates to an option,
        the user can choose a predefined index set upon which he whishes to apply the options.
        length = n
        Must be array-like type

        if not option in ['no treatment', 'constant', 'heterogeneity', 'negative']:
            raise ValueError('Wrong Options')

        :return: Vector Theta, length self.k (covariates), theta_0
        """

        # ratio list: [constant, heterogeneity, negative, no_treatment] (treatment_option_weights)
        # e.g. [0, 0.5, 0.1, 0.4], needs to sum up to 1 
        
        if treatment_option_weights is not None:            
            # make sure it's a numpy array
            treatment_option_weights = np.array(treatment_option_weights)
            if np.sum(treatment_option_weights) !=1:
                raise ValueError('Values in treatment_option_weights-vector must sum up to 1')
            if len(treatment_option_weights) !=4:
                raise ValueError('Treatment_option_weights-vector must be of length 4')
            
            
            # take times N to get absolute number of each option
            absolute_ratio = (self.N*treatment_option_weights).astype(int)
            
            # adjusting possible rounding errors by increasing highest value 
            if sum(absolute_ratio) < self.N:
                index_max = np.argmax(treatment_option_weights)
                absolute_ratio[index_max] = absolute_ratio[index_max] + (self.N-sum(absolute_ratio))
            
            # fill up index-array with options 1-4 according to the weights
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
            
            constant, heterogeneity, negative, no_treatment = tuple(options_boolean)
            
        # Process options
        options = []
        if constant:
            options.append(1)
        if heterogeneity:
            options.append(2)
        if negative:
            options.append(3)
        if no_treatment:
            options.append(4)
        
        if treatment_option_weights is None:
            if options ==[]:
                raise ValueError("At least one treatment effect option must be True")
            # assigning which individual gets which kind of treatment effect 
            # from options 1-4
            n_idx = np.random.choice(options, self.N, True)
            
        # array to fill up with theta values         
        theta_combined = np.zeros(self.N)
        
        if constant:
            # Option 1
            # Rules: Theta_0 = constant  (c) with c = 0.2
            # constant is independent from covariates 
            con = 0.2 #  constant value for treatment effect
            
            theta_combined[n_idx == 1] = con


        if heterogeneity:
            # Option 2
            # Rules:
            # (1) Apply trigonometric function
            # (2) Standardize the treatment effet within the interval [0.1, 0.3].
            # theta_0 is to be at most 30% of the baseline outcome g_0(X)
            
            # assigning randomly which covariates affect treatment effect
            r_idx = np.random.choice(options, size = self.k, replace = True)
            
            #(1) Trigonometric
            X_option2 = self.X[:,r_idx == 2].copy()
            
            w = np.random.normal(0,0.25,self.N)
            # Need to adjust weight_vector such that it complies with the alternated k (dimension)
            weight_vector_adj = self.weight_vector[r_idx == 2]
            
            gamma = np.sin(np.dot(X_option2, weight_vector_adj)) + w  # one gamma for each observation in n

            # (2) Standardize
            theta_option2 = standardize(gamma)
            
            theta_combined[n_idx == 2] = theta_option2[n_idx == 2]
            
        # negative effects between [-0.3,0]    
        if negative:
            theta_combined[n_idx == 3] = np.random.uniform(-0.3, 0, np.sum(n_idx == 3)) 
            
        if no_treatment:
            theta_combined[n_idx == 4] = 0 # not really necessary since vector was full of 0 
        
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
    def visualize_distribution(self, y, treatment):
        """
        input: outcome variable y_treated, y_not_treated, treatment
        :return: Depict
        # Add histogram data
    
        x1 = y[self.D == 0]
        x2 = y[self.D == 1]
        #x2 = treatment
        s Output Distribution
        """

        # Group data together
        hist_data = [x1, x2]

        group_labels = ['No Treatment Assignment', 'Treatment Assigment']

        # Create distplot with custom bin_size
        bin_s = list(np.arange(-50, 50)/10)
        fig = ff.create_distplot(hist_data, group_labels)#, bin_size = bin_s)

        # Plot!
        # Adjust title, legend
        plt.interactive(False)
        return plotly.offline.plot(fig, filename='Distplot with Multiple Bin Sizes')




##### New class that includes SimData class by initizilaizing it internally and 
##### only displays a few simple functions to user

class UserInterface:
    '''
    Class to wrap up all functionalities and give user just the functions that are 
    necessary to create the wanted variables y, X, and treatment
    '''
    def __init__(self, N, k, seed = None, skewed_covariates = True):
        '''
        Input:  N, Int with number of observations
                k, Int with number of covariates 
        
        Initilizes UserInterface class with number of observations N and number of covariates k.
        Generates Nxk matrix "X" with values for each covariate for all observations and saves 
        it in object s
        '''
        self.s = SimData(N, k, seed)
        self.s.generate_covariates(skew = skewed_covariates)
        print('plotting skewed covariates!')
        self.s.plot_covariates()
        
    def generate_treatment(self, random_assignment = True, assignment_prob = 0.5, constant = True, heterogeneous = False,
                                  negative = False, no_treatment = False, treatment_option_weights = None):
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
        self.s.generate_treatment_assignment(random_assignment, assignment_prob)
        self.s.generate_treatment_effect(treatment_option_weights, constant, heterogeneous, 
                                         negative, no_treatment)

        return None

    def output_data(self, binary = False):
        '''
        Generates output array "y" the following way: Y = Theta_0 * D + g_0(X) + U,
        where Theta_O is the treatment effect of each observation, D the dummy vector
        for assigning treatment, g_0() the non_linear transformation function, and U
        a normal-distributed noise-/error term
        
        return: y, X, treatment_effect
         '''                
        return self.s.generate_outcome_variable(binary)
    
    def plot_covariates_correlation(self):
        '''
        Shows a correlation heatmap of the covariates 
        '''
        self.s.visualize_correlation()
        return None

    def plot_distribution(self, y, treatment):
        """

        :return:
        """

        self.s.visualize_distribution(y, treatment)

# Of the following goals, discrete heterogeneity is still missing
# – No treatment effect (for all or for some people).
# – Constant (for all or for some people).
# – heterogeneity (discrete and continuous).
# – Even negative values seem realistic (for some people).
        
##### Also: It is not possible yet to get a heterogenous treatment effect for all
    # assigned individuals that just dependes on some covariates
    # Either Some heterogenous effects and some others (non, constant, negative) and just 
    # dependence on some covariates or
    # All heterogenous effects and depending on all covariates








