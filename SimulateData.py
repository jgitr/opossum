
from scipy import random, linalg
import numpy as np
import matplotlib.pyplot as plt
from helpers import sequence
from helpers import standardize
from math import cos

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

    def __init__(self):
        random.seed(10) # For debugging
        self.N = 10 # Natural, number of observations
        self.k = 10 # Natural, number of covariates
        self.p = 0.5

    def generate_outcome_variable(self):
        """
        options: binary, multilevel(discrete), continuous



        """

    def generate_covariates(self, plot = False, nonlinear = True):

        """
        Algorithm for Covariates

        1) Generate a random positive definite covariance matrix Sigma
        based on a uniform distribution over the space k*k of the correlation matrix

        2) Scale the covariance matrix. This equals the correlation matrix and can be seen as
        the covariance matrix of the standardized random variables sigma = x / sd(x)

        3) Generate random normal distributed variables X_{N*k} with mean = 0 and variance = sigma

        Create low- and high-dimensional dataset.
         Correlation between the covariates isrealistic.
         options:
            – continuous with some known distribution (e.g.  Normal()).
            – discrete (dummy and multilevel).
        """

        # 1)
        # Sigma
        A = random.rand(self.k, self.k) # drawn from uniform distribution in [0,1]
        Sigma = np.dot(A,A.transpose()) # a matrix multiplied with its transposed is aaaalways positive definite

        # 2)
        # Correlation Matrix P = Sigma * (1/sd)
        sd = 1  #  Frage an Daniel: Random, Intervall von 0 bis 1 oder was?
        p = Sigma * (1/sd)  # not used yet!

        # 3)
        mu = np.array([self.p] * self.k)
        X = np.random.multivariate_normal(mu, Sigma, self.N)

        if nonlinear:
            b = np.array([1/len(X)] * len(X))  #   # weight vector, per default uniform # or self.Sigma_dimension instead of len
            X = np.cos(X * b)  # overwrite with nonlinear covariates

        if plot:
            plt.interactive(False)
            plt.hist(X, bins=10)
            plt.ylabel('Test')
            plt.show(block=True)

        return X

    def generate_treatment_assignment(self, X, bernoulli = True):

        """
        Treatment assignment
        binary and multilevel (discrete).The generation should be
        –random (randomized control trial) with possible imbalanced assignment (e.g.  75% treatedand 25% control.

        :return:
        """

        if bernoulli:
            m_0 = 0.5  # probability

        # Remains to be tested
        else:
            s = sequence(self.N)
            weight_vector = [1/ele for ele in s] # can adjust the weight vector such that the weights dont necessarily
            # diminish, perhaps RVs
            a = X * weight_vector

            # Using empirical mean, sd
            a_mean = np.mean(a)
            a_sigma = np.std(a)
            z = (a - a_mean) / a_sigma
            m_0 = random.multivariate_normal(z)  #  Phi() # normal distribution - need expectation of a here!

        # m_0 exists
        D = np.random.binomial(1, m_0, self.N)

        return D, weight_vector

    def generate_treatment_effect(self, X, weight_vector,no_treament = False, constant = True, heterogeneity = False, negative = False):
        """
        options Theta(X), where X are covariates:
        –No treatment effect(for all or for some people).
        –Constant ( for all or for some people).
        –heterogeneity (discrete and continuous).
        –Even negative values seem realistic ( for some people).

        if not option in ['no treatment', 'constant', 'heterogeneity', 'negative']:
            raise ValueError('Wrong Options')

        :return: Vector Theta, length self.k (covariates)
        """

        # Randomly assign single covariates to options
        #covariate_idx = list(range(1, self.k + 1))
        r_idx = np.random.choice(a = [1, 2, 3, 4], size = self.k, replace = False) # option 1-4 assigned to cov.

        # To fill
        theta_0 = np.zeros(self.k)

        # Assign option 1 to columns in covariates (dim(cov) = n * k)
        # Rules for option 1: Theta_0 = constant  (c) with c = 0.2
        # X[:, r_idx == 1]
        con = 0.2 #  constat value for treatment effect
        theta_option1[r_idx == 1] = con

        # Option 2
        # Rules:
        # (1) Apply trigonometric function
        # (2) Standardize the treatment effet within the interval [0.1, 0.3].
        # theta_0 is to be at most 30% of the baseline outcome g_0(X)

        #(1)
        X_option1 = X[:, r_idx == 1]
        w = np.random.multivariate_normal(0, 0.25, np.shape(X_option1[0]))  # len(X_option1) should be length of rows of X
        gamma = sin(np.dot(X_option1, weight_vector)) + w

        # (2)
        theta_option2 = standardize(gamma)

        








