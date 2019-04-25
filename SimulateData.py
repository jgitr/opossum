
from scipy import random, linalg
import numpy as np
import matplotlib.pyplot as plt
from helpers import sequence
from math import cos

class SimData:
    """
    Model: Refer to Paper in Readme

    """

    def __init__(self):
        random.seed(10) # For debugging
        self.N = 10
        self.k = 10 # Sigma Dimension
        self.p = 0.5

    def generate_outcome_variable(self):
        """
        options: binary, multilevel(discrete), continuous



        """

    def generate_covariates(self, plot = True, nonlinear = True):

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
            

        else:
            print('bla')
            s = sequence(self.N)
            s_scaled = [1/ele for ele in s]
            a = X * s_scaled

            # Cheating expectation here! clarify!
            a_mean = numpy.mean(a)
            a_sigma = numpy.std(a)
            z = (a - a_mean) / a_sigma
            m_0 = random.multivariate_normal(z)  #  Phi() # normal distribution - need expectation of a here!

        # m_0 exists
        D = np.random.binomial(1, m_0, self.N)

        return D

    def generate_treatment_effect():
        """
        options:
        –No treatment effect(for all or for some people).
        –Constant ( for all or for some people).
        –heterogeneity (discrete and continuous).
        –Even negative values seem realistic ( for some people).

        :return:
        """
