
from scipy import random, linalg
import numpy as np
import matplotlib.pyplot as plt
from helpers import sequence

class SimData:
    """
    Model: Refer to Paper in Readme

    """

    def __init__(self):
        random.seed(10) # For debugging
        self.Sigma_dimension = 10
        self.N = 10

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
        A = random.rand(self.Sigma_dimension, self.Sigma_dimension) # drawn from uniform distribution in [0,1]
        Sigma = np.dot(A,A.transpose()) # a matrix multiplied with its transposed is aaaalways positive definite

        # 2)
        # Correlation Matrix P = Sigma * (1/sd)
        sd = random.rand(1)
        P = Sigma * (1/sd)

        # 3)
        mu = [0] * self.N
        X = np.random.multivariate_normal(mu, Sigma, self.N)

        if nonlinear:
            b = [1/len(X)] * len(X)  # weight vector, per default uniform # or self.Sigma_dimension instead of len
            X = cos(X * b)  # overwrite with nonlinear covariates

        if plot:
            plt.interactive(False)
            plt.hist(X, bins=10)
            plt.ylabel('Test')
            plt.show(block=True)

        return X

        def generate_treatment_assignment(X, bernoulli = True):

            """
            Treatment assignment
            binary and multilevel (discrete).The generation should be
            –random (randomized control trial) with possible imbalanced assignment (e.g.  75% treatedand 25% control.

            :return:
            """

            if bernoulli:
                m_0 = 0.5  # probability
                D  # draw from bernoulli
                np.random.binomial(1, m_0, self.N) # depict as bernoulli


            else:
                print('bla')
                s = sequence(self.N)
                s_scaled = [1/ele for ele in s]
                a = X * s_scaled

                # Cheating expectation here! clarify!
                a_mean = numpy.mean(a)
                a_sigma = numpy.std(a)
                z = (a - a_mean) / a_sigma
                m_0 = random.normal(z)  #  Phi() # normal distribution - need expectation of a here!

                D = ... Output

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
