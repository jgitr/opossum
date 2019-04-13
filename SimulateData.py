
from scipy import random, linalg
import numpy as np

class SimulateData:
    """
    Model: Refer to Paper in Readme

    """

    def __init__(self):
        random.seed(10) # For debugging

    def generate_outcome_variable(self):
        """
        options: binary, multilevel(discrete), continuous



        """

    def generate_covariates(self, Sigma_dimension = 10, N = 10):
        """
        Algorithm for Covariates

        1) Generate a random positive definite covariance matrix Sigma
        based on a uniform distribution over the space k*k of the correlation matrix

        2) Scale the covariance matrix. This equals the correlation matirx and can be seen as
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
        A = random.rand(Sigma_dimension, Sigma_dimension) # drawn from uniform distribution in [0,1]
        Sigma = np.dot(A,A.transpose()) # a matrix multiplied with its transposed is aaaalways positive definite

        # 2)
        # Correlation Matrix P = Sigma * (1/sd)
        sd = random.rand(1)
        P = Sigma * (1/sd)

        # 3)
        mu = [0] * N
        X = np.random.multivariate_normal(mu, Sigma, N)

        return X

        def generate_treatment_effect():
            """
            options:
            –No treatment effect(for all or for some people).
            –Constant ( for all or for some people).
            –heterogeneity (discrete and continuous).
            –Even negative values seem realistic ( for some people).

            :return:
            """
