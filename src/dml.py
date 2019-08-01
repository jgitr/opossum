import numpy as np
from sklearn.datasets import make_spd_matrix
import math
import statsmodels.api as sm # for OLS
from sklearn.ensemble import RandomForestRegressor # Our ML algorithm
import seaborn as sns
import matplotlib.pyplot as plt
from opossum import UserInterface

class double_ml:

    def __init__(self, Y, X, M, N):
        print(Y, X, M, N)
    
    def run(YY, # out
                  XX, # in
                  DD, # assigment
                  N,  # observations 
                  MC_no):


        # The input doesnt vary yet
        # First three params are lists 

        # Source
        # http://aeturrell.com/2018/02/10/econometrics-in-python-partI-ML/

        theta_est = np.zeros(shape=[MC_no,3])  # Array of estimated thetas to store results

        for i in range(MC_no):
            
            Y = YY[i]
            X = XX[i]
            D = DD[i]
            
            u = UserInterface(1000, 50, seed=None, categorical_covariates = None)
            u.generate_treatment(random_assignment=False, 
                                 treatment_option_weights=[1,0,0,0,0,0])
            Y, X, D, treatment = u.output_data()
#            avg_treat = np.mean(treatment[D==1])
            # Now run the different methods
            #
            # OLS --------------------------------------------------
            OLS = sm.OLS(Y,D)
            results = OLS.fit()
            theta_est[i][0] = results.params[0]

            # Naive double machine Learning ------------------------
            naiveDMLg =RandomForestRegressor(max_depth=2)
            # Compute ghat
            naiveDMLg.fit(X,Y)
            Ghat = naiveDMLg.predict(X)
            naiveDMLm =RandomForestRegressor(max_depth=2)
            naiveDMLm.fit(X,D)
            Mhat = naiveDMLm.predict(X)
            # vhat as residual
            Vhat = D-Mhat
            theta_est[i][1] = np.mean(np.dot(Vhat,Y-Ghat))/np.mean(np.dot(Vhat,D))

            # Cross-fitting DML -----------------------------------
            # Split the sample
            I = np.random.choice(N,np.int(N/2),replace=False)
            I_C = [x for x in np.arange(N) if x not in I]
            # Ghat for both
            Ghat_1 = RandomForestRegressor(max_depth=2).fit(X[I],Y[I]).predict(X[I_C])
            Ghat_2 = RandomForestRegressor(max_depth=2).fit(X[I_C],Y[I_C]).predict(X[I])
            # Mhat and vhat for both
            Mhat_1 = RandomForestRegressor(max_depth=2).fit(X[I],D[I]).predict(X[I_C])
            Mhat_2 = RandomForestRegressor(max_depth=2).fit(X[I_C],D[I_C]).predict(X[I])
            Vhat_1 = D[I_C]-Mhat_1
            Vhat_2 = D[I] - Mhat_2
            theta_1 = np.mean(np.dot(Vhat_1,(Y[I_C]-Ghat_1)))/np.mean(np.dot(Vhat_1,D[I_C]))
            theta_2 = np.mean(np.dot(Vhat_2,(Y[I]-Ghat_2)))/np.mean(np.dot(Vhat_2,D[I]))
            theta_est[i][2] = 0.5*(theta_1+theta_2)


#        theta_ols       = list(map(lambda x: x[0], theta_est))
#        theta_naive_dml = list(map(lambda x: x[1], theta_est))
#        theta_cross_dml = list(map(lambda x: x[2], theta_est))
#                
#        sns.distplot(theta_ols, bins = MC_no)
#        sns.distplot(theta_naive_dml, bins = MC_no)
#        sns.distplot(theta_cross_dml, bins = MC_no)
#        
#        plt.savefig('dml_estimator_distribution.png')

        # structure:
        # ols, naive double ml, cross-fitting double ml estimated coefficients
        return theta_est




