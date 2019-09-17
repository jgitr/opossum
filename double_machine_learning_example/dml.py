import numpy as np
import statsmodels.api as sm # for OLS
from sklearn.ensemble import RandomForestRegressor # Our ML algorithm
    
def dml_single_run(Y, X, D):
    # Source
    # http://aeturrell.com/2018/02/10/econometrics-in-python-partI-ML/
    
    # array to store OLS, naive and cross result
    theta_est = np.zeros(3)  
    
    N = len(Y)
    num_trees = 50
    # Now run the different methods
    #
    # OLS --------------------------------------------------
    OLS = sm.OLS(Y,D)
    results = OLS.fit()
    theta_est[0] = results.params[0]

    # Naive double machine Learning ------------------------
    naiveDMLg =RandomForestRegressor(num_trees , max_depth=2)
    # Compute ghat
    naiveDMLg.fit(X,Y)
    Ghat = naiveDMLg.predict(X)
    naiveDMLm =RandomForestRegressor(num_trees , max_depth=2)
    naiveDMLm.fit(X,D)
    Mhat = naiveDMLm.predict(X)
    # vhat as residual
    Vhat = D-Mhat
    theta_est[1] = np.mean(np.dot(Vhat,Y-Ghat))/np.mean(np.dot(Vhat,D))

    # Cross-fitting DML -----------------------------------
    # Split the sample
    N_array = np.arange(N)
    np.random.shuffle(N_array)
    I = N_array[:int(N/2)]
    I_C = N_array[int(N/2):]
    # Ghat for both
    Ghat_1 = RandomForestRegressor(num_trees , max_depth=2).fit(X[I],Y[I]).predict(X[I_C])
    Ghat_2 = RandomForestRegressor(num_trees , max_depth=2).fit(X[I_C],Y[I_C]).predict(X[I])
    # Mhat and vhat for both
    Mhat_1 = RandomForestRegressor(num_trees , max_depth=2).fit(X[I],D[I]).predict(X[I_C])
    Mhat_2 = RandomForestRegressor(num_trees , max_depth=2).fit(X[I_C],D[I_C]).predict(X[I])
    Vhat_1 = D[I_C]-Mhat_1
    Vhat_2 = D[I] - Mhat_2
    theta_1 = np.mean(np.dot(Vhat_1,(Y[I_C]-Ghat_1)))/np.mean(np.dot(Vhat_1,D[I_C]))
    theta_2 = np.mean(np.dot(Vhat_2,(Y[I]-Ghat_2)))/np.mean(np.dot(Vhat_2,D[I]))
    theta_est[2] = 0.5*(theta_1+theta_2)

    return theta_est




