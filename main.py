from SimulateData import SimData

if __name__ == "__main__":
    s = SimData()
    X = s.generate_covariates()
    D = s.generate_treatment_assignment(X, bernoulli = True) # returns treatment assigment vector [0,1,...]
    print(D)
    s.generate_treatment_effect(X, option = ['bla'])
