from SimulateData import SimData

if __name__ == "__main__":
    s = SimData()
    X = s.generate_covariates()
    test = s.generate_treatment_assignment(X, bernoulli = True)
    print(test)
