from SimulateData import SimData

if __name__ == "__main__":
    s = SimData()
    X = s.generate_covariates()
    D, weight_vector, prob_mean = s.generate_treatment_assignment(X, bernoulli = False) # returns treatment assigment vector [0,1,...]
    print(D)
    print(s.generate_treatment_effect(X, weight_vector))


np.sum(D)
