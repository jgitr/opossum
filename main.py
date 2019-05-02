from SimulateData import SimData

if __name__ == "__main__":
    s = SimData(100,10)
    s.generate_covariates()
    s.generate_treatment_assignment(False) # returns treatment assigment vector [0,1,...]
    s.generate_treatment_effect()
    print(s)



