from SimulateData import SimData
import numpy as np # necessary if predefined_idx is used in s.generate_treatment_effect

if __name__ == "__main__":
    s = SimData(100,10)
    s.generate_covariates()
    s.generate_treatment_assignment(False) # returns treatment assigment vector [0,1,...]
    s.generate_treatment_effect()
    realized_treatment_effect = s.generate_realized_treatment_effect()
    a = s.visualize_correlation()
    print(s, a)


