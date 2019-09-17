
# Opossum 
Package for simulation of data generating process to evaluate causal inference models.

## Getting Started
Latest release version: 0.2.1

### Download with pip

#### *nix OS
`pip3 install opossum`

#### Windows
`pip install opossum --user --upgrade`

### Build it yourself
`git clone https://github.com/jgitr/opossum.git`  
`cd opossum`  
`git checkout master`  
`cd opossum`  

#### *nix
`python3 main.py`

#### Windows
`python main.py`

#### SHA-256: opossum-0.2.0-py3-none-any.whl 
`19b0b705b37a71fd5deac40720d30da4caa6529a9f1e1e7db648ecd07aea3077`

#### SHA-256: opossum-0.2.0.tar.gz 
`e13000b1755576a80693dca5f2560cc9b7d3e1391701bdc5cae86f3a9b7eb036`

## Application
Bellow you can find a short description of the core functions that the package offers in code form. For more detailed information on how to apply the package and to get insight into the theoretical model that it is based on, please refer to the following blog post https://humboldt-wi.github.io/blog/research/applied_predictive_modeling_19/data_generating_process_blogpost/.

### Default Setting


```python
from opossum import UserInterface
# number of observations N and number of covariates k
N = 10000
k = 50
# initilizing class
u = UserInterface(N, k, seed=None, categorical_covariates = None)
# assign treatment and generate treatment effect inside of class object
u.generate_treatment(random_assignment = True, 
                     assignment_prob = 0.5, 
                     constant_pos = True, 
                     constant_neg = False,
                     heterogeneous_pos = False, 
                     heterogeneous_neg = False, 
                     no_treatment = False, 
                     discrete_heterogeneous = False,
                     treatment_option_weights = None, 
                     intensity = 5)
# generate output variable y and return all 4 variables
y, X, assignment, treatment = u.output_data(binary=False, x_y_relation = 'partial_nonlinear_simple')
```

### Choosing covariates


```python
N = 1000
k = 20
# whole dataset is binary
u = UserInterface(N, k, categorical_covariates = 2)
# one quarter of the dataset is binary
u = UserInterface(N, k, categorical_covariates = [5,2])
# dataset consists of 10 continuous, 4 binary, and 3 variables each with 3 and 5 categories respectively 
u = UserInterface(N, k, categorical_covariates = [10,[2,3,5])
```

### Creating treatment effects


```python
# random treatment assignment resulting in on average 20% treated observations 
u.generate_treatment(random_assignment = True, assignment_prob = 0.2)
# non-random treatment assignment with on average 65% treated observations
u.generate_treatment(random_assignment = False, assignment_prob = 'high')
# generating only a positive heterogeneous treatment effect
u.generate_treatment(constant_pos = False, constant_neg = False, heterogeneous_pos = True, heterogeneous_neg = False, 
                     no_treatment = False, discrete_heterogeneous = False)
# generating a heterogeneous treatment effect that is in 20% of cases negative and 80% positive
u.generate_treatment(treatment_option_weights = [0, 0, 0.8, 0.2, 0, 0]) 
```

### Creating output


```python
# Creating continuous y with partial nonlinear relation 
y, X, assignment, treatment = u.output_data(binary=False, x_y_relation = 'partial_nonlinear_simple')
# Creating binary y with underlying linear relation and added interaction terms of X
y, X, assignment, treatment = u.output_data(binary=True, x_y_relation = 'linear_interaction')
```
