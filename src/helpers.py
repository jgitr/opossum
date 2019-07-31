import numpy as np

def standardize(gamma, upper_val = 0.3, lower_val = 0.1):

    s = (gamma-np.min(gamma))/(np.max(gamma)-np.min(gamma))
    out = s * (upper_val - lower_val) + lower_val

    return out

def is_pos_def(x):
            return np.all(np.linalg.eigvals(x) > 0)


def adjusting_assignment_level(level):
    
    adjustment_dict = {'low' : -0.545, 'medium' : 0, 'high' : 0.545}
    
    return adjustment_dict[level]

def revert_string_prob(string):
    
    reverse_dict = {'low': 0.35, 'medium': 0.5, 'high': 0.65} 
    
    return reverse_dict[string]

# functions describing relation between X and y (g_0(X))
def linear_simple(self):
    return np.dot(self.X,self.weights_covariates_to_outputs)

def linear_interaction(self): 
    return np.dot(self.X, self.weights_covariates_to_outputs) \
            + np.dot(self.X_interaction, self.weights_interaction)

def partial_nonlinear_simple(self):
    return 2.5*np.cos(np.dot(self.X,self.weights_covariates_to_outputs))**3 \
            + 2.5*0.2*np.dot(self.X,self.weights_covariates_to_outputs)

def partial_nonlinear_interaction(self):
    return 2.5*np.cos(np.dot(self.X,self.weights_covariates_to_outputs) + \
                  np.dot(self.X_interaction, self.weights_interaction))**3 \
                  + 2.5*0.2*(np.dot(self.X,self.weights_covariates_to_outputs) 
                  + np.dot(self.X_interaction, self.weights_interaction))

def nonlinear_simple(self):
    return 3*np.cos(np.dot(self.X,self.weights_covariates_to_outputs))**3

def nonlinear_interaction(self):
    return 3*np.cos(np.dot(self.X,self.weights_covariates_to_outputs) + 
                  np.dot(self.X_interaction, self.weights_interaction))**3 

relation_dict = {'linear_simple': linear_simple,
                 'linear_interaction': linear_interaction,
                 'partial_nonlinear_simple': partial_nonlinear_simple,
                 'partial_nonlinear_interaction': partial_nonlinear_interaction,
                 'nonlinear_simple': nonlinear_simple,
                 'nonlinear_interaction': nonlinear_interaction}
# function to call in main class 
def relation_fct(self, x_y_relation):
    function = relation_dict.get(x_y_relation)
    return function(self)
