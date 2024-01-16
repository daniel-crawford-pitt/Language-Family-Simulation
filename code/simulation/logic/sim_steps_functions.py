import numpy as np

from sim_utils import *


def weighted_max(curr_val, adj_vals):            
    return max(
        min(1.0, 1.1* np.floor(np.divide(curr_val,1.0))), 
        min(1.0,0.9*safe_max(adj_vals))
    )


def adj_step(curr_val, adj_vals):
    return safe_max(adj_vals)

def rand_adj_step(curr_val, adj_vals):
    return safe_max((np.multiply(
        np.random.randint(2, size = adj_vals.shape),
        adj_vals
    )))

def weighted_rand_adj_step(curr_val, adj_vals):
    return safe_max((np.multiply(
        np.random.rand(adj_vals.shape[0],adj_vals.shape[1]),
        adj_vals
    )))
    
def prop_weighted_rand_adj_step(curr_val, adj_vals):
    return safe_max((np.multiply(
        np.random.rand(adj_vals.shape[0],adj_vals.shape[1]),
        adj_vals
    )))