import numpy as np

def combine_ll_to_rbg(ll):
    np.stack(np.array([l.map for l in ll]), axis = 2)


