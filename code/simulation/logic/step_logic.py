

import numpy as np

def step_boolean(): return True


def step(loc):
    x = loc[0]
    y = loc[1]


    return np.array([
        (x-1,y+1),(x,y+1),(x+1,y+1),
        (x-1,y),(x,y),(x+1,y),
        (x-1,y-1),(x,y-1),(x+1,y-1)
    ])






