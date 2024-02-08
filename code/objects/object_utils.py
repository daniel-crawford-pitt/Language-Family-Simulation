import numpy as np
import matplotlib
import numpy.random as r

def combine_ll_to_rbg(ll):
    np.stack(np.array([l.map for l in ll]), axis = 2)


def random_color_near(color_str):
    tpl = np.array(matplotlib.colors.to_rgb(color_str))
    rp = r.random(3)*0.3-0.15
    return tuple(np.clip(tpl+rp, 0,1))

if __name__ == "__main__":
    random_color_near("blue")