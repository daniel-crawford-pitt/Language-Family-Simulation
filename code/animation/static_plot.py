
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
import sys
import numpy as np



import sys
sys.path.insert(0, 'C:/Users\dcraw\OneDrive\Desktop\Language Family Simulation\code\simulation')
from sim_main import *

def plot_static(data):
    
    fig, ax = plt.subplots()


    # Custom colormap going from white (0) to blue (1)
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', [(1, 1, 1), (0, 0, 1)], N=256)


    ax.imshow(data, cmap=custom_cmap, vmin=0, vmax=1)

    ax.set_xticks(np.arange(0, 100, 10))
    ax.set_yticks(np.arange(0, 100, 10))
    

    plt.gca().invert_yaxis()
    plt.show()

if __name__ == '__main__':
    #data = np.random.rand(100, 100)
    data = sim(exist = np.zeros((100,100)))
    plot_static(data)