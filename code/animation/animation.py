"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

import sys
import numpy as np




import sys
sys.path.insert(0, 'C:/Users\dcraw\OneDrive\Desktop\Language Family Simulation\code\simulation')
from sim_main import *

pause = False
data = [np.zeros((100,100))]
data[0][0][0] = 1.0

def update(frame):
    if not pause:
        # Clear previous plot contents
        ax.clear()

        # Generate data for the grid plot (you can modify this for your specific use case)
        data[0] = sim(data[0])

        custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', [(1, 1, 1), (0, 0, 1)], N=256)


        ax.imshow(data[0], cmap=custom_cmap, vmin=0, vmax=1)

        ax.set_xticks(np.arange(0, 100, 10))
        ax.set_yticks(np.arange(0, 100, 10))


    return ax

def onClick(event):
    global pause
    pause ^= True


fig, ax = plt.subplots()
time_template = 'Time = %.1f s'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


fig.canvas.mpl_connect('button_press_event', onClick)
ani = animation.FuncAnimation(fig, update, 
    blit=False, interval=10, frames = 100, 
    cache_frame_data=False,
    repeat=True)
fig.show()

"""