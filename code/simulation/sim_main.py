import numpy as np
import copy
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import ffmpeg


sys.path.insert(0, 'C:/Users\dcraw\OneDrive\Desktop\Language Family Simulation\code\simulation')
from logic.step_logic import *
from logic.sim_steps_functions import *

from sim_utils import *
from animation.animation import *


next_exist_fxn = weighted_rand_adj_step
pause = False




def simulate(lang_list):
    def onClick(event):
        global pause
        pause ^= True

    def update_sim(frame, lang_list):

        if not pause:
            # Clear previous plot contents
            ax.clear()

            for l in lang_list:
                # Generate data for the grid plot
                l.map = sim(l.map)
                #custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', [(1, 1, 1), (0, 0, 1)], N=256)          
                ax.imshow(l.map, cmap=l.color, alpha = 0.5)

            ax.set_xticks(np.arange(0, 100, 10))
            ax.set_yticks(np.arange(0, 100, 10))
        

        return ax

    

    fig, ax = plt.subplots()
    time_template = 'Time = %.1f s'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


    fig.canvas.mpl_connect('button_press_event', onClick)

    
    ani = animation.FuncAnimation(fig, update_sim, fargs= (lang_list,),
        blit=False, interval=10, frames = 100, 
        cache_frame_data=False, save_count=100,
        repeat=True)
    #fig.show()

    return ani

    




