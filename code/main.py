from utils import *
read_config() #Need these first

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from objects.language import *
from objects.env import *
from simulation.sim_main import *



#Sim Env
env = Env()

sim_anim = env.sim()
plt.show()
#f = r"C:/Users/dcraw/OneDrive/Desktop/Language Family Simulation/anims" 
#writergif = animation.PillowWriter(fps=30) 
#sim_anim.save(f, writer=writergif)






