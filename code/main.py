from utils import *
read_config() #Need these first

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from objects.language import *
from objects.env import *
from simulation.sim_main import *

import os
os.environ['LANG_ID_CTR'] = "1"


#Sim Env
env = Env()

sim_anim = env.sim()

plt.show()
print('Done')





