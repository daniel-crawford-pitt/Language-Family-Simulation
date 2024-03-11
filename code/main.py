from utils import *
read_config() #Need these first

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from objects.language import *
from objects.env import *
from simulation.sim_main import *
from history_analytics import *
import os
os.environ['LANG_ID_CTR'] = "1"


#Sim Env
config = read_config()
print('Config Read!')
env = Env(config)
sim_anim = env.sim(config)

plt.show()
print('Simulation Complete Done!')
history_analysis([l.history for l in env.languages if l is not None])





