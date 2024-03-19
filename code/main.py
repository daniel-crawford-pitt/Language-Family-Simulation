from utils import *
config = read_config('.\config_files\config_exp1.json')

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from objects.language import *
from objects.env import *
from simulation.sim_main import *
from history_analytics import *

import time
import numpy as np
import os

t0 = time.time()

output_row = np.arange(0,91,10)

import os
if os.path.exists(os.path.abspath(config['OUTPUT_FILE'])):
  os.remove(os.path.abspath(config['OUTPUT_FILE']))
else:
    pass
  #print("The file does not exist")

with open(os.path.abspath(config['OUTPUT_FILE']), 'a') as f:
        writer = csv.writer(f)
        writer.writerow(output_row)



for i in range(2):
    os.environ['LANG_ID_CTR'] = "1"
    #Sim Env
    t_start = time.time()

    config = read_config('.\config_files\config_exp1.json')
    #print('Config Read!')

    t_config = time.time()
    env = Env(config)
    t_env = time.time()
    sim_anim = env.sim(config)
    t_sim = time.time()
    plt.show()
    #print('Simulation Complete Done!')

    history_analysis([l.history for l in env.languages if l is not None],config['OUTPUT_FILE'])
    time_ha = time.time()


    #print(f"TIME ANALYSIS +++++++++++++")
    #print(f"Config: \t{t_config-t0} s.")
    #print(f"SetEnv: \t{t_env-t_config} s.")
    #print(f"RunSim: \t{t_sim-t_env} s.")
    #print(f"HistAn: \t{time_ha-t_sim} s.")
    #print("=============================")
    #print(f"TotalT: \t{time_ha-t0} s.")

    print(f"Sim Number {i+1}; Sim Time: {round(time.time()-t_start,2)}; Total Time: {round(time.time()-t0,2)}")
