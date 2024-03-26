import matplotlib.pyplot as plt
from multiprocessing import Process

from objects.language import *
from objects.env import *
from simulation.sim_main import *
from history_analytics import *

import time
import numpy as np
import os


from utils import *
config_file = '.\config_files\config_test_pres.json'
multi_config_file = '.\multi_config_files\\pres.json'

config = read_config(config_file)
time_total_start = time.time()
def main(config, n, total):
  print(f"Beginning Simulation with Configuration {n} of {total}")
  for i in range(config["NUM_SIM_RUNS"]):
    t0 = time.time()
    os.environ['LANG_ID_CTR'] = "1"
    os.environ["SPLIT_THRESHOLD_MULTIPLIER"] = config["SPLIT_THRESHOLD_MULTIPLIER"]
    #Sim Env
    t_start = time.time()

    #print('Config Read!')

    t_config = time.time()

    

    env = Env(config)
    t_env = time.time()
    env.sim(config)

    

    
    
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

    print(f"Sim Number {i+1}; Sim Time: {round(time.time()-t_start,2)}; Total Time: {round(time.time()-time_total_start,2)}")





if multi_config_file is not None:
    multi_config_dict, output_file = read_multi_config(multi_config_file)
    config["OUTPUT_FILE"] = output_file


    #clear output file

    if os.path.exists(os.path.abspath(config["OUTPUT_FILE"])):
       os.remove(os.path.abspath(config["OUTPUT_FILE"]))


    print("\nMultiple Runs with:")
    print(combine_vals(multi_config_dict["change_vars"]))
    print(f"Writing to: {output_file}\n")


    os.environ["MAX_TIME_STEPS"] = str(config["MAX_TIME_STEPS"])
    
    output_row = ['Config', 'Absolute'] #+ list(np.arange(0 ,config["MAX_TIME_STEPS"]+1, 10))
    for t in np.arange(0,config['MAX_TIME_STEPS']+1,10):
        output_row.append(f't{t}_true')
        for h in np.arange(0,t+1,10):
            output_row.append(f't{t}_h{h}')
        

    with open(os.path.abspath(config['OUTPUT_FILE']), 'a') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(output_row)
    

    combined_vals = combine_vals(multi_config_dict["change_vars"])
    for mc_n, multi_con in enumerate(combined_vals):
        #set con
        os.environ["PRINT_PREAMBLE"] = str(multi_con)
        

        
        for key, val in multi_con.items():
          config[key] = val

          main(config, mc_n+1, len(combined_vals))

        

