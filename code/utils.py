import os
import json
import itertools

def read_config(fn):
    f = open(os.path.abspath(fn))
    config = json.load(f)
    #print('Config Read!')
    return config

def read_multi_config(fn):
    f = open(os.path.abspath(fn))
    config = json.load(f)
    #print('Config Read!')
    return config, config["OUTPUT_FILE"]
 
def combine_vals(d):
    keys, values = zip(*d.items())
    return [dict(zip(keys, con)) for con in [v for v in itertools.product(*values)]]
        


if __name__ == "__main__":
    #read_config()

    print(combine_vals(
        {
            "A":["A1","A2","A3"],
            "B":["B1","B2","B3"],
            "C":["C1","C2","C3"]         
        }
    ))
