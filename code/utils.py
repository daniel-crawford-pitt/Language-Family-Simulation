import os
import json


def read_config(fn):
    f = open(os.path.abspath(fn))
    config = json.load(f)
    #print('Config Read!')
    return config
 
if __name__ == "__main__":
    read_config()
