import os
import json


def read_config():
    f = open('code/config.json')
    config = json.load(f)
    #print('Config Read!')
    return config
 
if __name__ == "__main__":
    read_config()
