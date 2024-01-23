import os
import json


def read_config():
    f = open('code/config.json')
    config = json.load(f)

   
    
    os.environ["MAX_NUMBER_LANGUAGES"] = str(config["MAX_NUMBER_LANGUAGES"])
    os.environ["NUM_INIT_LANGS"] = str(config["NUM_INIT_LANGS"])
    os.environ["FIELD_SIZE_TUPLE"] = str(config["FIELD_SIZE_TUPLE"])

    print('Config Read!')
 
if __name__ == "__main__":
    read_config()
