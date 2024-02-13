import os
import json


def read_config():
    f = open('code/config.json')
    config = json.load(f)

   
    
    os.environ["MAX_NUMBER_LANGUAGES"] = str(config["MAX_NUMBER_LANGUAGES"])
    os.environ["NUM_INIT_LANGS"] = str(config["NUM_INIT_LANGS"])
    os.environ["FIELD_SIZE_TUPLE"] = str(config["FIELD_SIZE_TUPLE"])

    os.environ["MOMENTUM_FUNC_CLASS"] = str(config["MOMENTUM_FUNC_CLASS"])
    os.environ["MOMENTUM_FUNC_STATIC_BOOL"] = str(config["MOMENTUM_FUNC_STATIC_BOOL"])

    os.environ["SHOW_CONCAVE_HULL"] = str(config["SHOW_CONCAVE_HULL"])
    os.environ["SHOW_TREE_DIAGRAM"] = str(config["SHOW_TREE_DIAGRAM"])
    os.environ["SHOW_MAP"] = str(config["SHOW_MAP"])
    os.environ["SHOW_NOTHING"] = str(config["SHOW_NOTHING"])
    os.environ["MAX_TIME_STEPS"] = str(config["MAX_TIME_STEPS"])

    os.environ["SPLIT_THRESHOLD_FUNC_CLASS"] = str(config["SPLIT_THRESHOLD_FUNC_CLASS"])
    os.environ["SPLIT_THRESHOLD_CONST_VALUE"] = str(config["SPLIT_THRESHOLD_CONST_VALUE"])



    print('Config Read!')
 
if __name__ == "__main__":
    read_config()
