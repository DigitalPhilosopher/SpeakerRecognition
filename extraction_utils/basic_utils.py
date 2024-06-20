import os
import glob

import yaml




basepath = os.path.dirname(__file__)

base_path_package = os.path.dirname(os.path.dirname(basepath))

def get_config_yml():
    with open('config.yml', 'r') as stream:
        config = yaml.safe_load(stream)
    return config