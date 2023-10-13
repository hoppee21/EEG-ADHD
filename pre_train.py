import numpy as np
import os
import time
import yaml
import argparse
from utils import _transform_data

def main(args):
    old_time = time.time()

    if os.path.isfile(args.config):
        with open(args.config, "r") as fd:
            config = yaml.load(fd, Loader=yaml.FullLoader)
    else:
        raise ValueError("Config file does not exist.")
    
    root_dir = config['dataset']['dataset_root_dir']

    print('start transform data')
    _transform_data(root_dir)

    current_time = time.time()
    print('time: ' + str(current_time - old_time) + 's')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate our model')
    parser.add_argument('--config', type=str, metavar='DIR',
                        help='path to a config file')
    args = parser.parse_args()
    main(args)