import argparse
import json
import os
import sys
from pprint import pprint

import numpy as np
import torch
from easydict import EasyDict as edict


def makedir(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)


def one_hot(category_labels, num_categories):
    '''

    :param category_labels: a np.ndarray or a tensor with size [batch_size, ]
    :return: a tensor with size [batch_size, num_categories]
    '''
    if isinstance(category_labels, torch.Tensor):
        labels = category_labels.cpu().numpy()
    else:
        labels = category_labels
    num_samples = labels.shape[0]
    one_hot_labels = np.zeros((num_samples, num_categories), dtype=np.float32)  # [num_samples. dim_z_category]
    one_hot_labels[np.arange(num_samples), labels] = 1
    one_hot_labels = torch.from_numpy(one_hot_labels)

    if torch.cuda.is_available():
        one_hot_labels = one_hot_labels.cuda()
    return one_hot_labels



def parse_args():
    """
    Parse the arguments of the program
    :return: (config_args)
    :rtype: tuple
    """
    # Create a parser
    parser = argparse.ArgumentParser(description="VAE PyTorch Implementation")
    parser.add_argument('--version', action='version', version='%(prog)s 0.0.1')
    parser.add_argument('--config', default=None, type=str, help='Configuration file')

    # Parse the arguments
    args = parser.parse_args()

    # Parse the configurations from the config json file provided
    try:
        if args.config is not None:
            with open(args.config, 'r') as config_file:
                config_args_dict = json.load(config_file)
        else:
            print("Add a config file using \'--config file_name.json\'")
            exit(1)

    except FileNotFoundError:
        print("ERROR: Config file not found: {}".format(args.config))
        exit(1)
    except json.decoder.JSONDecodeError:
        print("ERROR: Config file is not a proper JSON file!")
        exit(1)

    config_args = edict(config_args_dict)

    pprint(config_args)
    print("\n")

    return config_args


def create_experiment_dirs(exp_dir):
    """
    Create Directories of a regular tensorflow experiment directory
    :param exp_dir:
    :return summary_dir, checkpoint_dir:
    """
    # experiment_dir = os.path.realpath(
    #     os.path.join(os.path.dirname(__file__))) + "/../experiments/" + exp_dir + "/"
    experiment_dir = os.path.join('/scratch4/sunxm/VAE/experiments', exp_dir)
    summary_dir = os.path.join(experiment_dir, 'summaries/')
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints/')

    dirs = [summary_dir, checkpoint_dir]
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        print("Experiment directories created!")
        # return experiment_dir, summary_dir, checkpoint_dir
        return summary_dir, checkpoint_dir
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)
