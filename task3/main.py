import os
import torch
import numpy as np
import json
import argparse
import random
import yaml
from load_data import CTDataset1
from torch.utils.data import DataLoader
import nibabel as nib
from nibabel.imageglobals import LoggingOutputSuppressor
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm
from matplotlib import pyplot as plt
import logging
import sys
from datetime import datetime
from model import SAM_classification
logger = logging.getLogger()
logging.getLogger('matplotlib.font_manager').disabled = True


def get_config(yaml_path):
    with open(yaml_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def get_time():
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


def prepare_dirs_loggers(config, script=""):
    logFormatter = logging.Formatter("%(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setLevel(logging.DEBUG)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    log_dir = config['log_dir'] + '/' + config['exp']
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    config['time_stamp'] = get_time()
    config['script'] = script
    dir_name = "{}-{}".format(config['time_stamp'], script) if script else config['time_stamp']
    config['session_dir'] = os.path.join(log_dir, dir_name)
    os.mkdir(config['session_dir'])

    fileHandler = logging.FileHandler(os.path.join(config['session_dir'],
                                                   'session.log'))
    fileHandler.setLevel(logging.DEBUG)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    # save config
    param_path = os.path.join(config['session_dir'], "params.json")
    with open(param_path, 'w') as fp:
        json.dump(config, fp, indent=4, sort_keys=True)


def get_dataset(config, model=None):
    train_set = CTDataset1(config, 'train', model=model)
    valid_set = CTDataset1(config, 'valid', model=model)
    test_set = valid_set
    return {'train': train_set, 'valid': valid_set, 'test': test_set}


def get_dataloader(config, dataset):
    train_loader = DataLoader(dataset=dataset['train'],
                            batch_size=config['batch_size'],
                            shuffle=True,
                            num_workers=config['num_workers'])
    valid_loader = DataLoader(dataset=dataset['valid'],
                            batch_size=1,
                            shuffle=False,
                            num_workers=config['num_workers'])
    test_loader = DataLoader(dataset=dataset['test'],
                            batch_size=1,
                            shuffle=False,
                            num_workers=config['num_workers'])
    return train_loader, valid_loader, test_loader


def get_model(config):
    model = sam_model_registry[config['model_name']](config)
    return model


def get_optimizer(config, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    return optimizer


if __name__ == '__main__':
    config = get_config('cfg.yaml')
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if config['use_gpu']:
        torch.cuda.manual_seed(config['seed'])
        torch.cuda.set_device(config['gpu_idx'])
    if config['logging']:
        prepare_dirs_loggers(config, os.path.basename(__file__).split('.')[0])
    model = SAM_classification(config)    
    with LoggingOutputSuppressor():
        if config['embedded']:
            dataset = get_dataset(config, model.model)
        else:
            dataset = get_dataset(config)
        train_loader, valid_loader, test_loader = get_dataloader(config, dataset)
    if config['mode'] == 'train':
        model.train(train_loader, valid_loader)
    elif config['mode'] == 'test':
        model.load_model(config['model_path'])
        model.validate(valid_loader)