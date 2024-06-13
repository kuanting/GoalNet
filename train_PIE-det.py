import pandas as pd
import yaml
import argparse
import torch
from model_det.model import GoalNet

CONFIG_FILE_PATH = 'config/pie.yaml'  # yaml config file containing all the hyperparameters
EXPERIMENT_NAME = 'pie_15-45_det'  # arbitrary name for this experiment
DATASET_NAME = 'pie'

TRAIN_DATA_PATH = 'data/PIE/train_pie.pkl'
TRAIN_IMAGE_PATH = 'data/PIE/train'
VAL_DATA_PATH = 'data/PIE/val_pie.pkl'
VAL_IMAGE_PATH = 'data/PIE/val'
TEST_DATA_PATH = 'data/PIE/test_pie.pkl'
TEST_IMAGE_PATH = 'data/PIE/test'
OBS_LEN = 15  # in timesteps
PRED_LEN = 45  # in timesteps

# Reproducibility
torch.manual_seed(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Avoid creating subprocesses recursively in win system
if __name__ == '__main__':
    with open(CONFIG_FILE_PATH) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    experiment_name = CONFIG_FILE_PATH.split('.yaml')[0].split('config/')[1]
    print(params)

    df_train = pd.read_pickle(TRAIN_DATA_PATH)
    df_test = pd.read_pickle(TEST_DATA_PATH)
    df_val = pd.read_pickle(VAL_DATA_PATH)
    print(df_train.head())

    model = GoalNet(obs_len=OBS_LEN, pred_len=PRED_LEN, params=params)

    # The Val ADE and FDE are without TTST and CWS to save time. Therefore, the numbers will be worse than the final values
    model.train(df_train, df_test, params, TRAIN_IMAGE_PATH, TEST_IMAGE_PATH, EXPERIMENT_NAME, params['batch_size'], device=None, dataset_name=DATASET_NAME, val_data=df_val, val_image_path=VAL_IMAGE_PATH)
    
    # Train w/o ReduceLROnPlateau lr_scheduler
    #model.train(df_train, df_test, params, TRAIN_IMAGE_PATH, TEST_IMAGE_PATH, EXPERIMENT_NAME, params['batch_size'], device=None, dataset_name=DATASET_NAME)
