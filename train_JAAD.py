import pandas as pd
import yaml
import argparse
import torch
from model.model import GoalNet

CONFIG_FILE_PATH = 'config/jaad.yaml'  # yaml config file containing all the hyperparameters
EXPERIMENT_NAME = 'jaad_15-45'  # arbitrary name for this experiment
DATASET_NAME = 'jaad'

TRAIN_DATA_PATH = 'data/JAAD/train_jaad.pkl'
TRAIN_IMAGE_PATH = 'data/JAAD/train'
VAL_DATA_PATH = 'data/JAAD/val_jaad.pkl'
VAL_IMAGE_PATH = 'data/JAAD/val'
TEST_DATA_PATH = 'data/JAAD/test_jaad.pkl'
TEST_IMAGE_PATH = 'data/JAAD/test'
OBS_LEN = 15  # in timesteps
PRED_LEN = 45  # in timesteps
NUM_GOALS = 20  # K_e
NUM_TRAJ = 1  # K_a

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
    model.train(df_train, df_test, params, TRAIN_IMAGE_PATH, TEST_IMAGE_PATH, EXPERIMENT_NAME, params['batch_size'], NUM_GOALS, NUM_TRAJ, device=None, dataset_name=DATASET_NAME, val_data=df_val, val_image_path=VAL_IMAGE_PATH)
    
    # Train w/o ReduceLROnPlateau lr_scheduler
    #model.train(df_train, df_test, params, TRAIN_IMAGE_PATH, TEST_IMAGE_PATH, EXPERIMENT_NAME, params['batch_size'], NUM_GOALS, NUM_TRAJ, device=None, dataset_name=DATASET_NAME)
