import pathlib
import numpy as np
import pandas as pd


TEST_PROP = 0.1

DATA_PATH_LOCAL = pathlib.Path('/home/sidorov/projects/cancer_det/data')  # 4GPUs
DATA_FRAME_FILE = DATA_PATH_LOCAL / 'data_frame_complete.csv'
DATA_FRAME_FILE_TRAIN = DATA_PATH_LOCAL / 'train_data_frame.csv'
DATA_FRAME_FILE_TEST = DATA_PATH_LOCAL / 'test_data_frame.csv'

# train_df = pd.read_csv(DATA_FRAME_FILE_TRAIN)
test_df = pd.read_csv(DATA_FRAME_FILE_TEST)

df = pd.read_csv(DATA_FRAME_FILE)
N = len(df)
N_TEST = int(N * TEST_PROP)
data_idxs = np.arange(N)

# - Test data frame
test_idxs = np.random.choice(data_idxs, N_TEST, replace=False)
test_df = df.loc[test_idxs].reset_index(drop=True)

# - Train data frame
train_idxs = np.setdiff1d(data_idxs, test_idxs)
train_df = df.loc[train_idxs].reset_index(drop=True)

test_df.to_csv(DATA_FRAME_FILE_TEST, index=False)
train_df.to_csv(DATA_FRAME_FILE_TRAIN, index=False)


