import os
import sys
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import albumentations as A
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm

from ml.nn.cnn.architectures.ResNet import get_resnet50
from ml.nn.utils.aux_funcs import get_train_val_split, get_data_loaders, plot_loss, save_checkpoint
from python_utils.image_utils import get_image

sys.path.append('/home/sidorov/dev')
plt.style.use('ggplot')


class DataSet(Dataset):
    def __init__(self, data_df: pd.DataFrame, augs: A.Compose):
        self.data_df = data_df
        self.augs = augs
        self.random_crop = A.RandomCrop(height=HEIGHT, width=WIDTH, p=1.0)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        # - Get the data of the current sample
        img_fl, pdl1, pdl2, rspns = self.data_df.loc[
            index, ['path', 'pdl1', 'pdl2', 'response']
        ].values.flatten()

        # - Get the image
        img = get_image(image_file=img_fl)

        # - Get the label
        lbl = np.array([rspns])
        # lbl = np.array([pdl1, pdl2])

        # - Run regular augmentations on the cropped / rescaled image
        img = self.augs(image=img, mask=img).get('image')

        # img = np.expand_dims(img[:, :, 0], -1)
        # img = np.expand_dims(img, 0)

        img, lbl = torch.tensor(img, dtype=torch.float), torch.tensor(lbl, dtype=torch.float)

        img = torch.permute(img, (2, 0, 1))

        return img, lbl


# - Util functions
def get_name_type(file_name: str) -> (str, str):
    """
    Receives a file name in form NAME_IDX1_IDX2.TIF and returns the name of the file
    :param file_name:
    :return: (fl_name, fl_type): tuple of strings representing the name and the type of the file
    """

    fl_rvrs = file_name[::-1]
    fl_name_rvrs = fl_rvrs[fl_rvrs.index('.') + 1:]
    fl_name_rvrs = fl_name_rvrs[fl_name_rvrs.index('_') + 1:]
    fl_name = fl_name_rvrs[fl_name_rvrs.index('_') + 1:][::-1]
    fl_type = fl_rvrs[:fl_rvrs.index('.')][::-1]

    return fl_name, fl_type


def get_train_augs():
    return A.Compose(
        [
            # A.ToGray(p=1.0),
            A.OneOf([
                A.RandomRotate90(),
                A.VerticalFlip(),
                A.HorizontalFlip(),
            ], p=0.5),
        ], p=1.0)


def get_test_augs():
    return A.Compose(
        [
            # A.ToGray(p=1.0),
            A.OneOf([
                A.RandomRotate90(),
                A.VerticalFlip(),
                A.HorizontalFlip(),
            ], p=0.5),
        ], p=1.0)


def get_data(data_root_dir: pathlib.Path or str, metadata_file: pathlib.Path or str, save_file: pathlib.Path or str):
    metadata_df = pd.read_csv(metadata_file)
    lbls_df = metadata_df.loc[7:35, ['Path number', 'PD1 score', 'PDL1 score', 'PDL2 score', 'Response binary']]

    data_df = pd.DataFrame(columns=['name', 'path', 'pd1', 'pdl1', 'pdl2', 'response'])
    img_fls = os.listdir(data_root_dir)
    for img_fl in tqdm(img_fls):
        if DEBUG:
            print(f'> Getting data from {data_root_dir} ...\n')
        file_name, file_type = get_name_type(file_name=img_fl)
        values = lbls_df.loc[lbls_df.loc[:, 'Path number'] == file_name, [
            'PD1 score', 'PDL1 score', 'PDL2 score', 'Response binary'
        ]
        ].values.flatten()
        if len(values) > 1:
            pd1_scr, pdl1_scr, pdl2_scr, rspns = values
            if DEBUG:
                print(f'''
                File name: {file_name}
                - Response {rspns}
                - PD1 score: {pd1_scr}
                - PDL1 score: {pdl1_scr}
                - PDL2 score: {pdl2_scr}
                - Response: {rspns}
                ''')
            file_data_df = pd.DataFrame(
                dict(
                    name=file_name,
                    path=f'{data_root_dir}/{img_fl}',
                    pd1=pd1_scr,
                    pdl1=pdl1_scr,
                    pdl2=pdl2_scr,
                    response=rspns,
                ),
                index=pd.Index([0])
            )
            data_df = pd.concat([data_df, file_data_df], axis=0, ignore_index=True)
    if isinstance(save_file, pathlib.Path) or isinstance(save_file, str):
        save_file = pathlib.Path(save_file)
        data_df.to_csv(save_file, index=False)
    return data_df


# - Hyperparameters
# -- General
DEBUG = False
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# -- Paths
# DATA_PATH = pathlib.Path('C:/Users/Michael/Desktop/University/PhD/Projects/CancerDet/Cancer Dataset')  # Windows
# DATA_PATH = pathlib.Path('/Users/mchlsdrv/Desktop/University/PhD/Projects/CancerDet/data')  # Mac
DATA_PATH = pathlib.Path('/home/sidorov/projects/cancer_det/data')  # 4GPUs
TRAIN_DATA_PATH = DATA_PATH / 'train'
TRAIN_DATA_DF_PATH = DATA_PATH / 'train_data_df.csv'
METADATA_FILE = DATA_PATH / 'Table.csv'
OUTPUT_DIR = pathlib.Path('/media/oldrrtammyfs/Users/sidorov/CancerDet/output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -- Architecture
HEIGHT = 256
WIDTH = 256
# CHANNELS = 1
CHANNELS = 3
CLASSES = 1
# CLASSES = 2
MODEL = get_resnet50(image_channels=CHANNELS, num_classes=CLASSES).to(DEVICE)

# -- Training
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 16
VAL_PROP = 0.2
LEARNING_RATE = 0.001
EPOCHS = 100
N_WORKERS = 4
PIN_MEMORY = True

OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)
LOSS_FUNC = nn.MSELoss()

LOSS_PLOT_RESOLUTION = 10

# - Main
if __name__ == '__main__':
    print(f'- Getting data')
    if not TRAIN_DATA_DF_PATH.is_file():
        train_data_frame = get_data(data_root_dir=TRAIN_DATA_PATH, metadata_file=METADATA_FILE,
                                    save_file=TRAIN_DATA_DF_PATH)
    else:
        train_data_frame = pd.read_csv(TRAIN_DATA_DF_PATH)

    # - Split data into train / validation datasets
    train_data_df, val_data_df = get_train_val_split(data_df=train_data_frame.loc[:1000, :], val_prop=VAL_PROP)

    train_ds = DataSet(data_df=train_data_df, augs=get_train_augs())
    val_ds = DataSet(data_df=val_data_df, augs=get_test_augs())

    train_data_loader, val_data_loader = get_data_loaders(
        train_dataset=train_ds,
        train_batch_size=TRAIN_BATCH_SIZE,
        val_dataset=val_ds,
        val_batch_size=VAL_BATCH_SIZE,
    )

    epoch_train_losses = np.array([])
    epoch_val_losses = np.array([])

    loss_plot_start_idx, loss_plot_end_idx = 0, LOSS_PLOT_RESOLUTION
    loss_plot_train_history = []
    loss_plot_val_history = []

    # - Training loop
    for epch in tqdm(range(EPOCHS)):
        # - Train
        btch_train_losses = np.array([])
        btch_val_losses = np.array([])
        for btch_idx, (imgs, lbls) in enumerate(train_data_loader):
            # - Store the data in CUDA
            imgs = imgs.to(DEVICE)
            lbls = lbls.to(DEVICE)

            # - Forward
            preds = MODEL(imgs)
            loss = LOSS_FUNC(preds, lbls)
            btch_train_losses = np.append(btch_train_losses, loss.item())

            # - Backward
            OPTIMIZER.zero_grad()
            loss.backward()

            # - Optimizer step
            OPTIMIZER.step()

        # - Validation
        with torch.no_grad():
            for btch_idx, (imgs, lbls) in enumerate(val_data_loader):
                imgs = imgs.to(DEVICE)
                lbls = lbls.to(DEVICE)

                preds = MODEL(imgs)
                loss = LOSS_FUNC(preds, lbls)

                btch_val_losses = np.append(btch_val_losses, loss.item())

        epoch_train_losses = np.append(epoch_train_losses, btch_train_losses.mean())
        epoch_val_losses = np.append(epoch_val_losses, btch_val_losses.mean())

        if len(epoch_train_losses) >= loss_plot_end_idx and len(epoch_val_losses) >= loss_plot_end_idx:
            # - Add the mean history
            loss_plot_train_history.append(epoch_train_losses[loss_plot_start_idx:loss_plot_end_idx].mean())
            loss_plot_val_history.append(epoch_val_losses[loss_plot_start_idx:loss_plot_end_idx].mean())

            # - Plot the mean history
            plot_loss(
                train_losses=loss_plot_train_history,
                val_losses=loss_plot_val_history,
                x_ticks=np.arange(1, (epch + 1) // LOSS_PLOT_RESOLUTION + 1) * LOSS_PLOT_RESOLUTION,
                x_label='Epochs',
                y_label='MSE',
                title='Train vs Validation Plot',
                train_loss_marker='b-', val_loss_marker='r-',
                train_loss_label='train', val_loss_label='val',
                output_dir=OUTPUT_DIR
            )

            # - Save model weights
            checkpoint_dir = OUTPUT_DIR / 'checkpoints'
            os.makedirs(checkpoint_dir, exist_ok=True)
            save_checkpoint(model=MODEL, filename=checkpoint_dir / f'weights_epoch_{epch}.pth.tar', epoch=epch)

            loss_plot_start_idx += LOSS_PLOT_RESOLUTION
            loss_plot_end_idx += LOSS_PLOT_RESOLUTION
