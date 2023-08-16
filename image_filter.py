import datetime
import os
import sys
import pathlib
import cv2
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import albumentations as A
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm

sys.path.append('/home/sidorov/dev')
from ml.nn.cnn.architectures.ResNet import get_resnet50
from ml.nn.utils.aux_funcs import (
    get_train_val_split,
    get_data_loaders,
    plot_loss,
    save_checkpoint,
    load_checkpoint,
    get_arg_parser,
    get_device
)
from python_utils.image_utils import get_image


plt.style.use('ggplot')

# - Classes
class DataSet(Dataset):
    def __init__(self, data_df: pd.DataFrame, augs: A.Compose, to_gray: bool,
                 binary_label: bool, channels_first: bool = False):
        self.data_df = data_df
        self.augs = augs
        self.to_gray = to_gray
        self.binary_label = binary_label
        self.channels_first = channels_first

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        # - Get the data of the current sample
        img_fl, valid = self.data_df.loc[index, ['image_file', 'valid']].values.flatten()

        # - Get the image
        img = get_image(image_file=img_fl, to_gray=self.to_gray)

        # - Get the label
        if self.binary_label:
            if valid > 0:
                lbl = np.array([1., 0.])
            else:
                lbl = np.array([0., 1.])
        else:
            lbl = np.array([valid])

        # - Run regular augmentations on the cropped / rescaled image
        img = self.augs(image=img, mask=img).get('image')

        img, lbl = torch.tensor(img, dtype=torch.float), torch.tensor(lbl, dtype=torch.float)
        if self.channels_first:
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
            A.OneOf([
                A.RandomRotate90(),
                A.VerticalFlip(),
                A.HorizontalFlip(),
            ], p=0.5),
        ], p=1.0)


def get_test_augs():
    return A.Compose(
        [
            A.OneOf([
                A.RandomRotate90(),
                A.VerticalFlip(),
                A.HorizontalFlip(),
            ], p=0.5),
        ], p=1.0)


def get_data(data_root_dir: pathlib.Path or str):
    data_df = pd.DataFrame(columns=['image_file', 'valid'])
    for root, folders, _ in os.walk(data_root_dir, topdown=False):
        for folder in folders:
            img_fls = os.listdir(f'{root}/{folder}')
            for img_fl in img_fls:
                if folder == 'low_signal':
                    file_data_df = pd.DataFrame(
                        dict(
                            image_file=f'{root}/{folder}/{img_fl}',
                            valid=0
                        ),
                        index=pd.Index([0])
                    )
                    data_df = pd.concat([data_df, file_data_df], axis=0, ignore_index=True)
                elif folder == 'high_signal':
                    file_data_df = pd.DataFrame(
                        dict(
                            image_file=f'{root}/{folder}/{img_fl}',
                            valid=1
                        ),
                        index=pd.Index([0])
                    )
                    data_df = pd.concat([data_df, file_data_df], axis=0, ignore_index=True)
    # - Shuffle the lines
    data_df = data_df.sample(frac=1).reset_index(drop=True)

    return data_df


# - Main
def train_model(model, epochs: int,
                train_data_loader: torch.utils.data.DataLoader, val_data_loader: torch.utils.data.DataLoader,
                save_dir: pathlib.Path or str):

    epoch_train_losses = np.array([])
    epoch_val_losses = np.array([])

    loss_plot_start_idx, loss_plot_end_idx = 0, LOSS_PLOT_RESOLUTION
    loss_plot_train_history = []
    loss_plot_val_history = []
    print(f'> Running on: ({DEVICE})')

    # - Training loop
    for epch in tqdm(range(epochs)):
        # - Train
        btch_train_losses = np.array([])
        btch_val_losses = np.array([])
        for btch_idx, (imgs, lbls) in enumerate(train_data_loader):
            # - Store the data in CUDA
            imgs = imgs.to(DEVICE)
            lbls = lbls.to(DEVICE)

            # - Forward
            preds = model(imgs)
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
            save_dir = pathlib.Path(save_dir)
            checkpoint_dir = save_dir / 'checkpoints'
            os.makedirs(checkpoint_dir, exist_ok=True)
            save_checkpoint(model=MODEL, filename=checkpoint_dir / f'weights_epoch_{epch}.pth.tar', epoch=epch)

            loss_plot_start_idx += LOSS_PLOT_RESOLUTION
            loss_plot_end_idx += LOSS_PLOT_RESOLUTION

    return model

def filter_images(model, image_dir_path: pathlib.Path, save_dir: pathlib.Path, save_image_type: str = 'tif'):
    image_dir_path = pathlib.Path(image_dir_path)
    img_fls = os.listdir(image_dir_path)
    for img_fl in tqdm(img_fls):

        img = get_image(image_file=image_dir_path / img_fl, to_gray=TO_GRAY)

        # - Add the batch dim
        img = np.expand_dims(img, 0)

        try:
            img_tnsr = torch.tensor(img, dtype=torch.float, device=DEVICE)

            img_tnsr = torch.permute(img_tnsr, (0, 3, 1, 2))

            valid = model(img_tnsr).item()
            img_name = img_fl[::-1]
            img_name = img_name[img_name.index('.')+1:][::-1]

            # - Create the output dirs
            os.makedirs(save_dir, exist_ok=True)

            valid_dir, invalid_dir = save_dir / 'valid', save_dir / 'invalid'
            os.makedirs(valid_dir, exist_ok=True)
            os.makedirs(invalid_dir, exist_ok=True)

            save_file = f'{img_name}.{save_image_type}'
            if valid > 0.5:
                save_file = valid_dir / save_file
            else:
                save_file = invalid_dir / save_file

            # - Reload the image
            img = get_image(image_file=image_dir_path / img_fl, to_gray=False)
            cv2.imwrite(str(save_file), img)
        except Exception as err:
            print(err)


# - Hyperparameters
# -- General
# - Get the current timestamp to be used for the run differentiate the run
TS = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
PARSER = get_arg_parser()
ARGS = PARSER.parse_args()

# - Hyperparameters
# -- General
DEBUG = ARGS.debug
DEVICE = get_device(gpu_id=ARGS.gpu_id)

# DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# -- Paths
TRAIN_DATA_PATH = pathlib.Path('/home/sidorov/projects/cancer_det/data/filter_data')  # 4GPUs
TEST_DATA_PATH = pathlib.Path('/home/sidorov/projects/cancer_det/data/train')  # 4GPUs
if isinstance(ARGS.name, str):
    OUTPUT_DIR = pathlib.Path(f'/media/oldrrtammyfs/Users/sidorov/CancerDet/output/image_filter/{ARGS.name}_{TS}')
else:
    OUTPUT_DIR = pathlib.Path(f'/media/oldrrtammyfs/Users/sidorov/CancerDet/output/image_filter/{TS}')
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHECKPOINT_FILE = pathlib.Path(
    '/media/oldrrtammyfs/Users/sidorov/CancerDet/output/image_filter/checkpoints_mse/weights_epoch_149.pth.tar')

# -- Architecture
HEIGHT = 256
WIDTH = 256
CHANNELS = 3
# CHANNELS = 1
CLASSES = 1
# CLASSES = 2
MODEL = get_resnet50(image_channels=CHANNELS, num_classes=CLASSES).to(DEVICE)

# -- Training
# > Hyperparameters
N_DATA_SAMPLES = 5000
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 16
VAL_PROP = 0.2
LEARNING_RATE = 0.001
EPOCHS = 100
TO_GRAY = True if CHANNELS == 1 else False
N_WORKERS = 4
PIN_MEMORY = True
CHANNELS_FIRST = True

# > Optimizer
OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)

# > Loss
BINARY_LABEL = False
# BINARY_LABEL = True
LOSS_FUNC = nn.CrossEntropyLoss() if BINARY_LABEL else nn.MSELoss()

# > Plots
LOSS_PLOT_RESOLUTION = 10


if __name__ == '__main__':
    if ARGS.load_weights and CHECKPOINT_FILE.is_file():
        load_checkpoint(model=MODEL, checkpoint=CHECKPOINT_FILE)

    if ARGS.train:
        print(f'- Getting data')
        train_data_frame = get_data(data_root_dir=TRAIN_DATA_PATH)

        # - Split data into train / validation datasets
        train_data_df, val_data_df = get_train_val_split(data_df=train_data_frame, val_prop=VAL_PROP)

        train_ds = DataSet(data_df=train_data_df, augs=get_train_augs(), to_gray=TO_GRAY, binary_label=BINARY_LABEL,
                           channels_first=CHANNELS_FIRST)

        val_ds = DataSet(data_df=val_data_df, augs=get_test_augs(), to_gray=TO_GRAY, binary_label=BINARY_LABEL,
                         channels_first=CHANNELS_FIRST)

        train_dl, val_dl = get_data_loaders(
            train_dataset=train_ds,
            train_batch_size=TRAIN_BATCH_SIZE,
            val_dataset=val_ds,
            val_batch_size=VAL_BATCH_SIZE,
        )
        MODEL = train_model(
            model=MODEL,
            epochs=EPOCHS,
            train_data_loader=train_dl,
            val_data_loader=val_dl,
            save_dir=OUTPUT_DIR
        )

    filter_images(
        model=MODEL,
        image_dir_path=TEST_DATA_PATH,
        save_dir=OUTPUT_DIR / 'filtered',
        save_image_type='tif'
    )
