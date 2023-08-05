import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import albumentations as A
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from ml.nn.cnn.architectures import ResNet

plt.style.use('ggplot')


# - Classes
class DataSet(Dataset):
    def __init__(self, data_df: pd.DataFrame, augs: A.Compose):
        self.data_df = data_df
        self.augs = augs

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        # - Get the data of the current sample
        img_fl, resp_bin = self.data_df.loc[index, ['image_file', 'valid']].values.flatten()

        # - Get the image
        img = get_image(image_file=img_fl)

        # - Get the label
        lbl = resp_bin
        lbl = np.expand_dims(lbl, 0)

        # - Run regular augmentations on the cropped / rescaled image
        img = self.augs(image=img, mask=img).get('image')

        img, lbl = torch.tensor(img, dtype=torch.float), torch.tensor(lbl, dtype=torch.float)

        img = torch.permute(img, (2, 0, 1))

        return img, lbl


# - Util functions
def get_resnet50(image_channels=3, num_classes=1000):
    return ResNet.ResNet(block=ResNet.ResBlock, layers=[3, 4, 6, 3], image_channels=image_channels,
                         num_classes=num_classes)


def get_resnet101(image_channels=3, num_classes=1000):
    return ResNet.ResNet(block=ResNet.ResBlock, layers=[3, 4, 23, 3], image_channels=image_channels,
                         num_classes=num_classes)


def get_resnet152(image_channels=3, num_classes=1000):
    return ResNet.ResNet(block=ResNet.ResBlock, layers=[3, 8, 36, 3], image_channels=image_channels,
                         num_classes=num_classes)


def get_image(image_file: str or pathlib.Path):
    img = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


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


def get_train_val_split(data_df: pd.DataFrame, val_prop: float = .2):
    n_items = len(data_df)
    item_idxs = np.arange(n_items)
    n_val_items = int(n_items * val_prop)

    # - Randomly pick the validation items' indices
    val_idxs = np.random.choice(item_idxs, n_val_items, replace=False)

    # - Pick the items for the validation set
    val_data = data_df.loc[val_idxs, :].reset_index(drop=True)

    # - The items for training are the once which are not included in the
    # validation set
    train_data = data_df.loc[np.setdiff1d(item_idxs, val_idxs), :].reset_index(drop=True)

    return train_data, val_data


def get_data_loaders(data: pd.DataFrame, batch_size: int, val_prop: float):
    # - Split data into train / validation datasets
    train_data_df, val_data_df = get_train_val_split(data_df=data, val_prop=val_prop)

    # - Create the train / validation dataloaders
    train_dl = DataLoader(
        DataSet(data_df=train_data_df, augs=get_train_augs()),
        batch_size=batch_size,
        num_workers=N_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True
    )

    val_dl = DataLoader(
        DataSet(data_df=val_data_df, augs=get_test_augs()),
        batch_size=batch_size // VAL_BATCH_SIZE_SCALE,
        num_workers=N_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False
    )

    return train_dl, val_dl


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
    return data_df


def get_x_ticks(epoch):
    x_ticks = np.arange(1, epoch)
    if 20 < epoch < 50:
        x_ticks = np.arange(1, epoch, 5)
    elif 50 < epoch < 100:
        x_ticks = np.arange(1, epoch, 10)
    elif 100 < epoch < 1000:
        x_ticks = np.arange(1, epoch, 50)
    elif 1000 < epoch < 10000:
        x_ticks = np.arange(1, epoch, 500)

    return x_ticks


def plot_loss(train_losses, val_losses, x_ticks: np.ndarray, x_label: str, y_label: str,
              title='Train vs Validation Plot',
              train_loss_marker='bo-', val_loss_marker='r*-',
              train_loss_label='train', val_loss_label='val', output_dir: pathlib.Path or str = './outputs'):
    fig, ax = plt.subplots()
    ax.plot(x_ticks, train_losses, train_loss_marker, label=train_loss_label)
    ax.plot(x_ticks, val_losses, val_loss_marker, label=val_loss_label)
    ax.set(xlabel=x_label, ylabel=y_label, xticks=get_x_ticks(epoch=epch))
    fig.suptitle(title)
    plt.legend()
    output_dir = pathlib.Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    plot_output_dir = output_dir / 'plots'
    os.makedirs(plot_output_dir, exist_ok=True)
    fig.savefig(plot_output_dir / 'loss.png')
    plt.close(fig)


def save_checkpoint(model: torch.nn.Module, filename: pathlib.Path or str = 'my_checkpoint.pth.tar', epoch: int = 0):
    if epoch > 0:
        print(f'\n=> Saving checkpoint for epoch {epoch}')
    else:
        print(f'\n=> Saving checkpoint')

    torch.save(model.state_dict(), filename)


def load_checkpoint(model, checkpoint):
    print('=> Loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])


# - Hyperparameters
# -- General
DEBUG = False
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# -- Paths
# DATA_PATH = pathlib.Path('C:/Users/Michael/Desktop/University/PhD/Projects/CancerDet/Cancer Dataset')  # Windows
# DATA_PATH = pathlib.Path('/Users/mchlsdrv/Desktop/University/PhD/Projects/CancerDet/data')  # Mac
DATA_PATH = pathlib.Path('/home/sidorov/projects/cancer_det/data')  # 4GPUs
METADATA_FILE = DATA_PATH / 'Rambam clinical table 26.6.23.csv'
OUTPUT_DIR = pathlib.Path('/media/oldrrtammyfs/Users/sidorov/CancerDet/output/image_filter')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -- Architecture
HEIGHT = 256
WIDTH = 256
CHANNELS = 3
CLASSES = 1
MODEL = get_resnet50(image_channels=CHANNELS, num_classes=CLASSES).to(DEVICE)

# -- Training
BATCH_SIZE = 64
VAL_BATCH_SIZE_SCALE = 16
VAL_PROP = 0.2
LEARNING_RATE = 0.001
EPOCHS = 1500
# N_WORKERS = 1
N_WORKERS = 4
PIN_MEMORY = True

OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)
LOSS_FUNC = nn.MSELoss()

LOSS_PLOT_RESOLUTION = 10

# - Main
if __name__ == '__main__':
    print(f'- Getting data')
    train_data_frame = get_data(data_root_dir=DATA_PATH)

    train_data_loader, val_data_loader = get_data_loaders(
        data=train_data_frame,
        batch_size=BATCH_SIZE,
        val_prop=VAL_PROP
    )

    epoch_train_losses = np.array([])
    epoch_val_losses = np.array([])

    loss_plot_start_idx, loss_plot_end_idx = 0, LOSS_PLOT_RESOLUTION
    loss_plot_train_history = []
    loss_plot_val_history = []
    print(f'> Running on: ({DEVICE})')
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
