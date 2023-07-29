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
from ml.nn.cnn.utils.regularization import stochastic_depth
plt.style.use('ggplot')


# - Classes
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super().__init__()
        self.channel_expansion = 4
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels * self.channel_expansion,
            kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.channel_expansion)
        self.activation = nn.ELU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        x_identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)

        if self.identity_downsample is not None:
            x_identity = self.identity_downsample(x_identity)

        x += x_identity
        x = self.activation(x)

        x = stochastic_depth(self, x, min_survival_prop=0.5)

        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels=image_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.activation = nn.ELU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(block=block, num_residual_blocks=layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(block=block, num_residual_blocks=layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(block=block, num_residual_blocks=layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(block=block, num_residual_blocks=layers[3], out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels * 4, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * 4)
            )

        layers.append(block(
            in_channels=self.in_channels, out_channels=out_channels,
            identity_downsample=identity_downsample, stride=stride))
        self.in_channels = out_channels * 4

        for i in range(num_residual_blocks - 1):
            layers.append(block(in_channels=self.in_channels, out_channels=out_channels))

        return nn.Sequential(*layers)


class DataSet(Dataset):
    def __init__(self, data_df: pd.DataFrame, augs: A.Compose):
        self.data_df = data_df
        self.augs = augs
        # self.resize = A.Resize(height=HEIGHT, width=WIDTH, p=1.0)
        self.random_crop = A.RandomCrop(height=HEIGHT, width=WIDTH, p=1.0)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        # - Get the data of the current sample
        img_fl, scale, pdl1, pdl2 = self.data_df.loc[index, ['path', 'scale', 'pdl1', 'pdl2']].values.flatten()

        # - Get the image
        img = get_image(image_file=img_fl)

        # - Get the label
        lbl = np.array([pdl1, pdl2])

        # - Run regular augmentations on the cropped / rescaled image
        img = self.augs(image=img, mask=img).get('image')

        img = np.expand_dims(img[:, :, 0], -1)

        img, lbl = torch.tensor(img, dtype=torch.float), torch.tensor(lbl, dtype=torch.float)

        img = torch.permute(img, (2, 0, 1))

        return img, lbl

# - Util functions
def get_resnet50(image_channels=3, num_classes=1000):
    return ResNet(block=ResBlock, layers=[3, 4, 6, 3], image_channels=image_channels, num_classes=num_classes)


def get_resnet101(image_channels=3, num_classes=1000):
    return ResNet(block=ResBlock, layers=[3, 4, 23, 3], image_channels=image_channels, num_classes=num_classes)


def get_resnet152(image_channels=3, num_classes=1000):
    return ResNet(block=ResBlock, layers=[3, 8, 36, 3], image_channels=image_channels, num_classes=num_classes)


def get_image(image_file: str or pathlib.Path):
    img = cv2.imread(str(image_file), cv2.IMREAD_UNCHANGED)
    return img


def get_name_scale_index_type(data_file: str):
    fl = data_file[::-1]
    fl_type = fl[:fl.index('.')][::-1]
    name_scale_idx = fl[fl.index('.') + 1:]
    name = name_scale_idx[name_scale_idx.index('.') + 1:]
    name = name[name.index('_') + 1:][::-1]
    idx = 1
    scale = name_scale_idx[:name_scale_idx.index('_')][::-1]
    if scale[-1] != 'x':
        idx = int(name_scale_idx[:name_scale_idx.index('_')][::-1])
        name_scale_idx = name_scale_idx[name_scale_idx.index('_') + 1:]

    scale = float(name_scale_idx[name_scale_idx.index('x') + 1:name_scale_idx.index('_')][::-1])

    return name, scale, idx, fl_type


def get_train_augs():
    return A.Compose(
        [
            A.ToGray(p=1.0),
            A.OneOf([
                A.RandomRotate90(),
                # A.RandomBrightnessContrast(),
                # A.GaussNoise(),
                A.VerticalFlip(),
                A.HorizontalFlip(),
            ], p=0.5),
            A.RandomCrop(
                height=HEIGHT,
                width=WIDTH,
                p=1.0
            )
        ], p=1.0)


def get_test_augs():
    return A.Compose(
        [
            A.ToGray(p=1.0),
            A.OneOf([
                A.RandomRotate90(),
                # A.RandomBrightnessContrast(),
                # A.GaussNoise(),
                A.VerticalFlip(),
                A.HorizontalFlip(),
            ], p=0.5),
            A.RandomCrop(
                height=HEIGHT,
                width=WIDTH,
                p=1.0
            )
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


def get_data(data_root_dir: pathlib.Path or str, metadata_file: pathlib.Path or str):
    metadata_df = pd.read_csv(metadata_file)
    lbls_df = metadata_df.loc[7:35, ['Path number', 'PD1 score', 'PDL1 score', 'PDL2 score']]

    data_df = pd.DataFrame(columns=['path', 'name', 'scale', 'index', 'type', 'pdl1', 'pdl2'])
    for root, folders, _ in os.walk(data_root_dir, topdown=False):
        for folder in folders:
            for sub_root, sub_folders, files in os.walk(pathlib.Path(root) / folder):
                if DEBUG:
                    print(f'> Getting data from {sub_root} ...\n')
                for file in tqdm(files):
                    file_name, file_scale, file_index, file_type = get_name_scale_index_type(data_file=file)
                    values = lbls_df.loc[lbls_df.loc[:, 'Path number'] == file_name, ['PDL1 score', 'PDL2 score']]. \
                        values. \
                        flatten()
                    if len(values) > 1:
                        pdl1, pdl2 = values
                        if DEBUG:
                            print(f'''
                            File name: {file_name}/{file_index}
                            - File scale: {file_scale}
                            - PDL1 score: {pdl1}
                            - PDL2 score: {pdl2}
                            ''')
                        file_data_df = pd.DataFrame(
                            dict(
                                path=f'{sub_root}/{file}',
                                name=file_name,
                                scale=file_scale,
                                index=file_index,
                                type=file_type,
                                pdl1=pdl1,
                                pdl2=pdl2
                            ),
                            index=pd.Index([0])
                        )
                        data_df = pd.concat([data_df, file_data_df], axis=0, ignore_index=True)
                    else:
                        if DEBUG:
                            print(f'(WARNING) No PDL1 / PDL2 values were found for file {sub_root}/{file}!')
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


# - Hyperparameters
# -- General
DEBUG = False
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# -- Paths
# DATA_PATH = pathlib.Path('C:/Users/Michael/Desktop/University/PhD/Projects/CancerDet/Cancer Dataset')  # Windows
# DATA_PATH = pathlib.Path('/Users/mchlsdrv/Desktop/University/PhD/Projects/CancerDet/data')  # Mac
DATA_PATH = pathlib.Path('/home/sidorov/projects/cancer_det/data')  # 4GPUs
METADATA_FILE = DATA_PATH / 'Rambam clinical table 26.6.23.csv'
OUTPUT_DIR = pathlib.Path('/media/oldrrtammyfs/Users/sidorov/CancerDet/output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -- Architecture
HEIGHT = 512
WIDTH = 512
CHANNELS = 1
# CHANNELS = 3
CLASSES = 2
MODEL = get_resnet50(image_channels=CHANNELS, num_classes=CLASSES).to(DEVICE)

# -- Training
BATCH_SIZE = 16
VAL_BATCH_SIZE_SCALE = 4
VAL_PROP = 0.2
LEARNING_RATE = 0.001
EPOCHS = 1500
N_WORKERS = 4
PIN_MEMORY = True

OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)
LOSS_FUNC = nn.MSELoss()

LOSS_PLOT_RESOLUTION = 10

# - Main
if __name__ == '__main__':
    print(f'- Getting data')
    train_data_frame = get_data(data_root_dir=DATA_PATH, metadata_file=METADATA_FILE)

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

            loss_plot_start_idx += LOSS_PLOT_RESOLUTION
            loss_plot_end_idx += LOSS_PLOT_RESOLUTION
