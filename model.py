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

# - Hyperparameters
# DATA_PATH = pathlib.Path('C:/Users/Michael/Desktop/University/PhD/Projects/CancerDet/Cancer Dataset')  # Windows
DATA_PATH = pathlib.Path('/Users/mchlsdrv/Desktop/University/PhD/Projects/CancerDet/data')  # Mac
METADATA_FILE = DATA_PATH / 'Rambam clinical table 26.6.23.csv'
OUTPUT_DIR = pathlib.Path('/Users/mchlsdrv/Desktop/University/PhD/Projects/CancerDet/output')
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

HEIGHT = 512
WIDTH = 512
N_WORKERS = 4
PIN_MEMORY = True
BATCH_SIZE = 16
VAL_BATCH_SIZE_SCALE = 16
VAL_PROP = 0.2


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


def get_resnet50(image_channels=3, num_classes=1000):
    return ResNet(block=ResBlock, layers=[3, 4, 6, 3], image_channels=image_channels, num_classes=num_classes)


def get_resnet101(image_channels=3, num_classes=1000):
    return ResNet(block=ResBlock, layers=[3, 4, 23, 3], image_channels=image_channels, num_classes=num_classes)


def get_resnet152(image_channels=3, num_classes=1000):
    return ResNet(block=ResBlock, layers=[3, 8, 36, 3], image_channels=image_channels, num_classes=num_classes)


def test_resnet50():
    net = get_resnet50()
    x = torch.randn(2, 3, 224, 224)
    y = net(x).to('cuda')
    print(y.shape)


def test_resnet101():
    net = get_resnet101()
    x = torch.randn(2, 3, 224, 224)
    y = net(x).to('cuda')
    print(y.shape)


def test_resnet152():
    net = get_resnet152()
    x = torch.randn(2, 3, 224, 224)
    y = net(x).to('cuda')
    print(y.shape)


test_resnet50()
test_resnet101()
test_resnet152()


class DataSet(Dataset):
    def __init__(self, data: pd.DataFrame, augs: A.Compose):
        self.data_df = data
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

        # print(scale)
        # if scale <= 2:
        #     # - If the image is zoomed-out - resize it to be able to fit into the network as is
        #     img = self.resize(image=img, mask=img).get('image')
        # else:
        #     # - If the image is zoomed in more than x2 - crop a random crop
        #     img = self.random_crop(image=img, mask=img).get('image')

        # - Run regular augmentations on the cropped / rescaled image
        img = self.augs(image=img, mask=img).get('image')

        return torch.tensor(img, dtype=torch.float), torch.tensor(lbl, dtype=torch.float)


# - Util functions
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
            A.OneOf([
                A.RandomRotate90(),
                A.RandomBrightnessContrast(),
                A.GaussNoise(),
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
            A.OneOf([
                A.RandomRotate90(),
                A.RandomBrightnessContrast(),
                A.GaussNoise(),
                A.VerticalFlip(),
                A.HorizontalFlip(),
            ], p=0.5),
            A.RandomCrop(
                height=HEIGHT,
                width=WIDTH,
                p=1.0
            )
        ], p=1.0)


def get_train_val_split(data: pd.DataFrame, val_prop: float = .2):
    n_items = len(data)
    item_idxs = np.arange(n_items)
    n_val_items = int(n_items * val_prop)

    # - Randomly pick the validation items' indices
    val_idxs = np.random.choice(item_idxs, n_val_items, replace=False)

    # - Pick the items for the validation set
    val_data = data.loc[val_idxs, :].reset_index(drop=True)

    # - The items for training are the once which are not included in the
    # validation set
    train_data = data.loc[np.setdiff1d(item_idxs, val_idxs), :].reset_index(drop=True)

    return train_data, val_data


def get_data_loaders(data: pd.DataFrame, batch_size: int, val_prop: float):
    # - Split data into train / validation datasets
    train_data, val_data = get_train_val_split(data=data, val_prop=val_prop)

    # - Create the train / validation dataloaders
    train_dl = DataLoader(
        DataSet(data=train_data, augs=get_train_augs()),
        batch_size=batch_size,
        num_workers=N_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True
    )

    val_dl = DataLoader(
        DataSet(data=val_data, augs=get_test_augs()),
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
                print(f'> Getting data from {sub_root} ...\n')
                for file in tqdm(files):
                    file_name, file_scale, file_index, file_type = get_name_scale_index_type(data_file=file)
                    values = lbls_df.loc[lbls_df.loc[:, 'Path number'] == file_name, ['PDL1 score', 'PDL2 score']]. \
                        values. \
                        flatten()
                    if len(values) > 1:
                        pdl1, pdl2 = values
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
                        print(f'(WARNING) No PDL1 / PDL2 values were found for file {sub_root}/{file}!')
    return data_df


# - Main
if __name__ == '__main__':
    print(f'- Getting data')
    data_df = get_data(data_root_dir=DATA_PATH, metadata_file=METADATA_FILE)
    print(data_df)

    print(f'- Getting sample batches')
    train_dl, val_dl = get_data_loaders(data=data_df, batch_size=BATCH_SIZE, val_prop=VAL_PROP)
    train_imgs, train_lbls = next(iter(train_dl))
    val_imgs, val_lbls = next(iter(val_dl))
    print(f'''
    - type(train_imgs): {type(train_imgs)}
    - train_lbls.shape: {train_lbls.shape}
    ''')
    print(f'''
    - type(val_imgs): {type(val_imgs)}
    - val_lbls.shape: {val_lbls.shape}
    ''')

    # - Hyperparameters
    learning_rate = 0.001
    epochs = 10

    model = get_resnet50(image_channels=3, num_classes=2).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.MSELoss()

    epoch_train_losses = np.array([])
    epoch_val_losses = np.array([])
    # - Training loop
    for epoch in tqdm(range(epochs)):
        # - Train
        btch_train_losses = np.array([])
        btch_val_losses = np.array([])
        for btch_idx, (imgs, lbls) in enumerate(train_dl):
            # - Store the data in CUDA
            imgs = imgs.to(DEVICE)
            lbls = lbls.to(DEVICE)

            # - Forward
            preds = model(imgs)
            loss = loss_func(preds, lbls)
            btch_train_losses = np.append(btch_train_losses, loss.item())

            # - Backward
            opt.zero_grad()
            loss.backward()

            # - Optimizer step
            opt.step()

        # - Validation
        with torch.no_grad():
            for btch_idx, (imgs, lbls) in enumerate(val_dl):
                imgs = imgs.to(DEVICE)
                lbls = lbls.to(DEVICE)

                preds = model(imgs)
                loss = loss_func(preds, lbls)

                btch_val_losses = np.append(btch_val_losses, loss.item())

        epoch_train_losses = np.append(epoch_train_losses, btch_train_losses.mean())
        epoch_val_losses = np.append(epoch_val_losses, btch_val_losses.mean())

        fig, ax = plt.subplots()
        ax.plot(np.arange(epoch + 1), epoch_train_losses, label='train')
        ax.plot(np.arange(epoch + 1), epoch_val_losses, label='val')
        ax.set(xlabel='Epoch', ylabel='MSE')
        fig.suptitle('Loss vs Epochs')
        plt.legend(True)
        plt.savefig(OUTPUT_DIR / 'loss.png')
        plt.close(fig)
