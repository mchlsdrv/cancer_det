import os
import pathlib
from PIL import Image
import numpy as np
import pandas as pd
import albumentations as A
import torch
from torch.utils.data import Dataset, DataLoader

# - Hyperparameters
DATA_PATH = pathlib.Path('C:/Users/Michael/Desktop/University/PhD/Projects/CancerDet/Cancer Dataset')  # Windows
# DATA_PATH = pathlib.Path('/Users/mchlsdrv/Desktop/University/PhD/Projects/CancerDet/data')  # Mac
METADATA_FILE = DATA_PATH / 'Rambam clinical table 26.6.23.csv'

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

HEIGHT = 512
WIDTH = 512
N_WORKERS = 4
PIN_MEMORY = True
BATCH_SIZE = 16
VAL_BATCH_SIZE_SCALE = 16
VAL_PROP = 0.2


# - Classes
class DataSet(Dataset):
    def __init__(self, data: pd.DataFrame, augs: A.Compose):
        self.data_df = data
        self.augs = augs

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        img_fl, pdl1, pdl2 = self.data_df.loc[index, ['path', 'pdl1', 'pdl2']].values.flatten()
        img = get_image(image_file=img_fl)
        aug_res = self.augs(image=img, mask=img)
        img = aug_res.get('image')
        img = np.expand_dims(img, 0)

        return \
            torch.tensor(img, dtype=torch.float), \
            torch.tensor(pdl1, dtype=torch.float), \
            torch.tensor(pdl2, dtype=torch.float)


# - Util functions
def get_image(image_file: str or pathlib.Path):
    img = Image.open(str(image_file))

    return np.asarray(img)


def get_name_magnification_index_type(data_file: str):
    fl = data_file[::-1]
    fl_type = fl[:fl.index('.')][::-1]
    name_mag_idx = fl[fl.index('.') + 1:]
    name = name_mag_idx[name_mag_idx.index('.') + 1:]
    name = name[name.index('_') + 1:][::-1]
    idx = 1
    mag = name_mag_idx[:name_mag_idx.index('_')][::-1]
    if mag[-1] != 'x':
        idx = name_mag_idx[:name_mag_idx.index('_')][::-1]
        name_mag_idx = name_mag_idx[name_mag_idx.index('_') + 1:]
        mag = name_mag_idx[:name_mag_idx.index('_')][::-1]

    return name, mag, idx, fl_type


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

    data_df = pd.DataFrame(columns=['path', 'name', 'magnification', 'index', 'type', 'pdl1', 'pdl2'])
    for root, folders, _ in os.walk(data_root_dir, topdown=False):
        for folder in folders:
            for sub_root, sub_folders, files in os.walk(pathlib.Path(root) / folder):
                for file in files:
                    file_name, file_magnification, file_index, file_type = get_name_magnification_index_type(data_file=file)
                    values = lbls_df.loc[lbls_df.loc[:, 'Path number'] == file_name, ['PDL1 score', 'PDL2 score']]. \
                        values. \
                        flatten()
                    if len(values) > 1:
                        pdl1, pdl2 = values
                        print(f'''
                        File name: {file_name}/{file_index}
                        - File magnification: {file_magnification}
                        - PDL1 score: {pdl1}
                        - PDL2 score: {pdl2}
                        ''')
                        file_data_df = pd.DataFrame(
                            dict(
                                path=f'{sub_root}/{file}',
                                name=file_name,
                                magnification=file_magnification,
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
    data_df = get_data(data_root_dir=DATA_PATH, metadata_file=METADATA_FILE)
    print(data_df)

    train_dl, val_dl = get_data_loaders(data=data_df, batch_size=BATCH_SIZE, val_prop=VAL_PROP)
    train_batch = next(iter(train_dl))
    val_batch = next(iter(val_dl))
    print(f'''
    - type(train_batch): {type(train_batch)}
    - train_batch.shape: {np.array(train_batch).shape}
    ''')
    print(f'''
    - type(val_batch): {type(val_batch)}
    - val_batch.shape: {np.array(val_batch).shape}
    ''')
