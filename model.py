import os
import pathlib
from PIL import Image
import pandas as pd
import albumentations as A
import torch
from torch.utils.data import Dataset, DataLoader

DATA_PATH = pathlib.Path('/Users/mchlsdrv/Desktop/University/PhD/Projects/CancerDet/data')
METADATA_FILE = DATA_PATH / 'Rambam clinical table 26.6.23.csv'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def augment():
    return A.Compose(
        [
            A.OneOf([
            A.RandomRotate90(),
            A.RandomContrast(),
            A.GaussNoise(),
            A.VerticalFlip(),
            A.HorizontalFlip(),
            ], p=0.5)
        ], p=0.5)


class ImageDS(Dataset):
    def __init__(self, data_file: pd.DataFrame, augmentations: A.Compose):
        self.data_tuples = data_tuples
        self.augs = augmentations

    def __len__(self):
        return len(self.data_tuples)

    def __getitem__(self, index):
        img, gt_mask, aug_mask, jaccard = self.data_tuples[index]
        aug_res = self.augs(image=img.astype(np.uint8), mask=aug_mask)
        img, mask = aug_res.get('image'), aug_res.get('mask')
        img, mask = np.expand_dims(img, 0), np.expand_dims(mask, 0)

        return torch.tensor(img, dtype=torch.float), \
            torch.tensor(mask.astype(np.int16), dtype=torch.float), \
            torch.tensor(jaccard, dtype=torch.float)


def get_data_loaders(data_file: str or pathlib.Path, batch_size, train_augs, validation_augmentations,
                     validation_proportion, validation_batch_size, logger: logging.Logger = None):
    # - Load the data
    data = np.load(str(data_file), allow_pickle=True)

    # - Split data into train / validation datasets
    train_data, val_data = get_train_val_split(
        data_list=data, validation_proportion=validation_proportion, logger=logger)

    # - Create the train / validation dataloaders
    train_dl = DataLoader(
        ImageDS(data_tuples=train_data, augs=train_augs()),
        batch_size=batch_size,
        num_workers=NUM_TRAIN_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True
    )

    val_dl = DataLoader(
        ImageDS(data_tuples=val_data, augs=validation_augmentations()),
        batch_size=validation_batch_size,
        num_workers=NUM_VAL_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False
    )

    return train_dl, val_dl
def load_image(image_file: str or pathlib.Path, device: torch.device):
    img = Image.open(str(image_file))
    aug = augment()
    img = aug(img).unsqueeze(0)
    return img.to(device)


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


metadata_df = pd.read_csv(METADATA_FILE)
lbls_df = metadata_df.loc[7:35, ['Path number', 'PD1 score', 'PDL1 score', 'PDL2 score']]


data_df = pd.DataFrame(columns=['name', 'magnification', 'index', 'type', 'pdl1', 'pdl2'])
for root, folders, _ in os.walk(DATA_PATH, topdown=False):
    for folder in folders:
        for sub_root, sub_folders, files in os.walk(pathlib.Path(root) / folder):
            for file in files:
                file_name, file_magnification, file_index, file_type = get_name_magnification_index_type(data_file=file)
                values = lbls_df.loc[lbls_df.loc[:, 'Path number'] == file_name, ['PDL1 score', 'PDL2 score']].\
                    values.\
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

print(data_df)


class CancerDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

    def __getitem__(self, item):
