import os
import pathlib
import pandas as pd
import albumentations as A
import torch
from torch.utils.data import Dataset, DataLoader


DATA_PATH = pathlib.Path('C:/Users/Michael/Desktop/University/PhD/Projects/CancerDet/Cancer Dataset')
METADATA_FILE = DATA_PATH / 'Rambam clinical table 26.6.23.csv'


metadata_df = pd.read_csv(METADATA_FILE)
lbls_df = metadata_df.loc[7:35, ['Path number', 'PD1 score', 'PDL1 score', 'PDL2 score']]
print(lbls_df)


def get_file_name_type_magnification(file: str):
    fl = file[::-1]
    fl_type = fl[:fl.index('.')][::-1]
    fl_name_mag = fl[fl.index('.') + 1:]
    fl_name = fl_name_mag[fl_name_mag.index('.') + 1:]
    fl_name = fl_name[fl_name.index('_') + 1:][::-1]
    fl_mag = fl_name_mag[:fl_name_mag.index('_')][::-1]

    return fl_name, fl_type, fl_mag


for root, folders, _ in os.walk(DATA_PATH, topdown=False):
    for folder in folders:
        for sub_root, sub_folders, files in os.walk(pathlib.Path(root) / folder):
            for file in files:
                file_name, file_type, file_magnification = get_file_name_type_magnification(file=file)
                pdl1, pdl2 = lbls_df.loc[lbls_df.loc[:, 'Path number'] == file_name, ['PDL1 score', 'PDL2 score']].values.flatten()
                print(f'''
                File name: {file_name}
                - File magnification: {file_magnification}
                - PDL1 score: {pdl1}
                - PDL2 score: {pdl2}
                ''')

