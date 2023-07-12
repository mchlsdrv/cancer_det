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


def get_name_magnification_index_type(file: str):
    fl = file[::-1]
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


data_df = pd.DataFrame(columns=['name', 'magnification', 'index', 'type', 'pdl1', 'pdl2'])
for root, folders, _ in os.walk(DATA_PATH, topdown=False):
    for folder in folders:
        for sub_root, sub_folders, files in os.walk(pathlib.Path(root) / folder):
            for file in files:
                file_name, file_magnification, file_index, file_type = get_name_magnification_index_type(file=file)
                values = lbls_df.loc[lbls_df.loc[:, 'Path number'] == file_name, ['PDL1 score', 'PDL2 score']].values.flatten()
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
                    data_df = data_df.append(file_data_df, ignore_index=True)
                else:
                    print(f'(WARNING) No PDL1 / PDL2 values were found for file {sub_root}/{file}!')

print(data_df)
