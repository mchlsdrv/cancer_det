import os
import pathlib

import numpy as np
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator
import cv2
from tqdm import tqdm

DATA_PATH = pathlib.Path('/Users/mchlsdrv/Desktop/University/PhD/Projects/CancerDet/data/raw')
SAVE_PATH = pathlib.Path('/Users/mchlsdrv/Desktop/University/PhD/Projects/CancerDet/data/train')
SAVE_FILE_TYPE = 'tif'
os.makedirs(SAVE_PATH, exist_ok=True)

CROP_SIZE = 256
OVERLAP = 0


def get_file_type(file: str):
    fl_name = ''
    fl_type = ''
    if '.' in file:
        fl_rvrs = file[::-1]
        fl_name = fl_rvrs[fl_rvrs.index('.')+1:][::-1]
        fl_type = fl_rvrs[:fl_rvrs.index('.')][::-1]
    return fl_name, fl_type


def create_data(data_root_dir: pathlib.Path, save_dir: pathlib.Path):
    files = os.listdir(data_root_dir)
    for idx, file in enumerate(files):
        print(f'> Working on file {file} {idx+1}/{len(files)} ')
        print('\t|' + ('=' * (idx + 1)) + f'|{((idx+1) / len(files)) * 100:.3f}%')
        try:
            fl_name, fl_type = get_file_type(file=file)
            if fl_type == 'mrxs':
                sld = open_slide(data_root_dir / file)
                tiles = DeepZoomGenerator(sld, tile_size=CROP_SIZE, overlap=OVERLAP, limit_bounds=False)

                max_res_lvl = len(tiles.level_tiles) - 1
                print(f'The slide has {max_res_lvl} levels')
                print(f'''
                Level {max_res_lvl}
                    - Crop shape: {tiles.level_tiles[max_res_lvl]}
                    - Number of crops: {tiles.level_tiles[max_res_lvl][0] * tiles.level_tiles[max_res_lvl][1]}
                ''')

                cols, rows = tiles.level_tiles[max_res_lvl - 1]
                for col in tqdm(range(cols)):
                    for row in range(rows):

                        save_file = save_dir / f'{fl_name}_{col}_{row}.{SAVE_FILE_TYPE}'
                        tile = tiles.get_tile(max_res_lvl, (col, row))
                        tile_rgb = tile.convert('RGB')
                        tile_np = np.array(tile_rgb)
                        if tile_np.mean() < 230 and tile_np.std() > 15:
                            cv2.imwrite(str(save_file), tile_np)
        except Exception as err:
            print(err)


create_data(data_root_dir=DATA_PATH, save_dir=SAVE_PATH)
