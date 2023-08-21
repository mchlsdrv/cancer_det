import datetime
import os
import sys
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import albumentations as A
import sklearn
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm

sys.path.append('/home/sidorov/dev')

from ml.nn.cnn.architectures.ResNet import get_resnet50
from ml.nn.utils.aux_funcs import (
    get_train_val_split,
    load_checkpoint,
    get_arg_parser,
    get_device, train, get_data_loaders_from_datasets
)
from python_utils.image_utils import get_image

plt.style.use('ggplot')
EPSILON = 1e-9
BETA = 1.0

TS = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
PARSER = get_arg_parser()
ARGS = PARSER.parse_args()

# - Hyperparameters
# -- General
DEBUG = ARGS.debug
DEVICE = get_device(gpu_id=ARGS.gpu_id)
# DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# -- Paths
DATA_PATH_LOCAL_ROOT = pathlib.Path('/home/sidorov/projects/cancer_det/data')  # 4GPUs

DATA_PATH_REMOTE = pathlib.Path(
    '/media/oldrrtammyfs/Users/sidorov/CancerDet/output/image_filter/filtered/valid')  # 4GPUs

# - Dataframe files
TRAIN_DATA_FRAME = DATA_PATH_LOCAL_ROOT / 'train_data_frame.csv'
TEST_DATA_FRAME = DATA_PATH_LOCAL_ROOT / 'test_data_frame.csv'

METADATA_FILE = DATA_PATH_LOCAL_ROOT / 'Table.csv'
if isinstance(ARGS.name, str):
    OUTPUT_DIR = pathlib.Path(f'/media/oldrrtammyfs/Users/sidorov/CancerDet/output/{ARGS.name}_{TS}')
else:
    OUTPUT_DIR = pathlib.Path(f'/media/oldrrtammyfs/Users/sidorov/CancerDet/output/{TS}')
os.makedirs(OUTPUT_DIR, exist_ok=True)
CHECKPOINT_FILE = pathlib.Path('/media/oldrrtammyfs/Users/sidorov/CancerDet/output/CELoss_train_test_no_act_2023-08-20_12-34-41/checkpoints/weights_epoch_99.pth.tar')

# -- Architecture
HEIGHT = 256
WIDTH = 256
CHANNELS = 1
OUTPUT_SIZE = 2

# -- Training
# > Hyperparameters
N_DATA_SAMPLES = 5000
TRAIN_BATCH_SIZE = ARGS.batch_size
VAL_BATCH_SIZE = TRAIN_BATCH_SIZE // 2
LEARNING_RATE = ARGS.learning_rate
EPOCHS = ARGS.epochs
TO_RGB = True if CHANNELS == 3 else False
TO_GRAY = True if CHANNELS == 1 else False
VAL_PROP = 0.2
N_WORKERS = 4
PIN_MEMORY = True
CHANNELS_FIRST = True

# > Loss
ONE_HOT_LABELS = True
CLASSIFICATION_MODEL = True
if CLASSIFICATION_MODEL:
    LOSS_FUNC = nn.CrossEntropyLoss() if ONE_HOT_LABELS else nn.BCELoss()
else:
    LOSS_FUNC = nn.MSELoss()

MODEL = get_resnet50(
    image_channels=CHANNELS,
    output_size=OUTPUT_SIZE,
    prediction_layer=None if ONE_HOT_LABELS else nn.Sigmoid(),
).to(DEVICE)

# > Optimizer
OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)


# > Plots
LOSS_PLOT_RESOLUTION = 10

# -- Test
POSITIVE_THRESHOLD = 0.5


class DataSet(Dataset):
    def __init__(self, data_df: pd.DataFrame, augs: A.Compose, to_gray: bool,
                 classification_model: bool = True, channels_first: bool = False):
        self.data_df = data_df
        self.augs = augs
        self.random_crop = A.RandomCrop(height=HEIGHT, width=WIDTH, p=1.0)
        self.to_gray = to_gray
        self.classification_model = classification_model
        self.channels_first = channels_first

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        # - Get the data of the current sample
        img_fl, pdl1, pdl2, rspns = self.data_df.loc[
            index, ['path', 'pdl1', 'pdl2', 'response']
        ].values.flatten()

        # - Get the image
        img = get_image(image_file=img_fl, to_gray=self.to_gray)

        # - Get the label
        if self.classification_model:
            # # - Use one-hot labels
            # if self.one_hot_labels:
            #     if rspns > 0:
            #         lbl = np.array([1., 0.])
            #     else:
            #         lbl = np.array([0., 1.])
            # # - Use continuous label in 0 or 1
            # else:
            lbl = rspns
            # lbl = np.array([rspns])
        # - Use continuous labels representing the values for regression model
        else:
            lbl = np.array([pdl1, pdl2])

        # - Run regular augmentations on the cropped / rescaled image
        img = self.augs(image=img, mask=img).get('image')

        img, lbl = torch.tensor(img, dtype=torch.float), torch.tensor(lbl, dtype=torch.long)

        if self.channels_first:
            img = torch.permute(img, (2, 0, 1))  # Add channels dim before width and height

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


def get_data_from_dir(data_root_dir: pathlib.Path or str, metadata_file: pathlib.Path or str, save_file: pathlib.Path or str):
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


# - Main
def get_data_loaders():
    print(f'- Getting data')
    if not TRAIN_DATA_FRAME.is_file():
        train_df = get_data_from_dir(
            data_root_dir=DATA_PATH_REMOTE,
            metadata_file=METADATA_FILE,
            save_file=TRAIN_DATA_FRAME
        )
    else:
        train_df = pd.read_csv(TRAIN_DATA_FRAME)

    # - Shuffle the lines
    train_df = train_df.sample(frac=1).reset_index(drop=True)

    # - Split data into train / validation datasets
    # train_df, val_df = get_train_val_split(data_df=train_df, val_prop=VAL_PROP)
    train_df, val_df = get_train_val_split(data_df=train_df.loc[:N_DATA_SAMPLES, :], val_prop=VAL_PROP)
    print(f'''
    - Training on {len(train_df)} samples
    - Validating on {len(val_df)} samples
    ''')

    train_ds = DataSet(
        data_df=train_df,
        augs=get_train_augs(),
        to_gray=TO_GRAY,
        classification_model=CLASSIFICATION_MODEL,
        channels_first=CHANNELS_FIRST
    )

    val_ds = DataSet(
        data_df=val_df,
        augs=get_test_augs(),
        to_gray=TO_GRAY,
        classification_model=CLASSIFICATION_MODEL,
        channels_first=CHANNELS_FIRST
    )

    train_data_loader, val_data_loader = get_data_loaders_from_datasets(
        train_dataset=train_ds,
        train_batch_size=TRAIN_BATCH_SIZE,
        val_dataset=val_ds,
        val_batch_size=VAL_BATCH_SIZE,
    )
    return train_data_loader, val_data_loader


def predict(image_files):
    """
    Returns a 1D vector of length len(images) with float values representing the scores in range [0., 1.] for each image
    :param image_files: Images to be predicted
    :return: Predictions (1D float vector)
    """
    # MODEL.eval()
    preds = []
    for img_fl in tqdm(image_files):
        img = get_image(
            image_file=img_fl,
            to_gray=True,
            to_tensor=True,
            channel_first=True,
            add_batch_dim=True,
            device=DEVICE
        )

        cls_probs = MODEL(img)
        _, pred = torch.max(cls_probs, 1)  # returns (values, indices), but as our class is the same as the index,
        # we may ignore the raw value

        # - Append the label to the labels
        preds.append(pred.item())

    return np.array(preds)


def test():
    test_df = pd.read_csv(TEST_DATA_FRAME)
    image_files, labels = test_df.loc[:, 'path'].values, test_df.loc[:, 'response'].values
    preds = predict(image_files=image_files)

    prcsn, rcll, f1_scr, _ = sklearn.metrics.precision_recall_fscore_support(preds, labels, beta=BETA, average='binary')

    plt.style.use('default')
    conf_mat = sklearn.metrics.confusion_matrix(preds, labels) / len(image_files)
    conf_mat_disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=conf_mat)
    conf_mat_disp.plot()

    return prcsn, rcll, f1_scr, conf_mat_disp.figure_


if __name__ == '__main__':
    if (
            ARGS.test or
            ARGS.infer or
            (ARGS.train and ARGS.load_weights and CHECKPOINT_FILE.is_file())
    ):
        load_checkpoint(MODEL, CHECKPOINT_FILE)

    if ARGS.train:
        train_dl, val_dl = get_data_loaders()
        train(model=MODEL, train_data_loader=train_dl, val_data_loader=val_dl, optimizer=OPTIMIZER, loss_func=LOSS_FUNC,
              epochs=EPOCHS, device=DEVICE, output_dir=OUTPUT_DIR)

    precision, recall, f1_score, conf_mat_fig = test()

    print(f'''
    TEST RESULTS:
        - Precision (TP / (TP + FP)): {precision:.3f}
        - Recall (TP / (TP + FN)): {recall:.3f}
        - F1 Score (2 * (Precision * Recall) / (Precision + Recall)): {f1_score:.3f}
    ''')

