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
    get_device, train, get_data_loaders_from_datasets, save_images, plot_scatter
)
from python_utils.image_utils import get_image

plt.style.use('ggplot')

# - Constants
EPSILON = 1e-9
BETA = 1.0
ERROR_THRESHOLD = 0.1


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
CHECKPOINT_FILE = '/media/oldrrtammyfs/Users/sidorov/CancerDet/output/MSELoss_train_test_all_data_reg_db_2023-08-30_12-40-28/checkpoints/weights_epoch_99.pth.tar'
# CHECKPOINT_FILE = '/media/oldrrtammyfs/Users/sidorov/CancerDet/output/MSELoss_train_all_data_cont_2023-08-28_13-56-46/checkpoints/weights_epoch_99.pth.tar'
# CHECKPOINT_FILE = '/media/oldrrtammyfs/Users/sidorov/CancerDet/output/CELoss_train_no_stoch_depth_gray_all_data_2023-08-25_23-27-22/checkpoints/weights_epoch_89.pth.tar'
# CHECKPOINT_FILE = '/media/oldrrtammyfs/Users/sidorov/CancerDet/output/CELoss_train_no_stoch_depth_gray_50k_2023-08-25_07-46-10/checkpoints/weights_epoch_49.pth.tar'
# -- Architecture
HEIGHT = 256
WIDTH = 256
# CHANNELS = 3
CHANNELS = 3 if ARGS.rgb else 1
OUTPUT_SIZE = 1
# OUTPUT_SIZE = 2

# -- Training
# > Hyperparameters
# N_DATA_SAMPLES = 50000
N_DATA_SAMPLES = -1
TRAIN_BATCH_SIZE = ARGS.batch_size
VAL_BATCH_SIZE = TRAIN_BATCH_SIZE // 2
LEARNING_RATE = ARGS.learning_rate
EPOCHS = ARGS.epochs
TO_RGB = True if ARGS.rgb else False
TO_GRAY = True if not ARGS.rgb else False
VAL_PROP = 0.2
N_WORKERS = 4
PIN_MEMORY = True
CHANNELS_FIRST = True

# > Loss
# * Choosing the model - binary classification or regression
if ARGS.model_type == 'bin':
    CLASSIFICATION_MODEL = True
    REGRESSION_MODEL = False
else:
    REGRESSION_MODEL = True
    CLASSIFICATION_MODEL = False

if CLASSIFICATION_MODEL:
    LOSS_FUNC = nn.CrossEntropyLoss()
else:
    LOSS_FUNC = nn.MSELoss()

MODEL = get_resnet50(
    image_channels=CHANNELS,
    output_size=OUTPUT_SIZE,
    prediction_layer=None,
).to(DEVICE)

# > Optimizer
OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)


# > Plots
LOSS_PLOT_RESOLUTION = 10


class DataSet(Dataset):
    def __init__(self, data_df: pd.DataFrame, augs: A.Compose, to_gray: bool,
                 model_type: str, label_length: int = 2, channels_first: bool = False):
        self.data_df = data_df
        self.label_length = label_length
        self.augs = augs
        self.random_crop = A.RandomCrop(height=HEIGHT, width=WIDTH, p=1.0)
        self.to_gray = to_gray
        self.model_type = model_type
        self.channels_first = channels_first

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        # - Get the data of the current sample
        img_fl, pd1, pdl1, pdl2, rspns = self.data_df.loc[
            index, ['path', 'pd1', 'pdl1', 'pdl2', 'response']
        ].values.flatten()

        # - Get the image
        img = get_image(image_file=img_fl, to_gray=self.to_gray)

        # - Get the label
        # - Use continuous labels representing the values for regression model
        if self.model_type == 'reg':
            if self.label_length == 1:
                lbl = torch.tensor([pd1], dtype=torch.float)
            else:
                lbl = torch.tensor(np.array([pdl1, pdl2]), dtype=torch.float)
        else:
            lbl = torch.tensor(np.array(rspns), dtype=torch.long)

        # - Run regular augmentations on the cropped / rescaled image
        img = self.augs(image=img, mask=img).get('image')

        # - Convert image to tensor
        img = torch.tensor(img, dtype=torch.float)

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


def get_data_from_dir(data_root_dir: pathlib.Path or str, metadata_file: pathlib.Path or str,
                      save_file: pathlib.Path or str):
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
    if N_DATA_SAMPLES > 0:
        train_df, val_df = get_train_val_split(data_df=train_df.loc[:N_DATA_SAMPLES, :], val_prop=VAL_PROP)
    else:
        train_df, val_df = get_train_val_split(data_df=train_df, val_prop=VAL_PROP)
    print(f'''
    - Training on {len(train_df)} samples
    - Validating on {len(val_df)} samples
    ''')

    train_ds = DataSet(
        data_df=train_df,
        augs=get_train_augs(),
        to_gray=TO_GRAY,
        model_type=ARGS.model_type,
        label_length=OUTPUT_SIZE,
        channels_first=CHANNELS_FIRST
    )

    val_ds = DataSet(
        data_df=val_df,
        augs=get_test_augs(),
        to_gray=TO_GRAY,
        model_type=ARGS.model_type,
        label_length=OUTPUT_SIZE,
        channels_first=CHANNELS_FIRST
    )

    train_data_loader, val_data_loader = get_data_loaders_from_datasets(
        train_dataset=train_ds,
        train_batch_size=TRAIN_BATCH_SIZE,
        val_dataset=val_ds,
        val_batch_size=VAL_BATCH_SIZE,
    )
    return train_data_loader, val_data_loader


def predict_label(image_files, continuous: bool = False):
    """
    Returns a 1D vector of length len(images) with float values representing the scores in range [0., 1.] for each image
    :param image_files: Images to be predicted
    :param continuous: If the label should be continuous or binary
    :return: Predictions (1D float vector)
    """
    # - Put the model in evaluation mode
    MODEL.eval()
    imgs = []
    img_nms = []
    preds = []
    print(f'- Predicting images')
    for img_fl in tqdm(image_files):
        img = get_image(
            image_file=img_fl,
            to_gray=TO_GRAY,
            to_tensor=True,
            channel_first=True,
            add_batch_dim=True,
            device=DEVICE
        )

        logits = MODEL(img)
        if continuous:
            pred = logits
        else:
            _, pred = torch.max(logits, 1)  # returns (values, indices), but as our class is the same as the index,
            # we may ignore the raw value

        # - Append the label to the labels
        if DEBUG:
            img = get_image(
                image_file=img_fl,
                to_gray=TO_GRAY,
                add_channel_dim=False
            )
            imgs.append(img)
            img_nm = pathlib.Path(img_fl).name
            img_nms.append(img_nm)
        preds.append(pred.cpu().detach().numpy()[0])

    # - Put the model back in train mode
    MODEL.train()

    return np.array(preds), np.array(imgs), np.array(img_nms)


def test_binary_classification():
    test_df = pd.read_csv(TEST_DATA_FRAME)
    image_files, labels = test_df.loc[:, 'path'].values, test_df.loc[:, 'response'].values
    preds, imgs, img_nms = predict_label(image_files=image_files, continuous=False)

    print(f'- Testing {len(image_files)} images')
    # - Monitor the false positive / negative samples
    if DEBUG:

        # - False positives are 1 while they should be 0, so to get to them, we reverse the label and multipy by
        # predictions e.g.,
        # > pred                   [1, 0, 0, 1, 1]
        # > label                  [0, 0, 1, 0, 1]
        # > inv(label)             [1, 1, 0, 1, 0]
        # => pred * inv(label)     [1, 0, 0, 1, 0]
        fp = preds * np.logical_not(labels)
        fp_idxs = np.argwhere(fp > 0).flatten()
        fp_imgs = imgs[fp_idxs]
        fp_img_nms = img_nms[fp_idxs]
        save_images(fp_imgs, image_names=fp_img_nms, save_dir=OUTPUT_DIR / 'errors/FP')

        # - False negatives are 0 while they should be 1, so to get to them, we reverse the predictions and multipy by
        # labels e.g.,
        # > pred                   [1, 0, 0, 1, 1]
        # > inv(pred)              [0, 1, 1, 0, 0]
        # > label                  [0, 0, 1, 0, 1]
        # => inv(pred) * label     [0, 0, 1, 0, 0]
        fn = np.logical_not(preds) * labels
        fn_idxs = np.argwhere(fn > 0).flatten()
        fn_imgs = imgs[fn_idxs]
        fn_img_nms = img_nms[fn_idxs]
        save_images(fn_imgs, image_names=fn_img_nms, save_dir=OUTPUT_DIR / 'errors/FN')

    prcsn, rcll, f1_scr, _ = sklearn.metrics.precision_recall_fscore_support(preds, labels, beta=BETA, average='binary')

    plt.style.use('default')
    conf_mat = sklearn.metrics.confusion_matrix(preds, labels) / len(image_files)
    conf_mat_disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=conf_mat)
    conf_mat_disp.plot()

    return prcsn, rcll, f1_scr, conf_mat_disp.figure_


# noinspection PyShadowingNames
def test_continuous_label():
    test_df = pd.read_csv(TEST_DATA_FRAME)
    image_files, labels_pdl1, labels_pdl2 = test_df.loc[:, 'path'].values, test_df.loc[:, 'pdl1'].values, test_df.loc[:, 'pdl2'].values
    preds, imgs, img_nms = predict_label(image_files=image_files, continuous=True)
    preds_pdl1, preds_pdl2 = preds[:, 0], preds[:, 1]

    print(f'- Testing {len(image_files)} images')
    # - Monitor the large error samples
    if DEBUG:
        # - PDL1 errors
        pdl1_sq_errs = np.sqrt(preds_pdl1 - labels_pdl1)
        large_errs_idxs = np.argwhere(pdl1_sq_errs > ERROR_THRESHOLD).flatten()
        err_imgs = imgs[large_errs_idxs]
        err_img_nms = img_nms[large_errs_idxs]
        save_images(err_imgs, image_names=err_img_nms, squared_errors=pdl1_sq_errs, save_dir=OUTPUT_DIR / 'pdl1_errors')

        # - PDL2 errors
        pdl2_sq_errs = np.sqrt(preds_pdl2 - labels_pdl2)
        large_errs_idxs = np.argwhere(pdl2_sq_errs > ERROR_THRESHOLD).flatten()
        err_imgs = imgs[large_errs_idxs]
        err_img_nms = img_nms[large_errs_idxs]
        save_images(err_imgs, image_names=err_img_nms, squared_errors=pdl2_sq_errs, save_dir=OUTPUT_DIR / 'pdl2_errors')

    scatter_fig = plot_scatter(true=[labels_pdl1, labels_pdl2], predicted=[preds_pdl1, preds_pdl2], labels=['pl1', 'pl2'])

    return scatter_fig


if __name__ == '__main__':
    CHECKPOINT_FILE = pathlib.Path(CHECKPOINT_FILE)
    if ARGS.test or ARGS.infer or (ARGS.load_weights and CHECKPOINT_FILE.is_file()):
        load_checkpoint(MODEL, CHECKPOINT_FILE)

    if ARGS.train:
        train_dl, val_dl = get_data_loaders()
        train(model=MODEL, train_data_loader=train_dl, val_data_loader=val_dl, optimizer=OPTIMIZER, loss_func=LOSS_FUNC,
              epochs=EPOCHS, device=DEVICE, output_dir=OUTPUT_DIR)

    if CLASSIFICATION_MODEL:
        precision, recall, f1_score, conf_mat_fig = test_binary_classification()

        print(f'''
        TEST RESULTS:
            - Precision (TP / (TP + FP)): {precision:.3f}
            - Recall (TP / (TP + FN)): {recall:.3f}
            - F1 Score (2 * (Precision * Recall) / (Precision + Recall)): {f1_score:.3f}
        ''')
        plot_output_dir = OUTPUT_DIR / 'plots'
        os.makedirs(plot_output_dir, exist_ok=True)
        conf_mat_fig.savefig(plot_output_dir / 'confusion matrix.png')
        plt.close(conf_mat_fig)
    elif REGRESSION_MODEL:
        scatter_fig = test_continuous_label()
        plot_output_dir = OUTPUT_DIR / 'plots'
        os.makedirs(plot_output_dir, exist_ok=True)
        scatter_fig.savefig(plot_output_dir / 'scatter plot.png')
        plt.close(scatter_fig)
