import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
plt.style.use('seaborn')

DATA_PATH = Path('/home/sidorov/projects/cancer_det/data/filter_data')


low_sig_imgs = os.listdir(DATA_PATH / 'low_signal')
high_sig_imgs = os.listdir(DATA_PATH / 'high_signal')


ls_img = cv2.imread(str(DATA_PATH / 'low_signal' / low_sig_imgs[0]), cv2.IMREAD_GRAYSCALE)
ls_img_flt = np.reshape(ls_img, -1)
hs_img = cv2.imread(str(DATA_PATH / 'high_signal' / high_sig_imgs[0]), cv2.IMREAD_GRAYSCALE)
hs_img_flt = np.reshape(hs_img, -1)

imgs = np.append(ls_img_flt, hs_img_flt)
low_sig_img_lbl = ['low signal'] * len(ls_img_flt)
high_sig_img_lbl = ['high signal'] * len(hs_img_flt)
lbls = low_sig_img_lbl + high_sig_img_lbl
img_df = pd.DataFrame(dict(img_intensity=imgs, img=lbls))
fig, ax = plt.subplots(figsize=(8, 8))

ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=1)
ax2 = plt.subplot2grid((4, 4), (0, 1), colspan=3)
ax3 = plt.subplot2grid((4, 4), (1, 0), colspan=3)

ax1.imshow(ls_img, cmap='gray')
ax1.set(title='Low Signal Image')

ax2.imshow(hs_img, cmap='gray')
ax2.set(title='Low Signal Image')

sns.displot(img_df, x='img_intensity', hue='img', kind='kde', ax=ax3)
plt.show()
