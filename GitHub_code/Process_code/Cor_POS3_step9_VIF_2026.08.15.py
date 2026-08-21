import os
import glob
import datetime
import sys

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.basemap import Basemap
from osgeo import gdal
import numpy as np
from joblib import parallel_backend
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import pearsonr
from joblib import Parallel, delayed
import matplotlib as mpl
from matplotlib import colormaps
from sklearn.linear_model import TheilSenRegressor
from scipy.stats import linregress
from scipy import stats
from scipy.stats import mode


import pymannkendall as mk


def get_band(tif_file, stack):
    tif = gdal.Open(tif_file)
    data = tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
    stack.append(data)

############################ 1  #############################

scale = 55

input_path = 'D:\CAU\phenology_swc_vpd\Global_test4\Data\Climate'

folder_cor_1 = fr'{input_path}\Correlation(SM_VPD_pearson)17_8_2'  #(POS-30) - POS

folder_sm_1 = fr'{input_path}\SM_preseason_mean12'  #(POS-30) - POS

folder_vpd_1 = fr'{input_path}\VPD_preseason_mean12'  #(POS-30) - POS

folder_ta_1 = fr'{input_path}\Ta_preseason_mean12'

folder_pre_1 = fr'{input_path}\Pre_preseason_sum12'

folder_srad_1 = fr'{input_path}\Srad_preseason_sum12'

tif_files_cor_1 = sorted(glob.glob(os.path.join(folder_cor_1, '*.tif')))

tif_files_sm_1 = sorted(glob.glob(os.path.join(folder_sm_1, '*.tif')))

tif_files_vpd_1 = sorted(glob.glob(os.path.join(folder_vpd_1, '*.tif')))

tif_files_ta_1 = sorted(glob.glob(os.path.join(folder_ta_1, '*.tif')))

tif_files_pre_1 = sorted(glob.glob(os.path.join(folder_pre_1, '*.tif')))

tif_files_srad_1 = sorted(glob.glob(os.path.join(folder_srad_1, '*.tif')))

############################ 2 Stack #############################
print('All stack start!')

## 数据堆叠
cor_stack_1 = []

sm_stack_1 = []

vpd_stack_1 = []

ta_stack_1 = []

pre_stack_1 = []

srad_stack_1 = []


for tif_file in tif_files_cor_1:
    get_band(tif_file, cor_stack_1)

for tif_file in tif_files_sm_1:
    get_band(tif_file, sm_stack_1)

for tif_file in tif_files_vpd_1:
    get_band(tif_file, vpd_stack_1)

for tif_file in tif_files_ta_1:
    get_band(tif_file, ta_stack_1)

for tif_file in tif_files_pre_1:
    get_band(tif_file, pre_stack_1)

for tif_file in tif_files_srad_1:
    get_band(tif_file, srad_stack_1)


cor_stack_1 = np.stack(cor_stack_1, axis=0)#[:, 505:510, 505:510]

sm_stack_1 = np.stack(sm_stack_1, axis=0)

vpd_stack_1 = np.stack(vpd_stack_1, axis=0)

ta_stack_1 = np.stack(ta_stack_1, axis=0)

pre_stack_1 = np.stack(pre_stack_1, axis=0)

srad_stack_1 = np.stack(srad_stack_1, axis=0)



############################ 3 VIF ##########################
from statsmodels.stats.outliers_influence import variance_inflation_factor


data = pd.DataFrame({
    'Cor': cor_stack_1.ravel(),

    'SM': sm_stack_1.ravel(),

    'VPD': vpd_stack_1.ravel(),

    'Ta': ta_stack_1.ravel(),

    'Pre': pre_stack_1.ravel(),

    'Srad': srad_stack_1.ravel()

})

# ==========================================
# Remove invalid values
# ==========================================

data = data.replace(
    [9999, -9999, 65535],
    np.nan
)

data = data.dropna()

print("有效观测数:", len(data))

# ==========================================
# Calculate VIF
# ==========================================

vif_data = pd.DataFrame()

vif_data["Variable"] = data.columns

vif_data["VIF"] = [
    variance_inflation_factor(
        data.values,
        i
    )
    for i in range(data.shape[1])
]

print('Global VIF:', vif_data)