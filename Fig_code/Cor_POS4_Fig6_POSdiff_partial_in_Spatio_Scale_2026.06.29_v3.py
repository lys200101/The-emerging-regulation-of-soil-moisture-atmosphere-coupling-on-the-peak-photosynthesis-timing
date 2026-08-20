
import os
import glob
import datetime
import sys

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
from duckdb.experimental.spark.sql.functions import isnan
from matplotlib.pyplot import subplot
from mpl_toolkits.basemap import Basemap
from osgeo import gdal
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
import pingouin as pg
from scipy.stats import pearsonr, alpha
from joblib import Parallel, delayed, parallel_backend
import matplotlib as mpl
from matplotlib import colormaps
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.model_selection import train_test_split
import pymannkendall as mk
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import shap
from sklearn.metrics import r2_score
import gc
from sklearn.preprocessing import StandardScaler
from joblib import parallel_backend

import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

######### Function #########
### 读取SM和VPD波段
def get_SPEIband(tif, stack, spei_length):
    tif_data = gdal.Open(tif)
    tif_array = tif_data.GetRasterBand(spei_length).ReadAsArray().astype(np.float32)
    stack.append(tif_array)
def get_band(tif_file, stack):
    tif = gdal.Open(tif_file)
    data = tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
    stack.append(data)


def save_tif_gdal(output_path, data, rows, cols, crs, transform):

        driver = gdal.GetDriverByName("GTiff")

        # 创建输出数据集（覆盖模式）
        output_ds = driver.Create(
            output_path,
            cols,  # 宽度（列数）
            rows,  # 高度（行数）
            1,  # 波段数
            gdal.GDT_Float32  # 默认数据类型（可根据需求修改）
        )
        if not output_ds:
            raise RuntimeError(f"无法创建输出文件: {output_path}")

        output_band = output_ds.GetRasterBand(1)  # 单波段索引为 1
        output_band.WriteArray(data, 0, 0)  # 写入数据（0,0 表示左上角起始）

        output_ds.SetProjection(crs)

        output_ds.SetGeoTransform(transform)

        output_ds = None  # 关闭数据集（必须！否则文件可能损坏）
        return True


def cal_partial(i, j, vars, y, pos_stack, cor_matrix, sm_matrix, vpd_matrix, ta_matrix, pre_matrix, srad_matrix):
# def cal_partial(i, j, vars, y, pos_stack, cor_matrix, sm_matrix, vpd_matrix):

    # mask = (~np.isnan(pos_stack[:, i, j]) &
    #         ~np.isnan(cor_matrix[:, i, j]) &
    #         ~np.isnan(sm_matrix[:, i, j]) &
    #         ~np.isnan(vpd_matrix[:, i, j]) &
    #         ~np.isnan(ta_matrix[:, i, j]) &
    #         ~np.isnan(pre_matrix[:, i, j]) &
    #         ~np.isnan(srad_matrix[:, i, j]))\

    all_vars = np.stack([
        pos_stack[:, i, j],
        cor_matrix[:, i, j],
        sm_matrix[:, i, j],
        vpd_matrix[:, i, j],
        ta_matrix[:, i, j],
        pre_matrix[:, i, j],
        srad_matrix[:, i, j]
    ], axis=0)
    # mask: 哪些时间点所有变量都不是 NaN
    mask = np.all(np.isfinite(all_vars), axis=0)   #如果np前加~就是返回了无效值

    if mask.sum() > 3: #统计有效值数量

        pos = np.array(pos_stack[:, i, j][mask])

        cor_time_series = np.array(cor_matrix[:, i, j][mask])
        sm_time_series = np.array(sm_matrix[:, i, j][mask])
        vpd_time_series = np.array(vpd_matrix[:, i, j][mask])
        ta_time_series = np.array(ta_matrix[:, i, j][mask])
        pre_time_series = np.array(pre_matrix[:, i, j][mask])
        srad_time_series = np.array(srad_matrix[:, i, j][mask])

        pixel_data = pd.DataFrame({
        'POS': pos,
        'cor': cor_time_series,   #cor
        'sm': sm_time_series,     #sm
        'vpd': vpd_time_series,    #vpd
        # })
        'ta': ta_time_series,   #ta
        'pre': pre_time_series, #pre
        'srad': srad_time_series #srad
        })

        # print('pixel_data:\n', pixel_data)

        # partial_coefficient[i, j] = partial_cor['r'].iloc[0].round(4)
        # partial_coefficient_p[i, j] = partial_cor['p-val'].iloc[0].round(4)

        res_list = []
        for var in vars:
            # Perform partial correlation for each variable with other variables
            res = pg.partial_corr(pixel_data, y=y, x=var, covar=[item for item in vars if item != var])
            res['var'] = var
            # print('res:\n', res)
            res_list.append(res)
        result_df = pd.concat(res_list, axis=0)
        # print('result_df:\n', result_df)

        # 排序并去重：按相关系数绝对值从大到小排序
        # top_vars = result_df.reindex(result_df['r'].abs().sort_values(ascending=False).index)
        top_vars = result_df.sort_values(by='r', key=lambda x: x.abs(), ascending=False)
        # print('top_vars:', top_vars)

        # 获取第一个变量（可用 drop_duplicates 确保不重复）
        top_vars_unique = top_vars.drop_duplicates(subset='var').head(1)

        # 显示前三大变量
        # print("Top 3 strongest partial correlations:")
        # print(top_vars_unique[['var', 'r', 'p-val']])

        varname = top_vars_unique['var'][0]
        # print('top var:', varname)
        r = top_vars_unique['r'][0]
        p = top_vars_unique['p-val'][0]

        sm_pcor = result_df.loc[result_df['var'] == 'sm', 'r'].iloc[0]
        vpd_pcor = result_df.loc[result_df['var'] == 'vpd', 'r'].iloc[0]
        cor_pcor = result_df.loc[result_df['var'] == 'cor', 'r'].iloc[0]
        ta_pcor = result_df.loc[result_df['var'] == 'ta', 'r'].iloc[0]
        pre_pcor = result_df.loc[result_df['var'] == 'pre', 'r'].iloc[0]
        srad_pcor = result_df.loc[result_df['var'] == 'srad', 'r'].iloc[0]


    else:
        varname = np.nan
        r = np.nan
        p = np.nan

        sm_pcor = np.nan
        vpd_pcor = np.nan
        cor_pcor = np.nan
        ta_pcor = np.nan
        pre_pcor = np.nan
        srad_pcor = np.nan

    return (i, j, varname, r, p, sm_pcor, vpd_pcor, cor_pcor, ta_pcor, pre_pcor, srad_pcor)



###################################### 1 数据读取及输出设定 ################################################

print(list(colormaps))


###########################  ==== 输入设定 ==== #################################
star_year = 2001
end_year = 2024
years_length = end_year - star_year + 1
print('years_length:', years_length)

y_list = ['POS drought normal diff']#, 'POS slope', 'POS std']  'POS drought normal diff' / 'POS wet normal diff' / 'POS corhigh corlow diff'

months_before_pos = 2

climate_test_number = spei_length = months_before_pos

drought_distinguish_way = 3

pheno = 'pos'

Basedon = 'Based_on_detrendPheno'  ### Based_on_detrendPheno 意思用去趋势的SOS、POS来做偏相关
                                   ### Based_on_OriginPheno 意思用原始的SOS、POS来做偏相关

Outlier = 'No'
SigCorPvalue = 'No'
analyze_by = 'All'  ## 'All'  /  'advance PPT'  /  'delay PPT'

drought_or_wet_times = 2  ### !!!重点修改
spei_strength = -1

veg_type = 'All'     ### Forest / Shrub / Savanna / Grass
ai_type = 'All'  ###  Arid / Semi-Arid / Dry sub-humid / Humid
cor_type = 'All'  ### Cor(-0.6~-0.5) / Cor(-0.5~-0.4) / Cor(-0.4~-0.3) / Cor(-0.3~-0.2)

ML_model_in_spatio = 'RF'  ##RF / XGBoost
test_size = 0.1

scale = 11

#### 输入Climate的tif path
input_same_path = rf'D:\CAU\phenology_swc_vpd\paper\Data_and_code_online\FigShare_data'

folder_cor = fr'{input_same_path}\{scale}km\Correlation(SM_VPD_pearson){cor_test_number}'  # (POS-30) - POS

folder_cor_pvalue = fr'{input_same_path}\{scale}km\Correlation(SM_VPD_pearson){cor_test_number}\Pvalue'  # (POS-30) - POS

folder_sm = fr'{input_same_path}\{scale}km\SM_preseason_mean{climate_test_number}'  # (POS-30) - POS

folder_vpd = fr'{input_same_path}\{scale}km\VPD_preseason_mean{climate_test_number}'  # (POS-30) - POS

folder_ta = fr'{input_same_path}\{scale}km\Ta_preseason_mean{climate_test_number}'

folder_pre = fr'{input_same_path}\{scale}km\Pre_preseason_sum{climate_test_number}'

folder_srad = fr'{input_same_path}\{scale}km\Srad_preseason_sum{climate_test_number}'

#### 输入POS的tif path
pos_origin_folder = fr'{input_same_path}\{scale}km\POS_55km'  # start
sos_origin_folder = fr'{input_same_path}\{scale}km\SOS_55km'  # start

pos_detrend_folder = fr'{input_same_path}\{scale}km\POSdetrend_55km'  #start
sos_detrend_folder = fr'{input_same_path}\{scale}km\SOSdetrend_55km'  #start


#### 输入SPEI识别的干旱年份的tif path
spei_strength_input_path = rf'{input_same_path}\{scale}km\NH_SPEI{spei_length}_{spei_length}monthBeforePOS'
drought_path = fr'{input_same_path}\{scale}km\drought_event(POS_SPEI{spei_length}_threshold10%_way3)'

#### 输入AI的tif path
ai_tif_file = rf'{input_same_path}\{scale}km\AI\NH30_84_AI(graident)_55km.tif'

#### 输入的植被类型 tif path
veg_type_file = rf"{input_same_path}\{scale}km\Veg_type\NH_veg_type_55km(Python).tif"

#### 输入的耦合梯度 tif path
cor_mean_file = fr'{input_same_path}\{scale}km\3Cor_mean_slope\mean\SM_VPD_Cor17_8_0\Cor_mean_55km_All.tif'  #SOS - POS


###########################  ==== 输出设定 ==== #################################
output_path = fr'D:\CAU\phenology_swc_vpd\paper\Data_and_code_online\Result'

####################################### 2 数据读取 #################################################
tif_files_cor = sorted(glob.glob(os.path.join(folder_cor, '*.tif')))
tif_files_cor_pvalue = sorted(glob.glob(os.path.join(folder_cor_pvalue, '*.tif')))
tif_files_sm = sorted(glob.glob(os.path.join(folder_sm, '*.tif')))
tif_files_vpd = sorted(glob.glob(os.path.join(folder_vpd, '*.tif')))
tif_files_ta = sorted(glob.glob(os.path.join(folder_ta, '*.tif')))
tif_files_pre = sorted(glob.glob(os.path.join(folder_pre, '*.tif')))
tif_files_srad = sorted(glob.glob(os.path.join(folder_srad, '*.tif')))

pos_origin_tif_files = sorted(glob.glob(os.path.join(pos_origin_folder, '*.tif')))
sos_origin_tif_files = sorted(glob.glob(os.path.join(sos_origin_folder, '*.tif')))
pos_detrend_tif_files = sorted(glob.glob(os.path.join(pos_detrend_folder, '*.tif')))
sos_detrend_tif_files = sorted(glob.glob(os.path.join(sos_detrend_folder, '*.tif')))

drought_year_tif_files = sorted(glob.glob(os.path.join(drought_path, '*.tif')))
spei_strength_year_tif_files = sorted(glob.glob(os.path.join(spei_strength_input_path, '*.tif')))



###################################################
if cor_test_number == '(Preseason)':
    first_tif = tif_files_sm_1[0]
else:
    first_tif = tif_files_sm[0]
sample_tif = gdal.Open(first_tif)

if sample_tif is None:
    raise RuntimeError(f"无法打开 TIF 文件：{sample_tif}（驱动不支持或文件损坏）")

# 获取地理变换参数：投影、像素大小
#坐标和投影         坐标参考系：即数据所在的空间参考框架
crs = sample_tif.GetProjectionRef()          # 自动获取输入的 CRS
# print('crs:', crs)
gt = sample_tif.GetGeoTransform()  #地理坐标：经纬度。将像素坐标转换为实际地理坐标的数学变换参数。
proj = sample_tif.GetProjection()  #投影坐标：xy（单位m）

#像素
pixel_width = gt[1]
pixel_height = gt[5]

top_left_x = gt[0]
top_left_y = gt[3]

#行列数
sample_tif = sample_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)

rows = sample_tif.shape[0]
cols = sample_tif.shape[1]
print('rows:', rows, 'cols:', cols)

row_indices = np.repeat(np.arange(rows), cols)  # 行索引重复cols次
col_indices = np.tile(np.arange(cols), rows)  # 列索引平铺rows次

# 计算经纬度范围（修正 pixel_height 为负的情况）
lon_min = top_left_x
lon_max = top_left_x + cols * pixel_width  # 右边界经度
lat_min = top_left_y + rows * pixel_height  # 下边界纬度（最南端，可能更小）
lat_max = top_left_y  # 上边界纬度（最北端，可能更大）
print(f"经度范围: {lon_min:.6f} -> {lon_max:.6f}")
print(f"纬度范围: {lat_min:.6f} -> {lat_max:.6f}")


############################################ 3 时间-堆叠 ###################################################

## 数据堆叠
cor_stack = []

cor_pvalue_stack = []

sm_stack = []

vpd_stack = []

ta_stack = []

pre_stack = []

srad_stack = []

for tif_file in tif_files_cor:
    get_band(tif_file, cor_stack)

for tif_file in tif_files_cor_pvalue:
    get_band(tif_file, cor_pvalue_stack)

for tif_file in tif_files_sm:
    get_band(tif_file, sm_stack)

for tif_file in tif_files_vpd:
    get_band(tif_file, vpd_stack)

for tif_file in tif_files_ta:
    get_band(tif_file, ta_stack)

for tif_file in tif_files_pre:
    get_band(tif_file, pre_stack)

for tif_file in tif_files_srad:
    get_band(tif_file, srad_stack)


cor_stack = np.stack(cor_stack, axis=0)

cor_pvalue_stack = np.stack(cor_pvalue_stack, axis=0)

sm_stack = np.stack(sm_stack, axis=0)
vpd_stack = np.stack(vpd_stack, axis=0)

ta_stack = np.stack(ta_stack, axis=0)

pre_stack = np.stack(pre_stack, axis=0)

srad_stack = np.stack(srad_stack, axis=0)


pos_origin_stack = []
sos_origin_stack = []

pos_detrend_stack = []
sos_detrend_stack = []

drought_year_stack = []
spei_strength_year_stack = []

for tif_file in pos_origin_tif_files:
    get_band(tif_file, pos_origin_stack)
for tif_file in sos_origin_tif_files:
    get_band(tif_file, sos_origin_stack)

for tif_file in pos_detrend_tif_files:
    get_band(tif_file, pos_detrend_stack)
for tif_file in sos_detrend_tif_files:
    get_band(tif_file, sos_detrend_stack)

for tif_file in drought_year_tif_files:
    get_band(tif_file, drought_year_stack)
for tif_file in spei_strength_year_tif_files:
    get_band(tif_file, spei_strength_year_stack)



ai_tif = gdal.Open(ai_tif_file)
ai_type_data = ai_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
print('AI shape:', ai_type_data.shape)

veg_type_tif = gdal.Open(veg_type_file)
veg_type_data = veg_type_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
print('veg_type_data shape:', veg_type_data.shape)

#### 耦合梯度数据
cor_mean_file = gdal.Open(cor_mean_file)
cor_mean_data = cor_mean_file.GetRasterBand(1).ReadAsArray().astype(np.float32)
print('cor_mean_data shape:', cor_mean_data.shape)


print('Stack start')

pos_origin_stack = np.stack(pos_origin_stack, axis=0)
pos_detrend_stack = np.stack(pos_detrend_stack, axis=0)

sos_origin_stack = np.stack(sos_origin_stack, axis=0)
sos_detrend_stack = np.stack(sos_detrend_stack, axis=0)

drought_year_stack = np.stack(drought_year_stack, axis=0)

spei_strength_year_stack = np.stack(spei_strength_year_stack, axis=0)

print('Stack end!')


##### ======== 4 逐年分识别发生复合干旱事件的像元，并计算干旱或湿润导致的POS的变化 ============ #######
time_length = drought_year_stack.shape[0]

drought_event = np.where(((drought_year_stack==1) & (spei_strength_year_stack <= spei_strength)), drought_year_stack, np.nan)
wet_event = np.where(((drought_year_stack==2) & (spei_strength_year_stack >= -spei_strength)), drought_year_stack, np.nan)

##### 4.1 统计发生次数和像元数
drought_event_count = np.nansum(drought_event, axis=0)
wet_event_count = np.nansum(wet_event, axis=0)
print('drought_event_count1:', drought_event_count[26, 501])


#########============== 5 逐像元标准化处理 ===============###########
### 5.1 去异常与否的mask
if Outlier == 'Yes':
    def Outlier_array_IQR(x, i, j, qmin, qmax):
        """
        Compute average of the 25-th to 75-th percentile of the data across specified zonal.
        Spatial statistics after the removal of outliers by quantiles
        https://medium.com/@prashant.nair2050/hands-on-outlier-detection-and-treatment-in-python-using-1-5-iqr-rule-f9ff1961a414
        """
        if not type(x) is np.ndarray:
            x = np.asarray(x, dtype=np.float32)
        # x1 = x1[~np.isnan(x)]
        x_flatten = x[np.isfinite(x)]
        if len(x_flatten) < years_length/2:
            # 如果没有有效数据，直接返回全NaN
            return np.full_like(x, np.nan), i, j
        else:
            # remove Outliers
            upper_quartile, lower_quartile = np.percentile(x_flatten, [qmax, qmin])
            IQR = (upper_quartile - lower_quartile)

            # from scipy.stats import iqr
            # x1= x.copy()
            # x1= np.where(x!=fillvalue, x1, np.nan)
            # IQR = iqr(x1, nan_policy='omit')

            lower_range = lower_quartile - (1.5 * IQR)
            upper_range = upper_quartile + (1.5 * IQR)

            # maxv = np.max(x_flatten)
            # minv = np.min(x_flatten)
            valid_mask = np.logical_and(x <= upper_range, x >= lower_range)
            x_masked = np.where(valid_mask, x, np.nan)

            if (len(np.isfinite(x_masked)) > (years_length/2)) & (len(np.isfinite(x_masked)) <= years_length):
                return x_masked, i, j  # IQR, lower_range, upper_range,  minv, maxv
            else:
                return np.full_like(x, np.nan), i, j


    outlier_pos_stack = np.full((years_length, rows, cols), np.nan)

    outlier_sos_stack = np.full((years_length, rows, cols), np.nan)
    outlier_cor_stack = np.full((years_length, rows, cols), np.nan)
    outlier_sm_stack = np.full((years_length, rows, cols), np.nan)
    outlier_vpd_stack = np.full((years_length, rows, cols), np.nan)
    outlier_ta_stack = np.full((years_length, rows, cols), np.nan)
    outlier_pre_stack = np.full((years_length, rows, cols), np.nan)
    outlier_srad_stack = np.full((years_length, rows, cols), np.nan)

    if Basedon == 'Based_on_detrendPheno':
        pos_input_stack = pos_detrend_stack
        sos_input_stack = sos_detrend_stack
    elif Basedon == 'Based_on_originPheno':
        pos_input_stack = pos_origin_stack
        sos_input_stack = sos_origin_stack

    ### IQR去 POS 异常值
    results = Parallel(n_jobs=15, verbose=10)(
        delayed(Outlier_array_IQR)(
            pos_input_stack[:, i, j], i, j, 25, 75
        ) for i, j in zip(row_indices, col_indices)
    )

    for data_mask, i, j in results:
        outlier_pos_stack[:, i, j] = data_mask

    ### IQR去 SOS 异常值
    results = Parallel(n_jobs=15, verbose=10)(
        delayed(Outlier_array_IQR)(
            sos_input_stack[:, i, j], i, j, 25, 75
        ) for i, j in zip(row_indices, col_indices)
    )

    for data_mask, i, j in results:
        outlier_sos_stack[:, i, j] = data_mask

    ### IQR去 Cor 异常值
    results = Parallel(n_jobs=15, verbose=10)(
        delayed(Outlier_array_IQR)(
            cor_stack[:, i, j], i, j, 25, 75
        ) for i, j in zip(row_indices, col_indices)
    )

    for data_mask, i, j in results:
        outlier_cor_stack[:, i, j] = data_mask

    ### IQR去 SM 异常值
    results = Parallel(n_jobs=15, verbose=10)(
        delayed(Outlier_array_IQR)(
            sm_stack[:, i, j], i, j, 25, 75
        ) for i, j in zip(row_indices, col_indices)
    )

    for data_mask, i, j in results:
        outlier_sm_stack[:, i, j] = data_mask

    ### IQR去 VPD 异常值
    results = Parallel(n_jobs=15, verbose=10)(
        delayed(Outlier_array_IQR)(
            vpd_stack[:, i, j], i, j, 25, 75
        ) for i, j in zip(row_indices, col_indices)
    )

    for data_mask, i, j in results:
        outlier_vpd_stack[:, i, j] = data_mask

    ### IQR去 Ta 异常值
    results = Parallel(n_jobs=15, verbose=10)(
        delayed(Outlier_array_IQR)(
            ta_stack[:, i, j], i, j, 25, 75
        ) for i, j in zip(row_indices, col_indices)
    )

    for data_mask, i, j in results:
        outlier_ta_stack[:, i, j] = data_mask

    ### IQR去 Pre 异常值
    results = Parallel(n_jobs=15, verbose=10)(
        delayed(Outlier_array_IQR)(
            pre_stack[:, i, j], i, j, 25, 75
        ) for i, j in zip(row_indices, col_indices)
    )

    for data_mask, i, j in results:
        outlier_pre_stack[:, i, j] = data_mask

    ### IQR去 Srad 异常值
    results = Parallel(n_jobs=15, verbose=10)(
        delayed(Outlier_array_IQR)(
            srad_stack[:, i, j], i, j, 25, 75
        ) for i, j in zip(row_indices, col_indices)
    )

    for data_mask, i, j in results:
        outlier_srad_stack[:, i, j] = data_mask

    ### 去异常值后的mask
    mask = (np.isfinite(outlier_pos_stack) & np.isfinite(outlier_sos_stack) &
                    np.isfinite(outlier_cor_stack) &
                    np.isfinite(outlier_sm_stack) & np.isfinite(outlier_vpd_stack) &
                    np.isfinite(outlier_ta_stack) & np.isfinite(outlier_pre_stack) & np.isfinite(outlier_srad_stack))


elif Outlier == 'No':
    if Basedon == 'Based_on_detrendPheno':
        ##不去异常值的mask
        mask = (np.isfinite(pos_detrend_stack) & np.isfinite(sos_detrend_stack) &
                np.isfinite(cor_stack)  &
                np.isfinite(sm_stack) & np.isfinite(vpd_stack) &
                np.isfinite(ta_stack) & np.isfinite(pre_stack) & np.isfinite(srad_stack))
    elif Basedon == 'Based_on_originPheno':
        ##不去异常值的mask
        mask = (np.isfinite(pos_origin_stack) & np.isfinite(sos_origin_stack) &
                np.isfinite(cor_stack) &
                np.isfinite(sm_stack) & np.isfinite(vpd_stack) &
                np.isfinite(ta_stack) & np.isfinite(pre_stack) & np.isfinite(srad_stack))

if SigCorPvalue == 'Yes':
    mask =  (cor_pvalue_stack <= 0.1) & np.isfinite(cor_pvalue_stack) & mask

vaild_year_length = np.sum(mask, axis=0)

space_mask = vaild_year_length > (years_length / 2)  # 形状：(rows, cols)


### 4.2 有效的mask+求cor mean
space_mask_3d = space_mask[np.newaxis, :, :]  # (1, rows, cols)
space_mask_3d = np.repeat(space_mask_3d, years_length, axis=0)  # (years_length, rows, cols)

cor_stack_masked = np.where(space_mask_3d, cor_stack, np.nan)

cor_mean = np.nanmean(cor_stack_masked, axis=0)  # 形状: (rows, cols)

### 4.3 时间序列标准化
def standardize_data(data_stack, mask_3d):
    """时间标准化"""

    def standardize_func(data_pixel, i, j):
        original_nan_mask = np.isnan(data_pixel)

        spatio_mean = np.nanmean(data_pixel)
        spatio_std = np.nanstd(data_pixel)

        standardized_pixel_data = (data_pixel - spatio_mean) / spatio_std

        standardized_pixel_data[original_nan_mask] = np.nan

        return standardized_pixel_data, i, j

    data_stack = np.where(mask_3d, data_stack, np.nan)

    standardized_data = np.full((years_length, rows, cols), np.nan)

    results = Parallel(n_jobs=15, verbose=10)(
        delayed(standardize_func)(
            data_stack[:, i, j],
            i, j
        ) for i, j in zip(row_indices, col_indices)
    )

    for standardized_pixel_data, i, j in results:
        standardized_data[:, i, j] = standardized_pixel_data




    return standardized_data

if Outlier == 'Yes':
    pos_standardized = standardize_data(outlier_pos_stack, space_mask_3d)
    sos_standardized = standardize_data(outlier_sos_stack, space_mask_3d)
    cor_standardized = standardize_data(outlier_cor_stack, space_mask_3d)
    sm_standardized = standardize_data(outlier_sm_stack, space_mask_3d)
    vpd_standardized = standardize_data(outlier_vpd_stack, space_mask_3d)
    ta_standardized = standardize_data(outlier_ta_stack, space_mask_3d)
    pre_standardized = standardize_data(outlier_pre_stack, space_mask_3d)
    srad_standardized = standardize_data(outlier_srad_stack, space_mask_3d)
if Outlier == 'No':
    if Basedon == 'Based_on_detrendPheno':
        pos_standardized = standardize_data(pos_detrend_stack, space_mask_3d)
        sos_standardized = standardize_data(sos_detrend_stack, space_mask_3d)
    elif Basedon == 'Based_on_originPheno':
        pos_standardized = standardize_data(pos_origin_stack, space_mask_3d)
        sos_standardized = standardize_data(sos_origin_stack, space_mask_3d)
    cor_standardized = standardize_data(cor_stack, space_mask_3d)
    sm_standardized = standardize_data(sm_stack, space_mask_3d)
    vpd_standardized = standardize_data(vpd_stack, space_mask_3d)
    ta_standardized = standardize_data(ta_stack, space_mask_3d)
    pre_standardized = standardize_data(pre_stack, space_mask_3d)
    srad_standardized = standardize_data(srad_stack, space_mask_3d)

### 5.3 计算有效像元的mean/slope/std ###
def drought_wet_normal_year_diff(data_stack, drought_wet_normal_year_stack):
    ######### 先分别筛选出干旱、湿润、正常年份 ##########
    drought_year_mask = (drought_wet_normal_year_stack == 1)
    wet_year_mask = (drought_wet_normal_year_stack == 2)
    normal_year_mask = (drought_wet_normal_year_stack == 0)

    ######### 计算干旱、湿润、正常年份数据的均值后分别做差 #########
    data_in_drought_year = np.where(drought_year_mask, data_stack, np.nan)
    data_in_wet_year = np.where(wet_year_mask, data_stack, np.nan)
    data_in_normal_year = np.where(normal_year_mask, data_stack, np.nan)

    data_in_drought_mean = np.nanmean(data_in_drought_year, axis=0)
    data_in_wet_mean = np.nanmean(data_in_wet_year, axis=0)
    data_in_normal_mean = np.nanmean(data_in_normal_year, axis=0)

    data_diff_in_drought_noraml_year = data_in_drought_mean - data_in_normal_mean
    data_diff_in_wet_noraml_year = data_in_wet_mean - data_in_normal_mean

    return data_diff_in_drought_noraml_year, data_diff_in_wet_noraml_year



### POS:
pos_stack = np.where(space_mask_3d, pos_origin_stack, np.nan)
pos_diff_drought_normal_origin, pos_diff_wet_normal_origin = drought_wet_normal_year_diff(pos_stack, drought_year_stack)
pos_diff_drought_normal, pos_diff_wet_normal = drought_wet_normal_year_diff(pos_standardized, drought_year_stack)

print(f'pos_diff_drought_normal有效像元数量：{np.count_nonzero(np.isfinite(pos_diff_drought_normal))}')
print(f'pos_diff_wet_normal有效像元数量：{np.count_nonzero(np.isfinite(pos_diff_wet_normal))}')

### SOS:
sos_diff_drought_normal, sos_diff_wet_normal = drought_wet_normal_year_diff(sos_standardized, drought_year_stack)

### Cor
cor_diff_drought_normal, cor_diff_wet_normal = drought_wet_normal_year_diff(cor_standardized, drought_year_stack)

### SM slope
sm_diff_drought_normal, sm_diff_wet_normal = drought_wet_normal_year_diff(sm_standardized, drought_year_stack)

### VPD slope
vpd_diff_drought_normal, vpd_diff_wet_normal = drought_wet_normal_year_diff(vpd_standardized, drought_year_stack)

### Ta slope
ta_diff_drought_normal, ta_diff_wet_normal = drought_wet_normal_year_diff(ta_standardized, drought_year_stack)

### Pre slope
pre_diff_drought_normal, pre_diff_wet_normal = drought_wet_normal_year_diff(pre_standardized, drought_year_stack)

### Srad slope
srad_diff_drought_normal, srad_diff_wet_normal = drought_wet_normal_year_diff(srad_standardized, drought_year_stack)

del cor_standardized, sm_standardized, vpd_standardized, ta_standardized, pre_standardized, srad_standardized


########################### 6 Preseason Pearson+Partial ################
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches


## a or b
def plot_cor_mean_or_slope_and_pvalue_forAllvegType(plot_data, colorbarmin, colorbarmax, drought_or_wet, name, ax):

    fig = ax.get_figure()
    if name == 'All':
        height = 0.18
        wspace = 0.05
    else:
        height = 0.25
        wspace = 0.15
    gs_inner = ax.get_subplotspec().subgridspec(2, 2,
                                                width_ratios=[5, 0.8],
                                                height_ratios=[5, 0.3],
                                                hspace=height, wspace=wspace)

    # 隐藏父级 ax，因为它只是个占位符
    ax.axis('off')

    plots = []  # 存储每个子图的plot对象

    # 创建内部真正的三个子轴
    ax1 = fig.add_subplot(gs_inner[0, 0])  # 地图
    ax2 = fig.add_subplot(gs_inner[0, 1])  # 纬度曲线
    if drought_or_wet == 'drought':
        if name in ['All', 'Grass', 'Humid', 'Cor(lt-0.4)']:
            ax3 = fig.add_subplot(gs_inner[1, :])  # Colorbar

    plot_data_mean = np.nanmean(plot_data)
    valid_mask = np.isfinite(plot_data)
    plot_data_count_sum = np.sum(valid_mask)

    plot_data_gte0 = np.where(plot_data>=0, plot_data, np.nan)
    plot_data_gte0_mean = np.nanmean(plot_data_gte0)
    plot_data_gte0_sum = np.sum(plot_data>=0)

    plot_data_lt0 = np.where(plot_data < 0, plot_data, np.nan)
    plot_data_lt0_mean = np.nanmean(plot_data_lt0)
    plot_data_lt0_sum = np.sum(plot_data < 0)

    advance_percent = (plot_data_lt0_sum / plot_data_count_sum) * 100
    delay_percent = (plot_data_gte0_sum / plot_data_count_sum) * 100

    ########### 子图1：空间分布 #################
    ax1.set_box_aspect(1)  #强制地图轴的形状为正方，使其直径撑满格子高度
    ax1.axis('off')
    ### 创建地图
    m = Basemap(ax=ax1,
                projection='npstere',   # 北极投影
                boundinglat=30,         # 最低显示纬度（你现在是30N）
                lon_0=0,                # 中心经度（可以改） 180:太平洋居中；90：亚洲居中
                resolution='l')

    # 生成网格坐标
    lons = np.linspace(-180, 180, cols, endpoint=False)
    lats = np.linspace(lat_max , lat_min , rows)
    lons, lats = np.meshgrid(lons, lats)

    # 设置经纬度刻度
    m.drawparallels(np.arange(30, 90, 30),
                    labels=[0, 0, 0, 0],
                    linewidth=0.5,
                    # fontsize=8,
                    color="black")

    m.drawmeridians(np.arange(0, 360, 60),
                    latmax=90,  #使经度线在北极交汇
                    labels=[0, 0, 0, 0],  #labels=[left, right, top, bottom] 控制经度显示与否
                    linewidth=0.5,
                    # fontsize=8,
                    color="gray")

    # # 填充大陆
    # m.fillcontinents(color='white', lake_color='white', zorder=1)

    # map_boundary = m.drawmapboundary(linewidth=0)  # 不显示边界线


    ### 绘制数据
    # 颜色映射
    color_list = ['#c51b7d', '#de77ae', '#f1b6da', '#fde0ef',
                  '#e6f5d0', '#b8e186', '#7fbc41', '#4d9221']  # PiYG / 8
    cmap = mpl.colors.ListedColormap(color_list)
    bins = np.linspace(colorbarmin, colorbarmax, 9)
    norm = mpl.colors.BoundaryNorm(bins, cmap.N)

    plot = m.pcolormesh(lons, lats, plot_data, cmap=cmap, norm=norm, shading='nearest', latlon=True,
                        zorder=1)  # 避免极区撕裂

    plots.append(plot)  # 保存plot对象

    ax1.set_frame_on(False)

    ### 绘制边界
    terrence = r'D:\CAU\Border\NH_terrence\NH_Terrence_30_90'
    m.readshapefile(terrence, 'NH_Terrence', drawbounds=False)
    for shape in m.NH_Terrence:
        # 将列表转为 numpy 数组方便计算
        points = np.array(shape)
        x, y = points[:, 0], points[:, 1]

        # 核心逻辑：计算相邻点之间的投影距离
        # 如果相邻两个点在投影平面上的距离突然变得非常大，说明这是一条“跨圆心”的回环线
        dist = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)

        # 设定一个阈值（投影坐标通常很大，比如 100000 级）
        # 只要相邻点距离超过地图直径的 1/10，就判定为异常跳变
        threshold = (ax1.get_xlim()[1] - ax1.get_xlim()[0]) * 0.1

        # 找到跳变点的索引
        break_indices = np.where(dist > threshold)[0]

        if len(break_indices) == 0:
            # 没有跳变，直接画整条线
            ax1.plot(x, y, color='black', linewidth=0.3, zorder=3)
        else:
            # 有跳变，将线段切断，分段画出
            # 这样既能去掉横跨圆心的直线，又能保留正常的边界
            start_idx = 0
            for break_idx in break_indices:
                ax1.plot(x[start_idx:break_idx + 1], y[start_idx:break_idx + 1],
                         color='black', linewidth=0.3, zorder=3)
                start_idx = break_idx + 1
            # 画最后一段
            ax1.plot(x[start_idx:], y[start_idx:], color='black', linewidth=0.3, zorder=3)

    # x_pole, y_pole = m(0, 90)
    # ax1.scatter(x_pole, y_pole, color='black', s=5, zorder=3)

    ### 最外边界的裁剪
    from matplotlib.patches import Circle

    x0, x1 = ax1.get_xlim()
    y0, y1 = ax1.get_ylim()

    center = [(x0 + x1) / 2, (y0 + y1) / 2]
    radius = (x1 - x0) / 2

    clip_circle = Circle(center, radius, transform=ax1.transData)

    for artist in ax1.collections + ax1.lines + ax1.patches:
        artist.set_clip_path(clip_circle)

    boundary_circle = Circle(
        center,
        radius,
        transform=ax1.transData,
        facecolor='none',
        edgecolor='black',  # 颜色
        linewidth=1,
        clip_on=False,
        zorder=4  # 放最上层
    )

    ax1.add_patch(boundary_circle)

    # if name == 'All':
    #     if drought_or_wet == 'drought':
    #         ax1.set_title(f'(a)', pad=10, fontweight='bold')
    #     elif drought_or_wet == 'wet':
    #         ax1.set_title(f'(b)', pad=10, fontweight='bold')
    # elif name == 'Forest' or name == 'Arid':
    #     if drought_or_wet == 'drought':
    #         ax1.set_title(f'(a) {name}', pad=10, fontweight='bold')
    #     elif drought_or_wet == 'wet':
    #         ax1.set_title(f'(b)', pad=10, fontweight='bold')
    # elif name == 'Shrub' or name == 'Semi-arid':
    #     if drought_or_wet == 'drought':
    #         ax1.set_title(f'(d) {name}', pad=10, fontweight='bold')
    #     elif drought_or_wet == 'wet':
    #         ax1.set_title(f'(e)', pad=10, fontweight='bold')
    # elif name == 'Savanna' or name == 'Dry sub-humid':
    #     if drought_or_wet == 'drought':
    #         ax1.set_title(f'(g) {name}', pad=10, fontweight='bold')
    #     elif drought_or_wet == 'wet':
    #         ax1.set_title(f'(h)', pad=10, fontweight='bold')
    # elif name == 'Grass' or name == 'Humid':
    #     if drought_or_wet == 'drought':
    #         ax1.set_title(f'(j) {name}', pad=10, fontweight='bold')
    #     elif drought_or_wet == 'wet':
    #         ax1.set_title(f'(k)', pad=10, fontweight='bold')

    # ax1.set_title(f'({word}) {name}', pad=10, fontweight='bold')
    if name == 'All':
        h = 0.24
        v = 0.815
    else:
        h = 0.22
        v = 0.8
    ax1.text(h, v,
             f'Mean = {plot_data_mean:.1f}\n'
             f'Advance = {plot_data_lt0_mean:.1f} ({advance_percent:.1f}%)\n'
             f'Delay = {plot_data_gte0_mean:.1f} ({delay_percent:.1f}%)',
             transform=ax1.transAxes,  # 使用相对坐标，方便定位
             multialignment='center',   # 垂直居中
             fontsize=6)


    ########### 子图2：逐纬度变化趋势

    # 使用实际纬度值作为y轴
    lat_centers = lats[:, 0]

    plot_data_lat = np.nanmean(plot_data, axis=1)

    # if drought_or_wet == 'drought':
    ax2.axvline(x=0, color='gray', linestyle='--', linewidth=1)

    # elif drought_or_wet == 'wet':
    #     ax2.axvline(x=-0.2, color='gray', linestyle='--', linewidth=1)

    ax2.plot(plot_data_lat, lat_centers, color='red', linewidth=1, alpha=0.8)

    ax2.tick_params(axis='both', length=2, pad=3)#,which='major', )

    # if drought_or_wet == 'drought':
    ax2.set_xlim(-6, 6)
    ax2.set_xticks(np.arange(-4, 4.1, 4))
    ax2.set_xticklabels(['-4 ', '0', '4'])  # 手动设置标签
    # elif drought_or_wet == 'wet':
    #     ax2.set_xlim(-0.5, 0)
    #     ax2.set_xticks(np.arange(-0.4, 0.01, 0.2))
    #     ax2.set_xticklabels(['-0.4', '-0.2', '0'], rotation=45)  # 手动设置标签

    ax2.set_ylim(30, 90)
    ax2.set_yticks(np.arange(30, 91, 10))
    ax2.set_yticklabels(f'{int(x)}°' for x in np.arange(30, 91, 10))



    ########### 子图3：Colorbar
    ### 生成Colorbar（使用最后一个位置）
    if drought_or_wet == 'drought':
        if name in ['All', 'Grass', 'Humid', 'Cor(lt-0.4)']:
            cbar = fig.colorbar(plots[0], cax=ax3, orientation='horizontal')

            cbar.set_ticks(bins)

            cbar.set_label('PPT difference (days)', labelpad=3)
            cbar.set_ticklabels([f'{int(x)}' for x in bins])


    plt.tight_layout()

    # 当前 ax1 左下角
    pos1 = ax1.get_position()

    pos2 = ax2.get_position()

    if drought_or_wet == 'drought':
        if name in ['All', 'Grass', 'Humid', 'Cor(lt-0.4)']:
            pos3 = ax3.get_position()

    # 重新设置 ax1
    if name == 'All':
        xpos = 0.06
        ax1.set_position([
            pos1.x0 - xpos,  # 左边不变
            pos2.y0,  # 和 ax2 对齐底部
            pos2.height,
            pos2.height
        ])  # [left, bottom, width, height]
    else:
        xpos = 0.02
        ax1.set_position([
            pos1.x0 - xpos,  # 仅左移
            pos1.y0 ,  # 保持原bottom
            pos1.width,  # 保持原width
            pos1.height  # 保持原height
        ])

    pos1_new = ax1.get_position()
    pos2_new = ax2.get_position()
    # print(f'ax1 height = {pos1_new.height}\n'
    #       f'ax1 width = {pos1_new.width}\n'
    #       f'ax2 height = {pos2_new.height}\n')
    if drought_or_wet == 'drought':
        if name in ['All', 'Grass', 'Humid', 'Cor(lt-0.4)']:
            pos1_new = ax1.get_position()
            ax3.set_position([
                pos1_new.x0+0.11,
                pos3.y0-0.01,
                pos2.x1 - pos1_new.x0,
                pos3.height])

    # plt.tight_layout()

    # plt.show()


## c
def plot_cor_mean_or_slope_lt0_sig_and_lt0_nosig_and_gt0_forAllvegType(drought_data, wet_data, colorbarmin, colorbarmax, name, ax):

    bins = np.arange(colorbarmin, colorbarmax+1, 1)
    count_drought_data, _ = np.histogram(drought_data, bins=bins)
    count_wet_data, _ = np.histogram(wet_data, bins=bins)

    ### plot
    fig = ax.get_figure()
    gs_inner = ax.get_subplotspec().subgridspec(
        2, 1,
        height_ratios=[5, 0.3],
        hspace=0.15
    )

    ax1 = fig.add_subplot(gs_inner[0])

    ax.axis('off')

    #设置截断y轴
    # if data_type == 'cor slope':
    #     ax.axis('off')
    #     bax = brokenaxes(
    #         ylims=((0, 5000), (10000, 11000)),
    #         hspace=0.1,
    #         height_ratios=[1, 5],  # 上图占1份，下图占2份。数值越大，对应的部分占地越广
    #         subplot_spec=sub_gs
    #     )
    #
    #     # 设置bar的位置
    #     bin_centers = (bins[:-1] + bins[1:]) / 2
    #     print(f'bin_centers:{bin_centers}')
    #
    #     total_width = 0.007  # 一个刻度位内柱子的总占用宽度
    #     n = 2  # 类别数量
    #     width = total_width / n  # 单个柱子的宽度
    #
    #     bax.bar(bin_centers - width / 2, count_lt0_sig, width=width, color='#b2182b', label='Cor sig-strong')
    #     bax.bar(bin_centers + width / 2, count_lt0_nosig, width=width, color='#fddbc7',
    #             label='Cor nonsig-strong')
    #     bax.bar(bin_centers - width / 2, count_gte0_nosig, width=width, color='#d1e5f0',
    #             label='Cor nonsig-weaken')
    #     bax.bar(bin_centers + width / 2, count_gte0_sig, width=width, color='#2166ac', label='Cor sig-weaken')
    #
    #
    #     ticks = np.arange(colorbarmin, colorbarmax + 0.0001, 0.007)
    #     labels = [f'{(c * 10):.2f}' for c in ticks]
    #
    #     bax.set_xlim(colorbarmin, colorbarmax)
    #
    #     for ax_part in bax.axs:
    #         ax_part.set_xticks(ticks)
    #         ax_part.set_xticklabels(labels)
    #
    #     bax.set_xlabel('VPD-SM coupling trend (per decade)', labelpad=20) #控制标签与刻度距离
    #
    #     bax.set_ylabel('Frequency', labelpad=31)  # 控制标签与刻度距离

    #     # Colorbar
    #     bax.legend(
    #         loc='upper right',
    #         bbox_to_anchor=(1.2, 1),
    #         ncol=1,
    #         frameon=False,  # 控制 legend（图例）外框是否显示
    #         handlelength=1,
    #         handleheight=1
    #     )


    # 不设置截断y轴
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    # 设置bar的位置
    bin_centers = (bins[:-1] + bins[1:]) / 2
    print(f'bin_centers:{bin_centers}')

    total_width = 1  # 一个刻度位内柱子的总占用宽度
    n = 2  # 类别数量
    width = total_width / n  # 单个柱子的宽度

    ax1.bar(bin_centers - width / 2, count_drought_data, width=width, color='#c51b7d', alpha = 0.5, label='drought - normal')
    ax1.bar(bin_centers + width / 2, count_wet_data, width=width, color='#4d9221', alpha = 0.5, label='wet - normal')

    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)

    ax1.tick_params(axis='both', length=2, pad=3)#,which='major', )

    ticks = np.arange(colorbarmin, colorbarmax + 0.0001, 3)
    ax1.set_xlim(colorbarmin, colorbarmax)
    ax1.set_xticks(ticks)
    ax1.tick_params(axis='x', labelsize=9)

    ax1.set_xlabel('PPT difference (days)', labelpad=5) #控制标签与刻度距离

    if  name == 'All':
        ax1.set_ylim(0, 2000)
        ax1.set_yticks(np.arange(0, 2000.1, 500))
        ax1.set_yticklabels(f'{int(x * 0.01)}' for x in np.arange(0, 2000.1, 500))
    elif name in ['Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']:
        ax1.set_ylim(0, 1000)
        ax1.set_yticks(np.arange(0, 1000.1, 200))
        ax1.set_yticklabels(f'{int(x * 0.01)}' for x in np.arange(0, 1000.1, 200))
    else:
        ax1.set_ylim(0, 700)
        ax1.set_yticks(np.arange(0, 700.1, 200))
        ax1.set_yticklabels(f'{int(x * 0.01)}' for x in np.arange(0, 700.1, 200))

    if name == 'All':
        x_pos = -0.15
        y_pos = 1.15
    else:
        x_pos = -0.15
        y_pos = 1.12

    ax1.text(
        x_pos,  # x = 刻度位置（数据坐标）
        y_pos,  # y = 稍微往下（轴坐标）
        r'$×10^{2}$',  # 你想要的内容
        transform=ax1.transAxes,
        ha='left',  # 向右展开（避免压缩）
        va='top',
        rotation=0,
        clip_on=False
    )


    # elif name in ['Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']:
    #     ax.set_ylim(0, 2500)
    #     ax.set_yticks(np.arange(0, 2500.1, 500))


    # Colorbar
    if name == 'All':
        c = 0.5
    else:
        c = 0.45
    if name in ['All', 'Grass', 'Humid', 'Cor(lt-0.4)']:
        ax1.legend(
            loc='lower center',
            bbox_to_anchor=(c, -0.4),
            ncol=2,
            columnspacing=0.4,
            frameon=False,  # 控制 legend（图例）外框是否显示
            handlelength=0.8,
            handleheight=0.8,
            fontsize = 8
        )

    ax1.set_ylabel('Frequency', labelpad=3)  # 控制标签与刻度距离

    # if name == 'Forest' or name == 'Arid' or name == 'All':
    #     word = 'c'
    # elif name == 'Shrub' or name == 'Semi-arid':
    #     word = 'f'
    # elif name == 'Savanna' or name == 'Dry sub-humid':
    #     word = 'i'
    # elif name == 'Grass' or name == 'Humid':
    #     word = 'l'
    # ax.set_title(f'({word})', pad=10, fontweight='bold')

    # elif data_type == 'cor mean':
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #
    #     total_width = 0.2  # 一个刻度位内柱子的总占用宽度
    #     n = 2  # 类别数量
    #     width = total_width / n  # 单个柱子的宽度
    #
    #     bin_centers = (bins[:-1] + bins[1:]) / 2
    #
    #     color_list = ['#a50f15', '#de2d26', '#fb6a4a', '#fc9272',
    #                   '#fcbba1', '#fee5d9', '#9ecae1']
    #
    #     # 逐个柱子绘制，以确保颜色严格对应
    #     for j in range(len(count_mean)):
    #         ax.bar(bin_centers[j], count_mean[j], width=width,
    #                 color=color_list[j], edgecolor='none')
    #
    #     ax.set_xlim(colorbarmin, colorbarmax)
    #     ax.set_xticks(np.arange(colorbarmin, colorbarmax, 0.2))
    #
    #     ax.set_xlabel('VPD-SM coupling', labelpad=3) #控制标签与刻度距离
    #     if  name in ['Forest', 'Shrub', 'Savanna', 'Grass']:
    #         ax.set_ylim(0, 2500)
    #         ax.set_yticks(np.arange(0, 2500.1, 500))
    #     elif name in ['Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']:
    #         ax.set_ylim(0, 4000)
    #         ax.set_yticks(np.arange(0, 4000.1, 1000))
    #
    #     # ax.axs[0].set_yticks([5000, 6000, 7500])
    #     # ax.axs[1].set_yticks([0, 1000, 2000, 3500])
    #
    #     ax.set_ylabel('Frequency', labelpad=3)  # 控制标签与刻度距离

    # plt.show()


## d & e & S26-28: Driving factors of Drought/Wet and normal diff
def Partial_ML_calculate_and_plot(pos_data, sos_data, cor_data, sm_data, vpd_data, ta_data, pre_data, srad_data, grade_by, veg_type, ax):

    var_list = ['POS mean', 'POS slope', 'POS std', 'POS drought normal diff', 'POS wet normal diff',
                'POS corhigh corlow diff',

                'SOS mean', 'SOS slope', 'SOS std', 'SOS drought normal diff', 'SOS wet normal diff',
                'SOS corhigh corlow diff',

                'Cor mean', 'Cor slope', 'Cor std', 'Cor drought normal diff', 'Cor wet normal diff',
                'Cor corhigh corlow diff',

                'SM mean', 'SM slope', 'SM std', 'SM drought normal diff', 'SM wet normal diff',
                'SM corhigh corlow diff',

                'VPD mean', 'VPD slope', 'VPD std', 'VPD drought normal diff', 'VPD wet normal diff',
                'VPD corhigh corlow diff',

                'Ta mean', 'Ta slope', 'Ta std', 'Ta drought normal diff', 'Ta wet normal diff',
                'Ta corhigh corlow diff',

                'Pre mean', 'Pre slope', 'Pre std', 'Pre drought normal diff', 'Pre wet normal diff',
                'Pre corhigh corlow diff',

                'Srad mean', 'Srad slope', 'Srad std', 'Srad drought normal diff', 'Srad wet normal diff',
                'Srad corhigh corlow diff']  ##修改此处顺序可以修改子图一的顺序

    if grade_by != 'All':
        plt.rcParams.update({
            'font.family': 'Arial',

            'mathtext.fontset': 'custom',

            'mathtext.rm': 'Arial',  # 正常
            'mathtext.it': 'Arial:italic',  # 斜体
            'mathtext.bf': 'Arial:bold',  # 粗体

            # 可选（推荐加）
            'mathtext.default': 'regular',  # 避免自动变斜体

            'font.size': 10,
            'axes.titlesize': 10,
            'axes.labelsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            # 'text.usetex': False,  # 不使用外部LaTeX
        })

    ### 划分植被类型/AI梯度
    var_mask = (np.isfinite(pos_data) & np.isfinite(sos_data) &
                np.isfinite(cor_data) & np.isfinite(sm_data) & np.isfinite(vpd_data) &
                np.isfinite(ta_data) & np.isfinite(pre_data) & np.isfinite(srad_data)
                )

    if grade_by == 'All':
        types = ['All']
        codes = [1]

        fig = ax.figure
        gs_inner = ax.get_subplotspec().subgridspec(1, 4,
                               width_ratios=[1, 1, 1, 0.1],  # 三列的宽度比
                               height_ratios=[1],  # 最后一个给colorbar
                               hspace=0.3, wspace=0.2)

        ax.axis('off')

    elif grade_by == 'Veg':
        if veg_type == 'All':
            types = ['Forest', 'Shrub', 'Savanna', 'Grass']
            codes = [1, 2, 3, 4]

            fig = plt.figure(figsize=(8.2, 14))
            gs = gridspec.GridSpec(4, 4,
                                   width_ratios=[1, 1, 1, 0.1],  # 三列的宽度比
                                   height_ratios=[1, 1, 1, 1],  # 最后一个给colorbar
                                   hspace=0.3, wspace=0.2)
        elif veg_type == 'Grass':
            types = ['Grass']
            codes = [4]

            fig = ax.figure
            gs_inner = ax.get_subplotspec().subgridspec(1, 4,
                                                        width_ratios=[1, 1, 1, 0.1],  # 三列的宽度比
                                                        height_ratios=[1],  # 最后一个给colorbar
                                                        hspace=0.3, wspace=0.2)
            ax.axis('off')

    elif grade_by == 'AI':
        types = ['Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']
        codes = [2, 3, 5, 6]

        fig = plt.figure(figsize=(8.2, 14))
        gs = gridspec.GridSpec(4, 4,
                               width_ratios=[1, 1, 1, 0.1],  # 三列的宽度比
                               height_ratios=[1, 1, 1, 1],  # 最后一个给colorbar
                               hspace=0.3, wspace=0.2)

    elif grade_by == 'Cor mean':
        types = ['Cor(-0.1~0)', 'Cor(-0.2~-0.1)', 'Cor(-0.3~-0.2)',
                 'Cor(-0.4~-0.3)', 'Cor(lt-0.4)']
        codes = [0, 1, 2, 3, 4]

        fig = plt.figure(figsize=(8.2, 15.5))
        gs = gridspec.GridSpec(5, 4,
                               width_ratios=[1, 1, 1, 0.1],  # 三列的宽度比
                               height_ratios=[1, 1, 1, 1, 1],  # 最后一个给colorbar
                               hspace=0.3, wspace=0.2)



    for i, (type, code) in enumerate(zip(types, codes)):
        if grade_by == 'All':
            ax1 = fig.add_subplot(gs_inner[0, 0])
            ax2 = fig.add_subplot(gs_inner[0, 1])
            ax3 = fig.add_subplot(gs_inner[0, 2])
            ax4 = fig.add_subplot(gs_inner[0, 3])
        elif grade_by == 'Veg' and veg_type == 'Grass':
            ax1 = fig.add_subplot(gs_inner[0, 0])
            ax2 = fig.add_subplot(gs_inner[0, 1])
            ax3 = fig.add_subplot(gs_inner[0, 2])
            ax4 = fig.add_subplot(gs_inner[0, 3])
        else:
            ax1 = plt.subplot(gs[i, 0])
            ax2 = plt.subplot(gs[i, 1])
            ax3 = plt.subplot(gs[i, 2])
            ax4 = plt.subplot(gs[i, 3])

        if grade_by == 'Veg':
            mask = (veg_type_data == code) & var_mask

        elif grade_by == 'AI':
            if code == 3:
                mask = ((ai_type_data == code) | (ai_type_data == 4)) & var_mask

            else:
                mask = (ai_type_data == code) & var_mask


        elif grade_by == 'Cor mean':
            if code == 4:
                mask = (
                        (cor_mean_data <= -0.4) & var_mask )
            else:
                mask = (
                        (cor_mean_data <= -(code*0.1)) & (cor_mean_data > -((code+1)*0.1)) & var_mask )

        elif grade_by == 'All':
            mask = var_mask

        valid_indices = np.where(mask.flatten())[0]

        ########## 空间尺度
        ## 标准化数据准备：drought/wet/normal year diff ###
        standardized_data_drought = pd.DataFrame({
            'POS drought normal diff': pos_diff_drought_normal.flatten()[valid_indices],

            'SOS drought normal diff': sos_diff_drought_normal.flatten()[valid_indices],

            'Cor drought normal diff': cor_diff_drought_normal.flatten()[valid_indices],

            'SM drought normal diff': sm_diff_drought_normal.flatten()[valid_indices],

            'VPD drought normal diff': vpd_diff_drought_normal.flatten()[valid_indices],

            'Ta drought normal diff': ta_diff_drought_normal.flatten()[valid_indices],

            'Pre drought normal diff': pre_diff_drought_normal.flatten()[valid_indices],

            'Srad drought normal diff': srad_diff_drought_normal.flatten()[valid_indices]
        })

        standardized_data_wet = pd.DataFrame({
            'POS wet normal diff': pos_diff_wet_normal.flatten()[valid_indices],

            'SOS wet normal diff': sos_diff_wet_normal.flatten()[valid_indices],

            'Cor wet normal diff': cor_diff_wet_normal.flatten()[valid_indices],

            'SM wet normal diff': sm_diff_wet_normal.flatten()[valid_indices],

            'VPD wet normal diff': vpd_diff_wet_normal.flatten()[valid_indices],

            'Ta wet normal diff': ta_diff_wet_normal.flatten()[valid_indices],

            'Pre wet normal diff': pre_diff_wet_normal.flatten()[valid_indices],

            'Srad wet normal diff': srad_diff_wet_normal.flatten()[valid_indices],
        })

        # standardized_data_corhigh_corlow= pd.DataFrame({
        #     'POS corhigh corlow diff': pos_diff_corhigh_corlow.flatten()[valid_indices],
        #
        #     'SOS corhigh corlow diff': sos_diff_corhigh_corlow.flatten()[valid_indices],
        #
        #     'Cor corhigh corlow diff': cor_diff_corhigh_corlow.flatten()[valid_indices],
        #
        #     'SM corhigh corlow diff': sm_diff_corhigh_corlow.flatten()[valid_indices],
        #
        #     'VPD corhigh corlow diff': vpd_diff_corhigh_corlow.flatten()[valid_indices],
        #
        #     'Ta corhigh corlow diff': ta_diff_corhigh_corlow.flatten()[valid_indices],
        #
        #     'Pre corhigh corlow diff': pre_diff_corhigh_corlow.flatten()[valid_indices],
        #
        #     'Srad corhigh corlow diff': srad_diff_corhigh_corlow.flatten()[valid_indices]
        # })

        # print(f'standardized_data drought的有效数据量：{len(standardized_data_drought.dropna())}')
        # print(f'standardized_data wet的有效数据量：{len(standardized_data_wet.dropna())}')

        standardized_data_drought = standardized_data_drought.dropna().reset_index()
        standardized_data_wet = standardized_data_wet.dropna().reset_index()
        # standardized_data_corhigh_corlow = standardized_data_corhigh_corlow.dropna().reset_index()

        # print('standardized_data_drought:\n', standardized_data_drought.iloc[:20, :])
        # print('standardized_data_wet:\n', standardized_data_wet.iloc[:20, :])

        # sys.exit()

        ######## Partial correlation & ML ###########
        def seq_pcorr(df, y, x_list):
            """
            Perform sequential partial correlation analysis.

            Parameters:
            - df (DataFrame): Input DataFrame containing the variables.
            - y (str): Dependent variable.
            - x_list (list): List of independent variables.

            Returns:
            - result_df (DataFrame): DataFrame containing the sequential partial correlation results.
            """
            res_list = []
            # print('df type:', df.dtypes)
            # print('y:', y)
            # print('x_list:', x_list)
            for var in x_list:
                # Perform partial correlation for each variable with other variables
                res = pg.partial_corr(df, y=y, x=var, covar=[item for item in x_list if item != var])
                res['var'] = var
                res_list.append(res)
            result_df = pd.concat(res_list, axis=0)
            return result_df

        results_dict = {}  # 用于存储每个y的结果

        for y in y_list :

            ### ============= 1. 偏相关分析 ============== ###
            print('Start partial analyze!')
            # if y == 'POS mean' :
            #     # x_list = ['SOS mean', 'SOS std', 'Cor mean', 'Cor std',
            #     #           'SM mean', 'SM std', 'VPD mean', 'VPD std', 'Ta mean', 'Ta std', 'Pre mean', 'Pre std', 'Srad mean', 'Srad std']
            #     x_list = ['SOS mean', 'Cor mean',
            #               'SM mean', 'VPD mean', 'Ta mean', 'Pre mean', 'Srad mean']
            #     df_pcorr = seq_pcorr(standardized_data_drought, y, x_list)[['var', 'r', 'p-val']].reset_index(drop=True).rename(
            #         columns={'r': 'r_pcorr', 'p-val': 'p_pcorr'})
            #     ml_data = standardized_data_drought
            #
            # elif y == 'POS std':
            #     x_list = ['SOS std', 'Cor std',
            #               'SM std', 'VPD std', 'Ta std', 'Pre std', 'Srad std']
            #     df_pcorr = seq_pcorr(outlier_standardized_data, y, x_list)[['var', 'r', 'p-val']].reset_index(drop=True).rename(
            #         columns={'r': 'r_pcorr', 'p-val': 'p_pcorr'})
            #     ml_data = outlier_standardized_data
            #
            #
            # elif y == 'POS slope':
            #     x_list = ['SOS slope', 'SOS std',
            #               'Cor slope', 'Cor std',
            #               'SM slope', 'SM std', 'VPD slope', 'VPD std',
            #               'Ta slope', 'Ta std', 'Pre slope', 'Pre std',
            #               'Srad slope', 'Srad std']
            #     df_pcorr = seq_pcorr(outlier_standardized_data, y, x_list)[['var', 'r', 'p-val']].reset_index(drop=True).rename(
            #         columns={'r': 'r_pcorr', 'p-val': 'p_pcorr'})
            #     ml_data = outlier_standardized_data


            if y == 'POS drought normal diff':
                x_list = ['SOS drought normal diff',
                          'Cor drought normal diff',
                          'SM drought normal diff',
                          'VPD drought normal diff',
                          'Ta drought normal diff',
                          'Pre drought normal diff',
                          'Srad drought normal diff']
                df_pcorr = seq_pcorr(standardized_data_drought, y, x_list)[['var', 'r', 'p-val']].reset_index(drop=True).rename(
                    columns={'r': 'r_pcorr', 'p-val': 'p_pcorr'})

            elif y == 'POS wet normal diff':
                x_list = ['SOS wet normal diff',
                          'Cor wet normal diff',
                          'SM wet normal diff',
                          'VPD wet normal diff',
                          'Ta wet normal diff',
                          'Pre wet normal diff',
                          'Srad wet normal diff']
                df_pcorr = seq_pcorr(standardized_data_wet, y, x_list)[['var', 'r', 'p-val']].reset_index(drop=True).rename(
                    columns={'r': 'r_pcorr', 'p-val': 'p_pcorr'})

            # elif y == 'POS corhigh corlow diff':
            #     x_list = ['SOS corhigh corlow diff',
            #               'Cor corhigh corlow diff',
            #               'SM corhigh corlow diff',
            #               'VPD corhigh corlow diff',
            #               'Ta corhigh corlow diff',
            #               'Pre corhigh corlow diff',
            #               'Srad corhigh corlow diff']
            #     df_pcorr = seq_pcorr(standardized_data_corhigh_corlow, y, x_list)[['var', 'r', 'p-val']].reset_index(drop=True).rename(
            #         columns={'r': 'r_pcorr', 'p-val': 'p_pcorr'})
            #     ml_data = standardized_data_corhigh_corlow



            df_pcorr = df_pcorr.rename(columns={'var': 'feature'})
            df_pcorr['Significance'] = ''

            for x in x_list:
                p_value = df_pcorr[df_pcorr['feature'] == x]['p_pcorr'].iloc[0]
                if p_value < 0.01:
                    significance = '**'  # p < 0.01 标记为 **
                elif p_value < 0.05:
                    significance = '*'  # 0.01 <= p < 0.05 标记为 *
                else:
                    significance = ''  # p >= 0.05 不标记

                df_pcorr.loc[df_pcorr['feature'] == x, 'Significance'] = significance

            df_pcorr['Abs_Coefficient'] = df_pcorr['r_pcorr'].abs()
            print('df_pcorr:\n', df_pcorr)

            # sys.exit()

            ###### ================= 2. ML =================== #######
            if y == 'POS drought normal diff':
                ml_data = standardized_data_drought
            elif y == 'POS wet normal diff':
                ml_data = standardized_data_wet
            # elif y == 'POS corhigh corlow diff':
            #     ml_data = standardized_data_corhigh_corlow

            X = ml_data[x_list]
            Y = ml_data[y]

            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
            print("test split done")


            def calculate_shap(xgb_model, x_train, y_train, x_test, y_test):

                # Step 1: Calculate MSE and R2 for train and test data
                mse_train = mean_squared_error(y_train, xgb_model.predict(x_train))
                r2_train = xgb_model.score(x_train, y_train)
                print(f'Train R2: {r2_train}, MSE: {mse_train}')

                y_train_pred = xgb_model.predict(x_test)
                r2_test = r2_score(y_test, y_train_pred)
                mse_test = mean_squared_error(y_test, y_train_pred)
                print(f'Test R2: {r2_test}, MSE: {mse_test}')

                # Step 2: Calculate SHAP values
                explainer = shap.TreeExplainer(xgb_model)
                print("SHAP explainer done")
                shap_values = explainer(x_train)
                print("SHAP values calculation done")

                # Convert SHAP values into a DataFrame
                shap_values_df = pd.DataFrame(shap_values.values, columns=x_train.columns, index=x_train.index)
                shap_values_df.columns = ['shap_' + col for col in x_train.columns]

                shap_values_df = pd.concat(
                    [x_train.reset_index(drop=True),
                     pd.DataFrame(shap_values.values,
                                  columns=[f'shap_{c}' for c in x_train.columns])],
                    axis=1
                )

                # Step 3: Consolidate feature importance and metrics
                df_xgb_model = pd.DataFrame(xgb_model.feature_importances_, index=x_list,
                                            columns=["importance"]).reset_index().rename(
                    columns={'index': 'var'})
                print("Feature importances calculated")
                print(df_xgb_model)

                df_xgb_model["model_mse"] = mse_train
                df_xgb_model["model_r2"] = r2_train
                df_xgb_model["model_test_mse"] = mse_test
                df_xgb_model["model_test_r2"] = r2_test

                # Step 4: Calculate mean absolute SHAP values
                mean_abs_shap_values = abs(shap_values.values).mean(0)
                print(f'Mean absolute SHAP values: {mean_abs_shap_values}')

                df_shap = pd.DataFrame([x_list, mean_abs_shap_values], index=["var", "shapvalue"]).T
                print("SHAP values calculated")

                pos_range = y_train.max() - y_train.min()
                shap_sum = shap_values.values.sum(axis=1)  # sum of SHAP per pixel

                print("POS_mean spatial range (days):", pos_range)
                print("SHAP sum range (days):", shap_sum.max() - shap_sum.min())

                # Step 5: Merge SHAP values and feature importance
                df_merge = pd.merge(df_shap, df_xgb_model, on="var")
                # print('df_merge1:\n', df_merge)
                df_merge = df_merge.rename(columns={'var': 'feature'})
                # df_merge['importance'] = df_merge['importance'] / df_merge['importance'].sum()  # Normalize feature importance
                # df_merge['shapvalue'] = df_merge['shapvalue'] / df_merge['shapvalue'].sum()  # Normalize SHAP importance
                # df_merge['mean'] = (df_merge['importance'] + df_merge['shapvalue']) / 2  # Calculate mean importance

                print('df_merge:\n', df_merge)

                # # Step 6: Sort features by SHAP value
                # df_merge = df_merge.sort_values(by='shapvalue', ascending=False)

                return shap_values_df, df_merge ,r2_test


            ############################### XGBoost ########################################
            if ML_model_in_spatio == 'XGBoost':
                ## K-fold和使用网格搜索寻找最佳参数 ##
                xgb_model = xgb.XGBRegressor(random_state=42)
                # 定义参数网格
                param_grid = {'n_estimators': [50, 100, 150, 200, 250, 300],
                              'max_depth': [5, 10, 15],
                              'learning_rate': [0.05, 0.1],
                              'subsample': [0.8, 0.9, 1],
                              'colsample_bytree': [0.8, 0.9, 1]}
                # # 定义 K 折交叉验证 (K-Fold)
                kfold = KFold(n_splits=10, shuffle=True, random_state=42)
                # 使用网格搜索寻找最佳参数
                grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='r2', cv=kfold, verbose=10, n_jobs=15, error_score='raise')

                # 拟合模型
                grid_search.fit(x_train, y_train)

                print(f'XGboost model done')

                # 输出最优参数和最优得分
                print(f"Best parameters: {grid_search.best_params_}")
                print(f"Best R2 score: {grid_search.best_score_}")

                Best_parameters = pd.DataFrame(grid_search.best_params_, index=[0])
                print(f'Best_parameters:{Best_parameters}')
                Best_parameters['mean R2 score'] = round(grid_search.best_score_, 4)
                Best_parameters.to_csv(
                    fr'D:\CAU\phenology_swc_vpd\Global_test4\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_{ML_model_in_spatio}_TestSize{test_size}_Best_parameters_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).csv')
                #
                # if grade_by == 'Veg':
                #     Best_parameters.to_csv(fr'D:\CAU\phenology_swc_vpd\Global_test4\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_VegType({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).csv')
                # elif grade_by == 'AI':
                #     Best_parameters.to_csv(fr'D:\CAU\phenology_swc_vpd\Global_test4\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_AI({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).csv')
                # elif grade_by == 'Cor mean':
                #     Best_parameters.to_csv(fr'D:\CAU\phenology_swc_vpd\Global_test4\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_Cormean({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).csv')
                # elif grade_by == 'All':
                #     Best_parameters.to_csv(fr'D:\CAU\phenology_swc_vpd\Global_test4\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_All_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).csv')
                print(f'Best_parameters done')

                # ## 使用最优参数训练模型 ##
                # xgboost = grid_search.best_estimator_

                ######################################################
                # #
                Best_parameters = pd.read_csv(
                    fr'D:\CAU\phenology_swc_vpd\Global_test4\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_{ML_model_in_spatio}_TestSize{test_size}_Best_parameters_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).csv')
                # if grade_by == 'Veg':
                #     Best_parameters = pd.read_csv(fr'D:\CAU\phenology_swc_vpd\Global_test4\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_VegType({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).csv')
                # elif grade_by == 'AI':
                #     Best_parameters = pd.read_csv(fr'D:\CAU\phenology_swc_vpd\Global_test4\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_AI({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).csv')
                # elif grade_by == 'Cor mean':
                #     Best_parameters = pd.read_csv(fr'D:\CAU\phenology_swc_vpd\Global_test4\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_Cormean({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).csv')
                # elif grade_by == 'All':
                #     Best_parameters = pd.read_csv(fr'D:\CAU\phenology_swc_vpd\Global_test4\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_All_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).csv')

                n_estimators = int(Best_parameters['n_estimators'][0])
                max_depth = int(Best_parameters['max_depth'][0])
                colsample_bytree = float(Best_parameters['colsample_bytree'][0])
                subsample = float(Best_parameters['subsample'][0])
                learning_rate = float(Best_parameters['learning_rate'][0])

                xgboost = XGBRegressor(n_estimators=n_estimators,
                                         max_depth=max_depth,
                                         colsample_bytree=colsample_bytree,         #每次分裂会考虑所有特征中 80% 的特征子集
                                         subsample=subsample,                 # 每棵树的随机采样比例
                                         learning_rate=learning_rate,
                                         random_state=42,
                                         n_jobs=20)

                xgboost.fit(x_train,y_train)
                print(f"xgb_model:", xgboost)


                shap_values_df, ModelImportance_and_ShapValueMean, r2_test = calculate_shap(xgboost, x_train, y_train, x_test, y_test)


            ############################ RF #########################################
            elif ML_model_in_spatio == 'RF':
                ## K-fold和使用网格搜索寻找最佳参数 ##
                rf_model = RandomForestRegressor(random_state=42, n_jobs=15)
                # 定义参数网格
                param_grid = {
                    'n_estimators': [50, 100, 150, 200, 250, 300],
                    'max_depth': [5, 10, 15, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7]
                }
                # # 定义 K 折交叉验证 (K-Fold)
                kfold = KFold(n_splits=10, shuffle=True, random_state=42)
                # 使用网格搜索寻找最佳参数
                grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, scoring='r2', cv=kfold, verbose=10, n_jobs=15, error_score='raise')
                # 拟合模型
                grid_search.fit(x_train, y_train)

                print(f'RF model done')

                # 输出最优参数和最优得分
                print(f"Best parameters: {grid_search.best_params_}")
                print(f"Best R2 score: {grid_search.best_score_}")

                Best_parameters = pd.DataFrame(grid_search.best_params_, index=[0])
                print(f'Best_parameters:{Best_parameters}')
                Best_parameters['mean R2 score'] = round(grid_search.best_score_, 4)
                if analyze_by == 'All':
                    Best_parameters.to_csv(
                        fr'D:\CAU\phenology_swc_vpd\Global_test4\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_{ML_model_in_spatio}_TestSize{test_size}_Best_parameters_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).csv')
                elif analyze_by != 'All':
                    Best_parameters.to_csv(
                        fr'D:\CAU\phenology_swc_vpd\Global_test4\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEI{spei_strength}_Cor{cor_test_number}_{ML_model_in_spatio}_TestSize{test_size}_BestParameters_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_analyzeBy({analyze_by}).csv')

                # if grade_by == 'Veg':
                #     Best_parameters.to_csv(
                #         fr'D:\CAU\phenology_swc_vpd\Global_test4\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_RF_Best_parameters_VegType({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).csv')
                # elif grade_by == 'AI':
                #     Best_parameters.to_csv(
                #         fr'D:\CAU\phenology_swc_vpd\Global_test4\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_RF_Best_parameters_AI({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).csv')
                # elif grade_by == 'Cor mean':
                #     Best_parameters.to_csv(
                #         fr'D:\CAU\phenology_swc_vpd\Global_test4\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_RF_Best_parameters_Cormean({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).csv')
                # elif grade_by == 'All':
                #     Best_parameters.to_csv(
                #         fr'D:\CAU\phenology_swc_vpd\Global_test4\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_RF_Best_parameters_All_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).csv')
                print(f'Best_parameters done')

                # ## 使用最优参数训练模型 ##
                # rf = grid_search.best_estimator_

                ######################################################
                # #
                if analyze_by == 'All':
                    Best_parameters = pd.read_csv(
                        fr'D:\CAU\phenology_swc_vpd\Global_test4\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_{ML_model_in_spatio}_TestSize{test_size}_Best_parameters_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).csv')
                elif analyze_by != 'All':
                    Best_parameters = pd.read_csv(
                        fr'D:\CAU\phenology_swc_vpd\Global_test4\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEI{spei_strength}_Cor{cor_test_number}_{ML_model_in_spatio}_TestSize{test_size}_BestParameters_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_analyzeBy({analyze_by}).csv')

                # if grade_by == 'Veg':
                #     Best_parameters = pd.read_csv(
                #         fr'D:\CAU\phenology_swc_vpd\Global_test4\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_RF_Best_parameters_VegType({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).csv')
                # elif grade_by == 'AI':
                #     Best_parameters = pd.read_csv(
                #         fr'D:\CAU\phenology_swc_vpd\Global_test4\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_RF_Best_parameters_AI({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).csv')
                # elif grade_by == 'Cor mean':
                #     Best_parameters = pd.read_csv(
                #         fr'D:\CAU\phenology_swc_vpd\Global_test4\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_RF_Best_parameters_Cormean({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).csv')
                # elif grade_by == 'All':
                #     Best_parameters = pd.read_csv(
                #         fr'D:\CAU\phenology_swc_vpd\Global_test4\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_RF_Best_parameters_All_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).csv')

                n_estimators = int(Best_parameters['n_estimators'][0])
                max_depth = int(Best_parameters['max_depth'][0])
                min_samples_split = int(Best_parameters['min_samples_split'][0])
                min_samples_leaf = int(Best_parameters['min_samples_leaf'][0])
                max_features = Best_parameters['max_features'][0]
                # 如果保存时是数字，才需要转换
                if isinstance(max_features, str):
                    max_features = max_features  # 保持字符串
                else:
                    max_features = float(max_features)

                rf = RandomForestRegressor(n_estimators=n_estimators,
                                         max_depth=max_depth,
                                         min_samples_split=min_samples_split,         #每次分裂会考虑所有特征中 80% 的特征子集
                                         min_samples_leaf=min_samples_leaf,                 # 每棵树的随机采样比例
                                         max_features=max_features,
                                         random_state=42,
                                         n_jobs=20)

                rf.fit(x_train, y_train)
                print(f"rf_model:", rf)

                shap_values_df, ModelImportance_and_ShapValueMean, r2_test = calculate_shap(rf, x_train, y_train, x_test, y_test)


            print('shap_values_df:\n', shap_values_df)
            print('ModelImportance_and_ShapValueMean:\n', ModelImportance_and_ShapValueMean)

            Best_parameters['r2_test'] = r2_test
            if analyze_by == 'All':
                Best_parameters.to_csv(
                    fr'D:\CAU\phenology_swc_vpd\Global_test4\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_{ML_model_in_spatio}_TestSize{test_size}_Best_parameters_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).csv')
            elif analyze_by != 'All':
                Best_parameters.to_csv(
                    fr'D:\CAU\phenology_swc_vpd\Global_test4\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEI{spei_strength}_Cor{cor_test_number}_{ML_model_in_spatio}_TestSize{test_size}_BestParameters_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_analyzeBy({analyze_by}).csv')

            # 将每个y的结果存储到字典中
            results_dict[y] = {
                'df_pcorr': df_pcorr,
                'importance_and_shap_values_mean': ModelImportance_and_ShapValueMean,
                'shap_values_df': shap_values_df,
                'x_list': x_list,
                'y': y
            }


            # srad_std_shap = shap_values_df.values[:, -1]
            print(f"SHAP值统计:")
            print(f"  最小值: {np.nanmin(shap_values_df.values):.1f}")
            print(f"  最大值: {np.nanmax(shap_values_df.values):.1f}")
            print(f"  平均值: {np.nanmean(shap_values_df.values):.1f}")

            # 统计-10到10范围内的占比
            in_range_mask = (shap_values_df >= -1) & (shap_values_df <= 1) & (~np.isnan(shap_values_df))
            total_valid = np.sum(~np.isnan(shap_values_df))
            in_range_count = np.sum(in_range_mask)

            # print(f"[-1, 1]区间统计:\n")
            # print(f"  区间内数量: {in_range_count}")
            # print(f"  有效总数: {total_valid}")
            # print(f"  占比: {(in_range_count / total_valid) * 100:.2f}%")
            #
            # sys.exit()

        ######## ================= 3. plot ====================== #######

        #### 驱动分析结果图绘制 ####
        for y, results in results_dict.items():
            # 使用存储的结果进行绘图
            df_pcorr = results['df_pcorr']
            ModelImportance_and_ShapValueMean = results['importance_and_shap_values_mean']
            shap_values_df = results['shap_values_df']
            x_list = results['x_list']

            # ========== 1. 定义“唯一的 y 轴顺序”（以 SHAP / importance 为准） ==========
            feature_order = ModelImportance_and_ShapValueMean['feature'].tolist()
            print(f'feature_order:{feature_order}')

            # # ========== 2. 创建画布 ==========
            # fig, axes = plt.subplots(1, 3, figsize=(10, 8), sharey=False)
            # ax1, ax2, ax3 = axes
            # plt.subplots_adjust(wspace=0.05)

            # ========== 3. 子图 1：Partial correlation ==========
            df_pcorr_plot = df_pcorr.copy()
            df_pcorr_plot['feature'] = pd.Categorical(
                df_pcorr_plot['feature'],
                categories=feature_order,
                ordered=True
            )
            df_pcorr_plot = df_pcorr_plot.sort_values('feature')
            print(f'df_pcorr_plot:{df_pcorr_plot}')

            colors = ['#4393c3' if r < 0 else '#d6604d' for r in df_pcorr_plot['r_pcorr']]

            sns.barplot(
                x='Abs_Coefficient',
                y='feature',
                data=df_pcorr_plot,
                palette=colors,
                width=0.4,
                ax=ax1,
                alpha=0.8
            )

            ax1.tick_params(axis='both', length=2, pad=3)  # ,which='major', )

            for i, (r, sig) in enumerate(zip(df_pcorr_plot['Abs_Coefficient'],
                                             df_pcorr_plot['Significance'])):
                ax1.text(r + 0.04, i, sig, va='center')#, fontsize=12)

            # if type == 'Forest' or type == 'Arid' or type == 'All':
            #     word = 'a'
            #     if type == 'All':
            #         ytitle = 1.1
            #     elif type == 'Forest':
            #         ytitle = 1.13
            #     elif type == 'Arid':
            #         ytitle = 1.13
            # elif type == 'Shrub' or type == 'Semi-arid':
            #     word = 'd'
            #     ytitle = 1.07
            # elif type == 'Savanna' or type == 'Dry sub-humid':
            #     word = 'g'
            #     ytitle = 1.07
            # elif type == 'Grass' or type == 'Humid':
            #     word = 'j'
            #     ytitle = 1.07
            # ax1.set_title(f'({word}) {type}', y=ytitle)
            if type in ['All', 'Grass', 'Humid', 'Cor(lt-0.4)']:
                ax1.set_xlabel(f'Partial correlation')
            else:
                ax1.set_xlabel('')

            ax1.set_xlim(0, 0.8)
            ax1.set_xticks(np.arange(0, 0.81, 0.2))
            ax1.set_xticklabels([f"{int(x)}" if x.is_integer() else f"{x:.1f}" for x in np.arange(0, 0.81, 0.2)])

            ax1.set_ylabel('')
            ax1.set_yticklabels(['SOS', 'Coupling', 'SM', 'VPD', 'Ta', 'Pre', 'Srad'])
            # ax1.set_yticklabels(['Cor', 'SM', 'VPD', 'Ta', 'Pre', 'Srad'])
            ax1.tick_params(axis='y', length=0)

            # legend_handles = [
            #     mpatches.Patch(color='#4393c3', label='Negative', alpha=0.8),
            #     mpatches.Patch(color='#d6604d', label='Positive', alpha=0.8)
            # ]
            # ax1.legend(handles=legend_handles, frameon=False, fontsize=8, loc='lower right')
            from matplotlib.lines import Line2D

            legend_handles = [
                Line2D([0], [0], linestyle='none', marker='s', markerfacecolor='#4393c3',
                       markeredgecolor='none', markersize=6, label='Negative', alpha=0.8),
                Line2D([0], [0], linestyle='none', marker='s', markerfacecolor='#d6604d',
                       markeredgecolor='none', markersize=6, label='Positive', alpha=0.8)
            ]

            ax1.legend(handles=legend_handles, frameon=False, loc='lower right', fontsize=8 ,
                       handletextpad=0.5, handlelength=1.0)  # 缩短句柄长度使其看起来更像纯方块

            # ========== 4. 子图 2：ML feature importance ==========
            importance_plot = ModelImportance_and_ShapValueMean.copy()

            importance_plot['feature'] = pd.Categorical(
                importance_plot['feature'],
                categories=feature_order,
                ordered=True
            )
            importance_plot = importance_plot.sort_values('feature')
            print(f'importance_plot:{importance_plot}')

            sns.barplot(
                x='importance',
                y='feature',
                data=importance_plot,
                color='#4393c3',
                alpha = 0.8,
                width=0.4,
                ax=ax2
            )

            ax2.tick_params(axis='both', length=2, pad=3)  # ,which='major', )

            # if type == 'Forest' or type == 'Arid' or type == 'All':
            #     word = 'b'
            #     if type == 'All':
            #         ytitle = 1.1
            #     elif type == 'Forest':
            #         ytitle = 1.13
            # elif type == 'Shrub' or type == 'Semi-arid':
            #     word = 'e'
            #     ytitle = 1.07
            # elif type == 'Savanna' or type == 'Dry sub-humid':
            #     word = 'h'
            #     ytitle = 1.07
            # elif type == 'Grass' or type == 'Humid':
            #     word = 'k'
            #     ytitle = 1.07
            # ax2.set_title(f'({word}) {type}', y=ytitle)
            if type in ['All', 'Grass', 'Humid', 'Cor(lt-0.4)']:
                ax2.set_xlabel(f'{ML_model_in_spatio} importance')
            else:
                ax2.set_xlabel('')

            if grade_by == 'All':
                x_min = 0
                x_max = 0.45
                interval = 0.1
            else:
                x_min = 0
                x_max = 0.8
                interval = 0.2
            ax2.set_xlim(x_min, x_max)
            ax2.set_xticks(np.arange(x_min, x_max + 0.01, interval))
            ax2.set_xticklabels([f"{int(x)}" if x.is_integer() else f"{x:.1f}" for x in np.arange(x_min, x_max + 0.01, interval)])

            ax2.set_ylabel('')
            ax2.tick_params(axis='y', length=0, labelright=False, labelleft=False)

            # ========== 5. 子图 3：SHAP beeswarm（关键：用 order 控制顺序） ==========
            plt.sca(ax3)
            ### 1 Shap value sactter
            # —— 构造 SHAP Explanation（按 x_list 顺序）——
            shap_values = shap.Explanation(
                values=shap_values_df[[f'shap_{c}' for c in x_list]].values,
                data=shap_values_df[x_list].values,
                feature_names=x_list
            )

            # —— 将 feature_order 映射为 SHAP 的索引顺序 ——
            order_idx = [x_list.index(f) for f in feature_order if f in x_list]
            print(f'order_idx:{order_idx}')

            shap.plots.beeswarm(
                shap_values,
                order=order_idx,
                max_display=len(order_idx),
                plot_size=None,  # 由外部 ax 控制
                show=False,
                color_bar=False
            )

            # 更改点的颜色
            new_cmap = plt.get_cmap('coolwarm')

            for coll in ax3.collections:
                coll.set_sizes([5])
                # 将现有的颜色数组映射到新的 cmap 上
                # SHAP 内部通常将 feature value 映射在 [0, 1] 之间
                coll.set_cmap(new_cmap)

            # 点大小
            for coll in ax3.collections:
                coll.set_sizes([5])

            # x 轴范围
            # if y == 'POS mean':
            #     ax3.set_xlim(-20, 20)
            # elif y == 'POS slope':
            #     ax3.set_xlim(-1, 1)
            # elif y == 'POS std':
            #     ax3.set_xlim(-5, 5)
            # else:
            ax3.tick_params(axis='both', length=2, pad=3)  # ,which='major', )

            fp = ax2.get_xticklabels()[0].get_fontproperties()
            ax3.set_xlim(-1.5, 1.5)
            ax3.set_xticks(np.arange(-1.5, 1.51, 0.5))
            ax3.set_xticklabels([f"{int(x)}" if x.is_integer() else f"{x:.1f}" for x in np.arange(-1.5, 1.51, 0.5)], fontproperties=fp)

            # if type == 'Forest' or type == 'Arid' or type == 'All':
            #     word = 'c'
            #     if type == 'All':
            #         ytitle = 1.1
            #     elif type == 'Forest':
            #         ytitle = 1.13
            # elif type == 'Shrub' or type == 'Semi-arid':
            #     word = 'f'
            #     ytitle = 1.07
            # elif type == 'Savanna' or type == 'Dry sub-humid':
            #     word = 'i'
            #     ytitle = 1.07
            # elif type == 'Grass' or type == 'Humid':
            #     word = 'l'
            #     ytitle = 1.07
            # ax3.set_title(f'({word}) {type}', y=ytitle)
            fp = ax2.xaxis.label.get_fontproperties()
            if type in ['All', 'Grass', 'Humid', 'Cor(lt-0.4)']:
                ax3.set_xlabel(f'{ML_model_in_spatio} SHAP value', fontproperties = fp)#, labelpad=4)
            else:
                ax3.set_xlabel('')
            ax3.tick_params(axis='y', labelleft=False)

            ### 2 Shap abs mean

            ax3_twin = ax3.twiny()
            ax3_twin.set_ylim(ax1.get_ylim())

            shapvalue_mean = ModelImportance_and_ShapValueMean.copy()

            shapvalue_mean['feature'] = pd.Categorical(
                shapvalue_mean['feature'],
                categories=feature_order[::-1],
                ordered=True
            )
            shapvalue_mean = shapvalue_mean.sort_values('feature')
            print(f'shapvalue_mean:{shapvalue_mean}')
            sns.barplot(
                x='shapvalue',
                y='feature',
                data=shapvalue_mean,
                color='gray',
                width=0.4,
                alpha=0.3,
                ax=ax3_twin,
                zorder=0
            )

            # 设置顶层 X 轴的范围和标签
            ax3_twin.set_xlim(0, 0.3)  # 给条形图留出余量
            ax3_twin.set_xticks(np.arange(0, 0.301, 0.1))  # 给条形图留出余量
            ax3_twin.set_xticklabels('0' if x == 0 else
                                    f'{x:.1f}' for x in np.arange(0, 0.301, 0.1))  # 给条形图留出余量
            ax3_twin.tick_params(axis='x', length=2, pad=1)
            if type in ['All', 'Forest', 'Arid', 'Cor(-0.1~0)']:
                ax3_twin.set_xlabel(f'mean |{ML_model_in_spatio} SHAP value|', labelpad = 6)
            elif veg_type == 'Grass' and grade_by == 'Veg':
                ax3_twin.set_xlabel(f'mean |{ML_model_in_spatio} SHAP value|', labelpad=6)
            else:
                ax3_twin.set_xlabel('')



            # # 隐藏右侧和多余的 Y 轴标签
            # ax3_twin.spines['right'].set_visible(False)
            # ax3.spines['right'].set_visible(False)
            # ax3_twin.get_yaxis().set_visible(False)  # 隐藏 twin 轴的 Y 标签，防止重叠

            # for ax, name in zip([ax2, ax3], ['ax2', 'ax3']):
            #     t = ax.get_xticklabels()[0]
            #     print(name)
            #     print("family:", t.get_fontfamily())
            #     print("size:", t.get_fontsize())
            #     print("weight:", t.get_fontweight())
            #     print("style:", t.get_fontstyle())
            #     print("name:", t.get_fontproperties().get_name())


            ### ================ 6. 子图 4：子图3的color ================ ###
            norm = mpl.colors.Normalize(vmin=0, vmax=1)

            cb = fig.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=new_cmap),
                cax=ax4,  # 确保使用预留的 ax4
                orientation='vertical'
            )

            cb.ax.tick_params(
                axis='y',  # 针对 y 轴（Colorbar 的纵向轴）
                length=0,  # 刻度线长度设为 0（隐藏刻度线）
                pad=2  # 标签与色条的间距，可根据需要微调
            )
            cb.set_ticks([0, 1])
            cb.set_ticklabels(['Low', 'High'])
            cb.set_label('Feature value', labelpad=-7)
            cb.outline.set_visible(False)  # 去掉 colorbar 的黑色边框显得更高级

            pos = ax4.get_position()

            new_width = pos.width * 0.5

            ax4.set_position([pos.x0-0.01, pos.y0, new_width, pos.height])

            # ========== 6. 统一边框风格 ==========
            for ax in [ax1, ax2, ax3]:
                for spine in ax.spines.values():
                    spine.set_linewidth(1)
                    spine.set_edgecolor('black')


            # ---------------------- 统一对齐 y 轴 ----------------------
            # 1. 以 ax1 为基准
            y_min, y_max = ax1.get_ylim()
            y_lim = (min(y_min, y_max), max(y_min, y_max))  # 确保正向

            # 2. 获取 ax1 的 ticks
            yticks = ax1.get_yticks()

            # 3. 循环应用到 ax2 和 ax3
            for ax in [ax3_twin]:
                # 设置相同的 ylim（正向保证）
                ax.set_ylim(y_lim)

                # 设置相同的 ticks
                ax.set_yticks(yticks)
                # ax.invert_yaxis()

                # 可选：保持 tick 标签与 ax1 一致
                # ticklabel 可以使用字符串或数值
                # yticklabels = [str(round(t, 2)) for t in yticks]  # 保留两位小数
                # ax.set_yticklabels(yticklabels)

            # base_ylim = ax1.get_ylim()
            #
            # # 2. 强制覆盖所有相关轴
            # for target_ax in [ax2, ax3, ax3_twin]:
            #     target_ax.set_ylim(base_ylim)
            #
            # # 3. 检查反转状态（关键！）
            # # 如果 ax1 是正常的（0在下，N在上），而 SHAP 点在 ax3 里还是反的
            # # 那么我们只需要反转 ax3 和它的 twin 轴
            # if ax3.yaxis_inverted() != ax1.yaxis_inverted():
            #     ax3.invert_yaxis()
            #     ax3_twin.invert_yaxis()  # 必须两个一起反转
            # ax3_twin.get_yaxis().set_visible(False)


            # ========== 7. 保存 ==========

            # plt.tight_layout()
            # fig_path = rf'D:\CAU\phenology_swc_vpd\Global_test4\Fig\6Partial\SM_VPD_Cor17\In spatio\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SM_VPD_Cor{cor_test_number}_Global_{grade_by}_{ML_model_in_spatio}_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).png'
            # elif grade_by == 'All':
            #     fig_path = rf'D:\CAU\phenology_swc_vpd\Global_test4\Fig\6Partial\SM_VPD_Cor17\In spatio\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SM_VPD_Cor{cor_test_number}_Global_All_{ML_model_in_spatio}_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).png'

    if grade_by in ['Veg', 'AI', 'Cor mean']:
        # plt.tight_layout()
        if analyze_by == 'All':
            fig_path = rf'D:\CAU\phenology_swc_vpd\Global_test5\Fig\Fig 6 Driver of POS difference in drought event\{ML_model_in_spatio}_by_all_drought_event\{grade_by}\DroughtEventTimes({drought_or_wet_times})_SPEIstrength({spei_strength})_SM_VPD_Cor{cor_test_number}_Global_{grade_by}_{ML_model_in_spatio}_TestSize({test_size})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).png'
        elif analyze_by == 'advance PPT':
            fig_path = rf'D:\CAU\phenology_swc_vpd\Global_test5\Fig\Fig 6 Driver of POS difference in drought event\{ML_model_in_spatio}_by_advancedPPT_drought_event\{grade_by}\DroughtEventTimes({drought_or_wet_times})_SPEIstrength({spei_strength})_SM_VPD_Cor{cor_test_number}_Global_{grade_by}_{ML_model_in_spatio}_TestSize({test_size})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).png'
        elif analyze_by == 'delay PPT':
            fig_path = rf'D:\CAU\phenology_swc_vpd\Global_test5\Fig\Fig 6 Driver of POS difference in drought event\{ML_model_in_spatio}_by_delayedPPT_drought_event\{grade_by}\DroughtEventTimes({drought_or_wet_times})_SPEIstrength({spei_strength})_SM_VPD_Cor{cor_test_number}_Global_{grade_by}_{ML_model_in_spatio}_TestSize({test_size})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).png'
        plt.savefig(
            fig_path,
            dpi=300, bbox_inches='tight')
        print(f'Fig保存到：{fig_path}')

    # plt.show()


### S23-25: Distrition of Drought/Wet and normal diff
def plot_pos(drought_difference_data, wet_difference_data, colorbarmin, colorbarmax, grade_by, ax):

    if grade_by != 'All':
        # 统一设置所有字体大小
        plt.rcParams.update({
            'font.family': 'Arial',

            'mathtext.fontset': 'custom',

            'mathtext.rm': 'Arial',  # 正常
            'mathtext.it': 'Arial:italic',  # 斜体
            'mathtext.bf': 'Arial:bold',  # 粗体

            # 可选（推荐加）
            'mathtext.default': 'regular',  # 避免自动变斜体

            'font.size': 10,
            'axes.titlesize': 10,
            'axes.labelsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            # 'text.usetex': False,  # 不使用外部LaTeX
        })

    if grade_by == 'All':
        fig = ax.figure
        gs = ax.get_subplotspec().subgridspec(1, 3,
                                            width_ratios=[5, 5, 4],  # 三列的宽度比
                                            height_ratios=[1],  # 最后一个给colorbar
                                            wspace=0.25)
        ax.axis('off')

        drought_data_list = [drought_difference_data]
        wet_data_list = [wet_difference_data]
        labels = ['All']

    elif grade_by == 'Veg' or grade_by == 'AI':

        fig = plt.figure(figsize=(8.6, 12.5))
        gs = gridspec.GridSpec(4, 4,
                               width_ratios=[5, 5, 0.04, 4],  # 三列的宽度比
                               height_ratios=[1, 1, 1, 1],  # 最后一个给colorbar
                               hspace=0.25, wspace=0.25)

        if grade_by == 'Veg':
            drought_data_list = [np.where(veg_type_data == i, drought_difference_data, np.nan) for i in [1, 2, 3, 4]]
            wet_data_list = [np.where(veg_type_data == i, wet_difference_data, np.nan) for i in [1, 2, 3, 4]]
            labels = ['Forest', 'Shrub', 'Savanna', 'Grass']
        elif grade_by == 'AI':
            drought_data_list = [
                np.where(ai_type_data == 2, drought_difference_data, np.nan),  # Arid
                np.where((ai_type_data == 3) | (ai_type_data == 4), drought_difference_data, np.nan),  # Semi-Arid (合并 3 和 4)
                np.where(ai_type_data == 5, drought_difference_data, np.nan),  # Dry sub-humid
                np.where(ai_type_data == 6, drought_difference_data, np.nan)  # Humid
            ]
            wet_data_list = [
                np.where(ai_type_data == 2, wet_difference_data, np.nan),  # Arid
                np.where((ai_type_data == 3) | (ai_type_data == 4), wet_difference_data, np.nan),
                # Semi-Arid (合并 3 和 4)
                np.where(ai_type_data == 5, wet_difference_data, np.nan),  # Dry sub-humid
                np.where(ai_type_data == 6, wet_difference_data, np.nan)  # Humid
            ]
            labels = ['Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']

    elif grade_by == 'Cor mean':

        fig = plt.figure(figsize=(8.6, 15.5))
        gs = gridspec.GridSpec(5, 4,
                               width_ratios=[5, 5, 0.03, 4],  # 三列的宽度比
                               height_ratios=[1, 1, 1, 1, 1],  # 最后一个给colorbar
                               hspace=0.25, wspace=0.25)

        drought_data_list = [
            np.where((cor_mean_data >= -0.1) & (cor_mean_data < 0), drought_difference_data, np.nan),
            np.where((cor_mean_data >= -0.2) & (cor_mean_data < -0.1), drought_difference_data, np.nan),
            np.where((cor_mean_data >= -0.3) & (cor_mean_data < -0.2), drought_difference_data, np.nan),
            np.where((cor_mean_data >= -0.4) & (cor_mean_data < -0.3), drought_difference_data, np.nan),
            np.where((cor_mean_data < -0.4), drought_difference_data, np.nan)
        ]
        wet_data_list = [
            np.where((cor_mean_data >= -0.1) & (cor_mean_data < 0), wet_difference_data, np.nan),
            np.where((cor_mean_data >= -0.2) & (cor_mean_data < -0.1), wet_difference_data, np.nan),
            np.where((cor_mean_data >= -0.3) & (cor_mean_data < -0.2), wet_difference_data, np.nan),
            np.where((cor_mean_data >= -0.4) & (cor_mean_data < -0.3), wet_difference_data, np.nan),
            np.where((cor_mean_data < -0.4), wet_difference_data, np.nan)
        ]

        labels = [  # 'Cor(<-0.5)',
            'Cor(-0.1~0)', 'Cor(-0.2~-0.1)', 'Cor(-0.3~-0.2)',
            'Cor(-0.4~-0.3)', 'Cor(lt-0.4)']

    plots = []  # 存储每个子图的plot对象



    for i, (drought_data, wet_data, name)  in enumerate(zip(drought_data_list, wet_data_list, labels)):
        if grade_by == 'All':
            pos_ax1 = fig.add_subplot(gs[0, 0])  # 地图
            pos_ax2 = fig.add_subplot(gs[0, 1])  # 纬度曲线
            pos_ax3 = fig.add_subplot(gs[0, 2])  # Colorbar横跨两列
        else:
            pos_ax1 = plt.subplot(gs[i, 0])
            pos_ax2 = plt.subplot(gs[i, 1])
            pos_ax3 = plt.subplot(gs[i, 3])

        print(f'{name}--pos_diff_drought_normal有效像元数量：{np.count_nonzero(np.isfinite(drought_data))}')
        print(f'{name}--pos_diff_drought_normal有效像元数量占比：{(np.count_nonzero(np.isfinite(drought_data))/np.count_nonzero(np.isfinite(pos_diff_drought_normal)))*100}%')
    #
        ###### 子图1：干旱与正常POS差异空间分布
        plot_cor_mean_or_slope_and_pvalue_forAllvegType(drought_data,  colorbarmin, colorbarmax, 'drought', name, pos_ax1)
        plot_cor_mean_or_slope_and_pvalue_forAllvegType(wet_data,  colorbarmin, colorbarmax,'wet', name, pos_ax2)
        plot_cor_mean_or_slope_lt0_sig_and_lt0_nosig_and_gt0_forAllvegType(drought_data, wet_data, -12, 12, name, pos_ax3)

    if grade_by != 'All':
        # plt.tight_layout()
        # plt.savefig(fr'D:\CAU\phenology_swc_vpd\Global_test4\Fig\5Drought_event\POS_diff\SPEI{spei_length}\{grade_by}\POSmean_difference(drought and normal)_way3.png', dpi=600, bbox_inches='tight')
        if analyze_by == 'All':
            plt.savefig(fr'D:\CAU\phenology_swc_vpd\Global_test5\Fig\Fig 6 Driver of POS difference in drought event\{ML_model_in_spatio}_by_all_drought_event\{grade_by}\DroughtEventTimes({drought_or_wet_times})_SPEIstrength({spei_strength})_POSmean_difference(drought and normal)_way3_CorSig({SigCorPvalue}).png', dpi=300, bbox_inches='tight')
        elif analyze_by == 'advance PPT':
            plt.savefig(
                fr'D:\CAU\phenology_swc_vpd\Global_test5\Fig\Fig 6 Driver of POS difference in drought event\{ML_model_in_spatio}_by_advancedPPT_drought_event\{grade_by}\DroughtEventTimes({drought_or_wet_times})_SPEIstrength({spei_strength})_POSmean_difference(drought and normal)_way3_CorSig({SigCorPvalue}).png',
                dpi=300, bbox_inches='tight')
        elif analyze_by == 'delay PPT':
            plt.savefig(
                fr'D:\CAU\phenology_swc_vpd\Global_test5\Fig\Fig 6 Driver of POS difference in drought event\{ML_model_in_spatio}_by_delayedPPT_drought_event\{grade_by}\DroughtEventTimes({drought_or_wet_times})_SPEIstrength({spei_strength})_POSmean_difference(drought and normal)_way3_CorSig({SigCorPvalue}).png',
                dpi=300, bbox_inches='tight')

    # plt.show()


### Fig 6
def plot_fig6(drought_data, wet_data,
              colorbarmin, colorbarmax,
              pos_data, sos_data, cor_data, sm_data, vpd_data, ta_data, pre_data, srad_data):

    # 统一设置所有字体大小
    plt.rcParams.update({
        'font.family': 'Arial',

        'mathtext.fontset': 'custom',

        'mathtext.rm': 'Arial',  # 正常
        'mathtext.it': 'Arial:italic',  # 斜体
        'mathtext.bf': 'Arial:bold',  # 粗体

        # 可选（推荐加）
        'mathtext.default': 'regular',  # 避免自动变斜体

        'font.size': 10,
        'axes.titlesize': 10,
        'axes.labelsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        # 'text.usetex': False,  # 不使用外部LaTeX
    })

    fig = plt.figure(figsize=(8.4, 7))
    gs = gridspec.GridSpec(2, 3,
                           width_ratios=[0.5, 4, 0.5],  # 三列的宽度比
                           height_ratios=[1, 1],  # 最后一个给colorbar
                           hspace=0.7, wspace=0.15)


    ### a Drought and normal diff
    combine_ax1 = plt.subplot(gs[0, :])
    plot_pos(drought_data, wet_data, colorbarmin, colorbarmax,'All', combine_ax1)

    ### d Driving factor analysis of All pixels
    combine_ax2 = plt.subplot(gs[1, 1])
    Partial_ML_calculate_and_plot(pos_data, sos_data, cor_data, sm_data, vpd_data, ta_data, pre_data, srad_data, 'All', 'All', combine_ax2)

    # ### d Driving factor analysis of only grassland
    # combine_ax3 = plt.subplot(gs[1, 1])
    # Partial_ML_calculate_and_plot(pos_data, sos_data, cor_data, sm_data, vpd_data, ta_data, pre_data, srad_data, 'Veg', 'Grass', combine_ax3)

    # plt.savefig(fr'D:\CAU\phenology_swc_vpd\Global_test4\Fig\6Partial\SM_VPD_Cor17\In spatio\SPEI{spei_length}\Climate{climate_test_number}\Based_on_detrendPheno\POSdiff_combined_SM_VPD_Cor17_8_{months_before_pos}_Global_All_XGBoost_POS drought normal diff_Outlier({Outlier})_CorSig({SigCorPvalue}).png',
    if analyze_by == 'All':
        plt.savefig(fr'{output_path}\DroughtEventTimes({drought_or_wet_times})_SPEIstrength({spei_strength})_POSdiff_combined_SM_VPD_Cor17_8_{months_before_pos}_Global_All_{ML_model_in_spatio}_TestSize{test_size}_POS drought normal diff_Outlier({Outlier})_CorSig({SigCorPvalue}).png',
                    dpi=300, bbox_inches='tight')
    elif analyze_by == 'advance PPT':
        plt.savefig(fr'{output_path}\DroughtTimes({drought_or_wet_times})_SPEI({spei_strength})_POSdiff_combined_SM_VPD_Cor17_8_{months_before_pos}_All_{ML_model_in_spatio}_TestSize{test_size}_POS drought normal diff_Outlier({Outlier})_CorSig({SigCorPvalue}).png',
                    dpi=300, bbox_inches='tight')
    elif analyze_by == 'delay PPT':
        plt.savefig(fr'{output_path}\DroughtTimes({drought_or_wet_times})_SPEI({spei_strength})_POSdiff_combined_SM_VPD_Cor17_8_{months_before_pos}_All_{ML_model_in_spatio}_TestSize{test_size}_POS drought normal diff_Outlier({Outlier})_CorSig({SigCorPvalue}).png',
                    dpi=300, bbox_inches='tight')

    print('Fig 6 plot done!')

    # plt.show()




### 控制 干旱事件次数 + 干旱强度
pos_mean_difference_indrought_Nodrought = np.where((drought_event_count>=drought_or_wet_times), pos_diff_drought_normal_origin, np.nan)
pos_mean_difference_inwet_Nowet = np.where((wet_event_count>=drought_or_wet_times), pos_diff_wet_normal_origin, np.nan)

print(f'pos_mean_difference_indrought_Nodrought有效像元数量：{np.count_nonzero(np.isfinite(pos_mean_difference_indrought_Nodrought))}')
print(f'pos_mean_difference_inwet_Nowet有效像元数量：{np.count_nonzero(np.isfinite(pos_mean_difference_inwet_Nowet))}')

if analyze_by == 'All':
    mask = drought_event_count>=drought_or_wet_times
elif analyze_by == 'advance PPT':
    mask = (drought_event_count >= drought_or_wet_times) & (pos_mean_difference_indrought_Nodrought < 0)
elif analyze_by == 'delay PPT':
    mask = (drought_event_count >= drought_or_wet_times) & (pos_mean_difference_indrought_Nodrought > 0)


pos_diff_drought_normal = np.where(mask, pos_diff_drought_normal, np.nan)
sos_diff_drought_normal = np.where(mask, sos_diff_drought_normal, np.nan)
cor_diff_drought_normal = np.where(mask, cor_diff_drought_normal, np.nan)
sm_diff_drought_normal = np.where(mask, sm_diff_drought_normal, np.nan)
vpd_diff_drought_normal = np.where(mask, vpd_diff_drought_normal, np.nan)
ta_diff_drought_normal = np.where(mask, ta_diff_drought_normal, np.nan)
pre_diff_drought_normal = np.where(mask, pre_diff_drought_normal, np.nan)
srad_diff_drought_normal = np.where(mask, srad_diff_drought_normal, np.nan)

# sys.exit()
## Fig 6
plot_fig6(pos_mean_difference_indrought_Nodrought, pos_mean_difference_inwet_Nowet,
          -12, 12,
          pos_diff_drought_normal, sos_diff_drought_normal,
          cor_diff_drought_normal, sm_diff_drought_normal, vpd_diff_drought_normal,
          ta_diff_drought_normal, pre_diff_drought_normal, srad_diff_drought_normal)
print('Fig6 or S22 plot done!')

# # # Fig S 23 - 25
# ax = plt.figure(figsize=(17, 11))
# plot_pos(pos_mean_difference_indrought_Nodrought, pos_mean_difference_inwet_Nowet, -12, 12, 'Veg', ax)
# print('S23 plot done!')
# plot_pos(pos_mean_difference_indrought_Nodrought, pos_mean_difference_inwet_Nowet, -12, 12, 'AI', ax)
# print('S24 plot done!')
# plot_pos(pos_mean_difference_indrought_Nodrought, pos_mean_difference_inwet_Nowet, -12, 12, 'Cor mean', ax)
# print('S25 plot done!')

# # # Fig S 26 - 27
# ax = plt.figure(figsize=(17, 11))
# Partial_ML_calculate_and_plot(pos_diff_drought_normal, sos_diff_drought_normal,
#                               cor_diff_drought_normal, sm_diff_drought_normal, vpd_diff_drought_normal,
#                               ta_diff_drought_normal, pre_diff_drought_normal, srad_diff_drought_normal,
#                               'Veg', 'All', ax)
# print('S26 plot done!')
#
# Partial_ML_calculate_and_plot(pos_diff_drought_normal, sos_diff_drought_normal,
#                               cor_diff_drought_normal, sm_diff_drought_normal, vpd_diff_drought_normal,
#                               ta_diff_drought_normal, pre_diff_drought_normal, srad_diff_drought_normal,
#                               'AI', 'All', ax)
# print('S27 plot done!')
#
# Partial_ML_calculate_and_plot(pos_diff_drought_normal, sos_diff_drought_normal,
#                               cor_diff_drought_normal, sm_diff_drought_normal, vpd_diff_drought_normal,
#                               ta_diff_drought_normal, pre_diff_drought_normal, srad_diff_drought_normal,
#                               'Cor mean', 'All', ax)
# print('S28 plot done!')
