
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



######### Function ##########

### 读取时间信息，提取每个 TIF 文件的日期（最后8位）
def extract_date_from_filename(filename):
    # 提取纯文件名（不含路径和扩展名）
    basename = os.path.basename(filename)
    filename_without_ext = os.path.splitext(basename)[0]

    # 截取最后8位
    date_str = filename_without_ext[-8:]

    if date_str[-1] == '1':
        # 最后一个字符是'1'，将日期设为1号
        year_month = date_str[:6]  # 获取年月部分
        formatted_date = f"{year_month}01"  # 添加01作为日期
        # print(f'formatted_date:{formatted_date}')
        return datetime.datetime.strptime(formatted_date, "%Y%m%d")

    elif date_str[-1] == '2':
        # 最后一个字符是'2'，将日期设为2号
        year_month = date_str[:6]  # 获取年月部分
        formatted_date = f"{year_month}16"  # 添加02作为日期
        # print(f'formatted_date:{formatted_date}')
        return datetime.datetime.strptime(formatted_date, "%Y%m%d")


def get_band(tif_file, stack):
    tif = gdal.Open(tif_file)
    data = tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
    stack.append(data)



def calculate_partial(y, x_idx, covars_list, min_years=12):
    """
    计算 y 与 covars_list[x_idx] 的偏相关，控制 covars_list 中其余所有变量
    """
    # 转换为数组
    y = np.asarray(y, dtype=float)
    X_all = np.column_stack([np.asarray(c, dtype=float) for c in covars_list])

    # 确定当前的目标自变量和控制变量
    x = X_all[:, x_idx]
    Z = np.delete(X_all, x_idx, axis=1)  # 删除当前 x，剩下的作为控制变量

    # 掩码：排除 NaN 和 Inf
    valid = np.isfinite(y) & np.isfinite(x) & np.all(np.isfinite(Z), axis=1)

    # 自由度检查 (n > k + 2)
    min_required = max(min_years, Z.shape[1] + 2)

    if np.sum(valid) < min_required:
        return np.nan

    x = x[valid]
    y = y[valid]
    Z = Z[valid]

    # 回归残差法计算偏相关
    Z_ = np.column_stack([np.ones(len(Z)), Z])

    # x ~ Z 的残差
    beta_x, _, _, _ = np.linalg.lstsq(Z_, x, rcond=None)
    rx = x - Z_ @ beta_x

    # y ~ Z 的残差
    beta_y, _, _, _ = np.linalg.lstsq(Z_, y, rcond=None)
    ry = y - Z_ @ beta_y

    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan

    return np.corrcoef(rx, ry)[0, 1]



def cal_best_perseason_pixel(i, j, pos_stack_pixel, sos_stack_pixel,
                             cor_data,
                             sm_data,
                             vpd_data,
                             ta_data,
                             pre_data,
                             srad_data):

    pheno_valid_mask = np.isfinite(pos_stack_pixel) & np.isfinite(sos_stack_pixel)
    if np.sum(pheno_valid_mask) < 12:
        return (i, j, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    # var_names = ['Cor', 'SM', 'VPD', 'Ta', 'Pre', 'Srad']

    pcor_matrix = np.full((6, 3), np.nan)

    for win_idx in range(3):  # 对应 30, 60, 90 天
        # 准备该窗口下的协变量列表
        current_covars = [
            sos_stack_pixel,  # 0
            cor_data[win_idx, :],  # 1
            sm_data[win_idx, :],  # 2
            vpd_data[win_idx, :],  # 3
            ta_data[win_idx, :],  # 4
            pre_data[win_idx, :],  # 5
            srad_data[win_idx, :]  # 6
        ]

        # 计算该窗口下，每个气象变量与 POS 的偏相关
        for v_idx in range(6):
            # v_idx 从 0 开始，但在 current_covars 中对应的索引是 v_idx + 1
            pcor_matrix[v_idx, win_idx] = calculate_partial(
                y=pos_stack_pixel,
                x_idx=v_idx + 1,
                covars_list=current_covars
            )

    best_lens = []
    for v_idx in range(6):
        vals = pcor_matrix[v_idx, :]
        if np.any(np.isfinite(vals)):
            # 这里选取绝对值最大的索引，+1 是为了转为 1, 2, 3
            best_lens.append(float(np.nanargmax(np.abs(vals)) + 1))
        else:
            best_lens.append(np.nan)

    print(f'i={i}, j={j} pcor_matrix:\n'
          f'{pcor_matrix}\n'
          f'cor_bl={best_lens[0]}\n'
          f'sm_bl={best_lens[1]}\n'
          f'vpd_bl={best_lens[2]}\n'
          f'ta_bl={best_lens[3]}\n'
          f'pre_bl={best_lens[4]}\n'
          f'srad_bl={best_lens[5]}\n')

    return (i, j, *best_lens)  #* 号解包列表


def save_tif_gdal(output_path, data, crs, transform):
    """保存TIFF文件，自动获取数据尺寸并应用地理变换"""
    rows, cols = data.shape
    driver = gdal.GetDriverByName("GTiff")

    output_ds = driver.Create(
        output_path,
        cols, rows, 1, gdal.GDT_Float32
    )
    if not output_ds:
        raise RuntimeError(f"无法创建输出文件: {output_path}")

    output_band = output_ds.GetRasterBand(1)
    output_band.WriteArray(data, 0, 0)
    output_band.SetNoDataValue(np.nan)  # 设置NaN值

    output_ds.SetProjection(crs)
    output_ds.SetGeoTransform(transform)  # 使用调整后的变换参数
    output_ds = None
    return True




###################################### 1 数据读取及输出设定 ################################################
###################### ===================== 输入设定 ======================== ########################
#### 输入的SM和VPD的tif    ### 请仔细修改这里 ⬇⬇⬇⬇⬇⬇⬇⬇

star_year = 2001
end_year = 2024
years_length = end_year - star_year + 1
print('years_length:', years_length)
years = range(star_year, end_year + 1)

scale = 55

input_same_path = 'D:\CAU\phenology_swc_vpd\paper\Data_and_code_online\FigShare_data'

Basedon = 'Based_on_detrendPheno'  ### Based_on_detrendPheno 意思用去趋势的SOS、POS来做偏相关
                                   ### Based_on_OriginPheno 意思用原始的SOS、POS来做偏相关

folder_cor_1 = fr'{input_same_path}\{scale}km\Climate_data\Correlation(SM_VPD_pearson)1'  #(POS-30) - POS
folder_cor_2 = fr'{input_same_path}\{scale}km\Climate_data\Correlation(SM_VPD_pearson)2'  #(POS-60) - POS
folder_cor_3 = fr'{input_same_path}\{scale}km\Climate_data\Correlation(SM_VPD_pearson)3'  #(POS-90) - POS

folder_sm_1 = fr'{input_same_path}\{scale}km\Climate_data\SM_preseason_mean1'  #(POS-30) - POS
folder_sm_2 = fr'{input_same_path}\{scale}km\Climate_data\SM_preseason_mean2'  #(POS-60) - POS
folder_sm_3 = fr'{input_same_path}\{scale}km\Climate_data\SM_preseason_mean3'  #(POS-90) - POS

folder_vpd_1 = fr'{input_same_path}\{scale}km\Climate_data\VPD_preseason_mean1'  #(POS-30) - POS
folder_vpd_2 = fr'{input_same_path}\{scale}km\Climate_data\VPD_preseason_mean2'  #(POS-60) - POS
folder_vpd_3 = fr'{input_same_path}\{scale}km\Climate_data\VPD_preseason_mean3'  #(POS-90) - POS

folder_ta_1 = fr'{input_same_path}\{scale}km\Climate_data\Ta_preseason_mean1'
folder_ta_2 = fr'{input_same_path}\{scale}km\Climate_data\Ta_preseason_mean2'
folder_ta_3 = fr'{input_same_path}\{scale}km\Climate_data\Ta_preseason_mean3'

folder_pre_1 = fr'{input_same_path}\{scale}km\Climate_data\Pre_preseason_sum1'
folder_pre_2 = fr'{input_same_path}\{scale}km\Climate_data\Pre_preseason_sum2'
folder_pre_3 = fr'{input_same_path}\{scale}km\Climate_data\Pre_preseason_sum3'

folder_srad_1 = fr'{input_same_path}\Climate_data\Srad_preseason_sum1'
folder_srad_2 = fr'{input_same_path}\Climate_data\Srad_preseason_sum2'
folder_srad_3 = fr'{input_same_path}\Climate_data\Srad_preseason_sum3'


#### 输入的POS和SOS
if Basedon == 'Based_on_detrendPheno':
    pos_folder = fr'{input_same_path}\POSdetrend_55km'  #start
    sos_folder = fr'{input_same_path}\SOSdetrend_55km'  #start
elif Basedon == 'Based_on_OriginPheno':
    pos_folder = fr'{input_same_path}\POS_55km'  # start
    sos_folder = fr'{input_same_path}\SOS_55km'  # start


###################### ===================== 输出设定 ======================== ########################
ouput_path = rf'{input_same_path}\Climate_data\Best_preseason_length\17_8_1'

####################################################################################

tif_files_cor_1 = sorted(glob.glob(os.path.join(folder_cor_1, '*.tif')))
tif_files_cor_2 = sorted(glob.glob(os.path.join(folder_cor_2, '*.tif')))
tif_files_cor_3 = sorted(glob.glob(os.path.join(folder_cor_3, '*.tif')))

tif_files_sm_1 = sorted(glob.glob(os.path.join(folder_sm_1, '*.tif')))
tif_files_sm_2 = sorted(glob.glob(os.path.join(folder_sm_2, '*.tif')))
tif_files_sm_3 = sorted(glob.glob(os.path.join(folder_sm_3, '*.tif')))

tif_files_vpd_1 = sorted(glob.glob(os.path.join(folder_vpd_1, '*.tif')))
tif_files_vpd_2 = sorted(glob.glob(os.path.join(folder_vpd_2, '*.tif')))
tif_files_vpd_3 = sorted(glob.glob(os.path.join(folder_vpd_3, '*.tif')))

tif_files_ta_1 = sorted(glob.glob(os.path.join(folder_ta_1, '*.tif')))
tif_files_ta_2 = sorted(glob.glob(os.path.join(folder_ta_2, '*.tif')))
tif_files_ta_3 = sorted(glob.glob(os.path.join(folder_ta_3, '*.tif')))

tif_files_pre_1 = sorted(glob.glob(os.path.join(folder_pre_1, '*.tif')))
tif_files_pre_2 = sorted(glob.glob(os.path.join(folder_pre_2, '*.tif')))
tif_files_pre_3 = sorted(glob.glob(os.path.join(folder_pre_3, '*.tif')))

tif_files_srad_1 = sorted(glob.glob(os.path.join(folder_srad_1, '*.tif')))
tif_files_srad_2 = sorted(glob.glob(os.path.join(folder_srad_2, '*.tif')))
tif_files_srad_3 = sorted(glob.glob(os.path.join(folder_srad_3, '*.tif')))

pos_tif_files = sorted(glob.glob(os.path.join(pos_folder, '*.tif')))
sos_tif_files = sorted(glob.glob(os.path.join(sos_folder, '*.tif')))

if not tif_files_sm_1:
    raise FileNotFoundError("未找到任何 tif_files_sm_1 TIF 文件！")
if not tif_files_sm_2:
    raise FileNotFoundError("未找到任何 tif_files2 TIF 文件！")
if not tif_files_sm_3:
    raise FileNotFoundError("未找到任何 tif_files3 TIF 文件！")
if not pos_tif_files:
    raise FileNotFoundError("未找到任何 POS TIF 文件！")


####################### 2 提取tif信息 ############################
first_tif = tif_files_sm_1[0]
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


############################ 3 堆叠 #############################
print('All stack start!')

## 数据堆叠
cor_stack_1 = []
cor_stack_2 = []
cor_stack_3 = []

sm_stack_1 = []
sm_stack_2 = []
sm_stack_3 = []

vpd_stack_1 = []
vpd_stack_2 = []
vpd_stack_3 = []

ta_stack_1 = []
ta_stack_2 = []
ta_stack_3 = []

pre_stack_1 = []
pre_stack_2 = []
pre_stack_3 = []

srad_stack_1 = []
srad_stack_2 = []
srad_stack_3 = []

pos_stack = []
sos_stack = []

for tif_file in tif_files_cor_1:
    get_band(tif_file, cor_stack_1)
for tif_file in tif_files_cor_2:
    get_band(tif_file, cor_stack_2)
for tif_file in tif_files_cor_3:
    get_band(tif_file, cor_stack_3)

for tif_file in tif_files_sm_1:
    get_band(tif_file, sm_stack_1)
for tif_file in tif_files_sm_2:
    get_band(tif_file, sm_stack_2)
for tif_file in tif_files_sm_3:
    get_band(tif_file, sm_stack_3)

for tif_file in tif_files_vpd_1:
    get_band(tif_file, vpd_stack_1)
for tif_file in tif_files_vpd_2:
    get_band(tif_file, vpd_stack_2)
for tif_file in tif_files_vpd_3:
    get_band(tif_file, vpd_stack_3)

for tif_file in tif_files_ta_1:
    get_band(tif_file, ta_stack_1)
for tif_file in tif_files_ta_2:
    get_band(tif_file, ta_stack_2)
for tif_file in tif_files_ta_3:
    get_band(tif_file, ta_stack_3)

for tif_file in tif_files_pre_1:
    get_band(tif_file, pre_stack_1)
for tif_file in tif_files_pre_2:
    get_band(tif_file, pre_stack_2)
for tif_file in tif_files_pre_3:
    get_band(tif_file, pre_stack_3)

for tif_file in tif_files_srad_1:
    get_band(tif_file, srad_stack_1)
for tif_file in tif_files_srad_2:
    get_band(tif_file, srad_stack_2)
for tif_file in tif_files_srad_3:
    get_band(tif_file, srad_stack_3)

for tif_file in pos_tif_files:
    get_band(tif_file, pos_stack)

for tif_file in sos_tif_files:
    get_band(tif_file, sos_stack)

cor_stack_1 = np.stack(cor_stack_1, axis=0)#[:, 505:510, 505:510]
cor_stack_2 = np.stack(cor_stack_2, axis=0)
cor_stack_3 = np.stack(cor_stack_3, axis=0)
cor_stack = np.stack([cor_stack_1, cor_stack_2, cor_stack_3], axis=0)

sm_stack_1 = np.stack(sm_stack_1, axis=0)
sm_stack_2 = np.stack(sm_stack_2, axis=0)
sm_stack_3 = np.stack(sm_stack_3, axis=0)
sm_stack = np.stack([sm_stack_1, sm_stack_2, sm_stack_3], axis=0)
# print('sm_stack shape:\n', sm_stack.shape)
# print('sm_stack:\n', sm_stack[:,90:100,90:100])
vpd_stack_1 = np.stack(vpd_stack_1, axis=0)
vpd_stack_2 = np.stack(vpd_stack_2, axis=0)
vpd_stack_3 = np.stack(vpd_stack_3, axis=0)
vpd_stack = np.stack([vpd_stack_1, vpd_stack_2, vpd_stack_3], axis=0)

ta_stack_1 = np.stack(ta_stack_1, axis=0)
ta_stack_2 = np.stack(ta_stack_2, axis=0)
ta_stack_3 = np.stack(ta_stack_3, axis=0)
ta_stack = np.stack([ta_stack_1, ta_stack_2, ta_stack_3], axis=0)

pre_stack_1 = np.stack(pre_stack_1, axis=0)
pre_stack_2 = np.stack(pre_stack_2, axis=0)
pre_stack_3 = np.stack(pre_stack_3, axis=0)
pre_stack = np.stack([pre_stack_1, pre_stack_2, pre_stack_3], axis=0)

srad_stack_1 = np.stack(srad_stack_1, axis=0)
srad_stack_2 = np.stack(srad_stack_2, axis=0)
srad_stack_3 = np.stack(srad_stack_3, axis=0)
srad_stack = np.stack([srad_stack_1, srad_stack_2, srad_stack_3], axis=0)

pos_stack = np.stack(pos_stack, axis=0)
sos_stack = np.stack(sos_stack, axis=0)
print('pos_stack shape:\n', pos_stack.shape)

print('All stack done!')


######################################### 4 逐像元计算每个变量的最佳季前期 ################################################
cor_preseason_len = np.full((rows, cols), np.nan)
sm_preseason_len = np.full((rows, cols), np.nan)
vpd_preseason_len = np.full((rows, cols), np.nan)
ta_preseason_len = np.full((rows, cols), np.nan)
pre_preseason_len = np.full((rows, cols), np.nan)
srad_preseason_len = np.full((rows, cols), np.nan)


print(f'像元最佳时间窗口计算：')
# 并行处理所有像元（使用所有CPU核心）
# with parallel_backend("threading", n_jobs=15):
results = Parallel(n_jobs=15)(
    delayed(cal_best_perseason_pixel)(
        i, j,
        pos_stack[:, i, j], sos_stack[:, i, j],
        cor_stack[:, :, i, j], sm_stack[:, :, i, j], vpd_stack[:, :, i, j],
        ta_stack[:, :, i, j], pre_stack[:, :, i, j], srad_stack[:, :, i, j]
    )
    for i, j in zip(row_indices, col_indices)
)


for (i, j,
     cor_best_preseason_len, sm_best_preseason_len, vpd_best_preseason_len,
     ta_best_preseason_len, pre_best_preseason_len, srad_best_preseason_len) in results:

    cor_preseason_len[i, j] = cor_best_preseason_len
    sm_preseason_len[i, j] = sm_best_preseason_len
    vpd_preseason_len[i, j] = vpd_best_preseason_len
    ta_preseason_len[i, j] = ta_best_preseason_len
    pre_preseason_len[i, j] = pre_best_preseason_len
    srad_preseason_len[i, j] = srad_best_preseason_len
print(f'像元最佳时间窗口计算结束，开始输出')

####################### 利用最佳季前期求各变量的年均值或年总量 ####################################
output_path0 = os.path.join(ouput_path, f"Cor_preseason_length.tif")
save_tif_gdal(
    output_path0,
    cor_preseason_len,
    crs, gt
)

output_path1 = os.path.join(ouput_path, f"SM_preseason_length.tif")
save_tif_gdal(
    output_path1,
    sm_preseason_len,
    crs, gt
)

output_path2 = os.path.join(ouput_path, f"VPD_preseason_length.tif")
save_tif_gdal(
    output_path2,
    vpd_preseason_len,
    crs, gt
)

output_path3 = os.path.join(ouput_path, f"Ta_preseason_length.tif")
save_tif_gdal(
    output_path3,
    ta_preseason_len,
    crs, gt
)

output_path4 = os.path.join(ouput_path, f"Pre_preseason_length.tif")
save_tif_gdal(
    output_path4,
    pre_preseason_len,
    crs, gt
)

output_path5 = os.path.join(ouput_path, f"Srad_preseason_length.tif")
save_tif_gdal(
    output_path5,
    srad_preseason_len,
    crs, gt
)

print(f"耦合效应、SM、VPD、Ta、Pre、Srad的最佳季前期长度结果已保存")

