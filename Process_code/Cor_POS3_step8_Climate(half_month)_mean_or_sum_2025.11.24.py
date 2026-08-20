
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

    # # 验证日期格式
    # if not date_str.isdigit() or len(date_str) != 8:
    #     raise ValueError(f"文件 {filename} 的最后8位不是有效日期（需为YYYYMMDD）！")

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

    data = (tif.GetRasterBand(1).ReadAsArray().astype(np.float32))

    stack.append(data)

    tif = None
    del data




def cal_pixel_timelength_mean(i, j, data):

    time_series = np.array(data[:, i, j])

    time_series_pd = pd.Series(time_series)
    time_series_clean = time_series_pd.dropna().values  # 删除 NaN 后转换为 numpy 数组
    # print('time_series_clean:', time_series_clean)

    if len(time_series_clean) < 1:
        return (i, j, np.nan)
    else:

        result = time_series_clean.mean()


        return (i, j, result)



def extract_time_window(year, sos, pos, dates):
    # """根据像元的sos和pos，提取年份year对应的时间窗口索引"""
    # 计算该像元在年份year的生长季起止日期

    # print('pos：', pos, flush=True)

    ### 只考虑SOS-POS
    # start_date1 = pd.to_datetime(f"{year}-01-01") + pd.Timedelta(days=int(sos))
    start_date1 = pd.to_datetime(f"{year}-01-01") + pd.Timedelta(days=int(pos - interval))
    end_date = pd.to_datetime(f"{year}-01-01") + pd.Timedelta(days=int(pos))

    start_month = start_date1.month

    if start_date1.day >=1 and start_date1.day <= 15 :
        start_date2 = pd.to_datetime(f'{year}-{start_month}-01')
    elif start_date1.day >= 16 :
        start_date2 = pd.to_datetime(f'{year}-{start_month}-16')

    # print(f'原始时间范围：{start_date1} ~ {end_date}\n'
    #       f'用于获取气候数据的时间范围：{start_date2} ~ {end_date}')


    # 找到时间序列中落在[start_date, end_date]内的索引
    # print('start_date:', start_date, 'end_date:', end_date)
    valid_mask = (dates >= start_date2) & (dates <= end_date)
    valid_indices = np.where(valid_mask)[0]
    # print('valid_indices:', valid_indices)
    return valid_indices


def process_pixel(i, j, year, pos, sos, dates,
                   sm_data, vpd_data, ta_data, pre_data, srad_data):
    """处理单个像元的函数（供并行调用）"""
    sos_pixel = sos[i, j]
    pos_pixel = pos[i, j]

    if not pd.isna(pos_pixel) and not pd.isna(sos_pixel):
        valid_indices = extract_time_window(year, sos_pixel, pos_pixel, dates)

        # 提取SM和VPD的时间序列数据（形状：[time, 1, 1] → 展平为[time]）
        sm_series = sm_data[valid_indices, i, j]
        vpd_series = vpd_data[valid_indices, i, j]
        ta_series = ta_data[valid_indices, i, j]
        pre_series = pre_data[valid_indices, i, j]
        srad_series = srad_data[valid_indices, i, j]

        sm_mean_year = np.nanmean(sm_series)
        vpd_mean_year = np.nanmean(vpd_series)
        ta_mean_year = np.nanmean(ta_series)

        pre_sum_year = np.nansum(pre_series)
        srad_sum_year = np.nansum(srad_series)

        return (i, j, sm_mean_year, vpd_mean_year, ta_mean_year, pre_sum_year, srad_sum_year)

    else:
        return (i, j, np.nan, np.nan, np.nan, np.nan, np.nan)


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

test_number = '3'  ###请仔细修改这里!!!!!!
if test_number == '1':
    interval = 30
elif test_number == '2':
    interval = 60
elif test_number == '3':
    interval = 90

scale = 55

input_path = f'I:\Data\ERA5_Land\ERA5_Land_NH_{scale}km_daily_deseason_half_month'

folder1 = fr'{input_path}\ERA5_Land_NH_{scale}km_half_month_0-100cmSM(2001-2024)'  ### 请仔细修改这里   SM:0-100cmSM  Ta
folder2 = fr'{input_path}\ERA5_Land_NH_{scale}km_half_month_VPD(2001-2024)'  ### 请仔细修改这里   VPD           Pre
folder3 = fr'{input_path}\ERA5_Land_NH_{scale}km_half_month_Ta(2001-2024)'
folder4 = fr'{input_path}\ERA5_Land_NH_{scale}km_half_month_Pre(2001-2024)'
folder5 = fr'{input_path}\ERA5_Land_NH_{scale}km_half_month_Srad(2001-2024)'


#### 输入的POS和SOS
pos = 'POS'
sos = 'SOS'

pos_folder = fr'D:\{pos}_{scale}km'
sos_folder = fr'D:\{sos}_{scale}km'
# sos_folder = 'no'

###################### ===================== 输出设定 ======================== ########################
ouput_path = f'D:\Climate_data'

output_sm_tif_path = fr'{ouput_path}\SM_preseason_mean{test_number}'
output_vpd_tif_path = fr'{ouput_path}\VPD_preseason_mean{test_number}'
output_ta_tif_path = fr'{ouput_path}\Ta_preseason_mean{test_number}'
output_pre_tif_path = fr'{ouput_path}\Pre_preseason_sum{test_number}'
output_srad_tif_path = fr'{ouput_path}\Srad_preseason_sum{test_number}'

output_climate_mean_slope_tif_path = fr'D:\Result'

tif_files1 = sorted(glob.glob(os.path.join(folder1, '*.tif')))
tif_files2 = sorted(glob.glob(os.path.join(folder2, '*.tif')))
tif_files3 = sorted(glob.glob(os.path.join(folder3, '*.tif')))
tif_files4 = sorted(glob.glob(os.path.join(folder4, '*.tif')))
tif_files5 = sorted(glob.glob(os.path.join(folder5, '*.tif')))

pos_tif_files = sorted(glob.glob(os.path.join(pos_folder, '*.tif')))
if sos_folder != 'no':
    sos_tif_files = sorted(glob.glob(os.path.join(sos_folder, '*.tif')))

if not tif_files1:
    raise FileNotFoundError("未找到任何 tif_files1 TIF 文件！")
if not tif_files2:
    raise FileNotFoundError("未找到任何 tif_files2 TIF 文件！")
if not tif_files3:
    raise FileNotFoundError("未找到任何 tif_files3 TIF 文件！")
if not tif_files4:
    raise FileNotFoundError("未找到任何 tif_files4 TIF 文件！")
if not tif_files5:
    raise FileNotFoundError("未找到任何 tif_files5 TIF 文件！")
if not pos_tif_files:
    raise FileNotFoundError("未找到任何 POS TIF 文件！")


####################### 2 提取tif信息 ############################
first_tif = tif_files1[0]
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



tif_dates = []
for tif_file in tif_files1:
    date = extract_date_from_filename(tif_file)
    tif_dates.append(date)

print('前五个日期：', [d.strftime('%Y%m%d') for d in tif_dates[:5]])

print('All stack start!')


########################## 数据堆叠 ##########################
sm_stack = []
vpd_stack = []
ta_stack = []
pre_stack = []
srad_stack = []

pos_stack = []
sos_stack = []

for tif_file in tif_files1:
    get_band(tif_file, sm_stack)

for tif_file in tif_files2:
    get_band(tif_file, vpd_stack)

for tif_file in tif_files3:
    get_band(tif_file, ta_stack)

for tif_file in tif_files4:
    get_band(tif_file, pre_stack)

for tif_file in tif_files5:
    get_band(tif_file, srad_stack)


for tif_file in pos_tif_files:
    get_band(tif_file, pos_stack)

if sos_folder != 'no':
    for tif_file in sos_tif_files:
        get_band(tif_file, sos_stack)

sm_stack = np.stack(sm_stack, axis=0)#[:, 505:510, 505:510]
vpd_stack = np.stack(vpd_stack, axis=0)#[:, 505:510, 505:510]
ta_stack = np.stack(ta_stack, axis=0)
pre_stack = np.stack(pre_stack, axis=0)
srad_stack = np.stack(srad_stack, axis=0)

pos_stack = np.stack(pos_stack, axis=0)#[:, 505:510, 505:510]
if sos_folder != 'no':
    sos_stack = np.stack(sos_stack, axis=0)#[:, 505:510, 505:510]
print('pos_stack shape:\n', pos_stack.shape)

print('All stack done!')


######################################### 4 耦合效应、mean计算 ################################################

dates = pd.to_datetime(tif_dates)
print('前五个日期：', [d.strftime('%Y%m%d') for d in dates[:5]])


sm_mean = np.full((years_length, rows, cols), np.nan)
vpd_mean = np.full((years_length, rows, cols), np.nan)
ta_mean = np.full((years_length, rows, cols), np.nan)
pre_sum = np.full((years_length, rows, cols), np.nan)
srad_sum = np.full((years_length, rows, cols), np.nan)


for year in years:
    print(f"正在处理年份：{year}")

    k = year - 2001

    year_mask = (dates >= f"{year}-01-01") & (dates <= f"{year}-12-31")

    ## 用顺序索引
    year_idx = np.where(year_mask)[0]
    print('year_idx:\n', year_idx)
    # 这一年的日期与数据切片（注意：这里不涉及 reset_index）
    year_dates = dates[year_idx]
    sm_stack_year = sm_stack[year_idx, :, :]
    vpd_stack_year = vpd_stack[year_idx, :, :]
    ta_stack_year = ta_stack[year_idx, :, :]
    pre_stack_year = pre_stack[year_idx, :, :]
    srad_stack_year = srad_stack[year_idx, :, :]


    pos_year = pos_stack[k, :, :]
    sos_year = sos_stack[k, :, :]

    ############ 并行处理

    print(f'year={year}pixel Pearson cor and mean calculate start')
    # 并行处理所有像元（使用所有CPU核心）
    with parallel_backend("threading", n_jobs=15):
        results = Parallel(verbose=10)(
            delayed(process_pixel)(
                i, j, year, pos_year, sos_year, year_dates,
                sm_stack_year, vpd_stack_year, ta_stack_year, pre_stack_year, srad_stack_year
            )
            for i, j in zip(row_indices, col_indices)
        )


    for i, j, sm_mean_year, vpd_mean_year, ta_mean_year, pre_sum_year, srad_sum_year in results:
        sm_mean[k, i, j] = sm_mean_year
        vpd_mean[k, i, j] = vpd_mean_year
        ta_mean[k, i, j] = ta_mean_year
        pre_sum[k, i, j] = pre_sum_year
        srad_sum[k, i, j] = srad_sum_year


    output_path1 = os.path.join(output_sm_tif_path, f"SM_pearson_mean_{year}.tif")
    save_tif_gdal(
        output_path1,
        sm_mean[k, :, :],
        crs, gt
    )

    output_path2 = os.path.join(output_vpd_tif_path, f"VPD_pearson_mean_{year}.tif")
    save_tif_gdal(
        output_path2,
        vpd_mean[k, :, :],
        crs, gt
    )

    output_path3 = os.path.join(output_ta_tif_path, f"Ta_pearson_mean_{year}.tif")
    save_tif_gdal(
        output_path3,
        ta_mean[k, :, :],
        crs, gt
    )

    output_path4 = os.path.join(output_pre_tif_path, f"Pre_pearson_sum_{year}.tif")
    save_tif_gdal(
        output_path4,
        pre_sum[k, :, :],
        crs, gt
    )

    output_path5 = os.path.join(output_srad_tif_path, f"Srad_pearson_sum_{year}.tif")
    save_tif_gdal(
        output_path5,
        srad_sum[k, :, :],
        crs, gt
    )

    print(f"{year}年结果已保存")




