import os
import glob
import datetime
import sys

import pandas as pd
from osgeo import gdal
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from joblib import Parallel, delayed


######### Function ##########

### 读取时间信息，提取每个 TIF 文件的日期（最后8位）
def extract_date_from_filename(filename):
    # 提取纯文件名（不含路径和扩展名）
    basename = os.path.basename(filename)
    filename_without_ext = os.path.splitext(basename)[0]

    # 截取最后8位
    date_str = filename_without_ext[-8:]

    # 验证日期格式
    if not date_str.isdigit() or len(date_str) != 8:
        raise ValueError(f"文件 {filename} 的最后8位不是有效日期（需为YYYYMMDD）！")

    # 转换为 datetime 对象
    return datetime.datetime.strptime(date_str, "%Y%m%d")



### 读取SM和VPD波段
def get_band(tif_file, stack):
    tif = gdal.Open(tif_file)
    climate_data = tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
    stack.append(climate_data)
    tif = None  # 及时释放资源


### 计算半月气候态
def calculate_half_month_mean_or_sum(pixel_time_series, dates, var_cal_method, i, j):
    """
    计算每年每半月的均值或总值

    返回: (i, j, yearly_half_month_values)
    yearly_half_month_values: {(year, month, half): value}
    """
    # 确保dates是datetime格式
    dates = pd.Series(pd.to_datetime(dates))

    # 获取年份范围
    years = np.unique(dates.dt.year)

    # 存储每年每半月的值
    yearly_half_month_value = {}

    for year in years:
        # 筛选该年份的数据
        year_mask = dates.dt.year == year

        for month in range(1, 13):
            # 筛选该月份的数据
            month_mask = dates.dt.month == month
            full_mask = year_mask & month_mask

            if not np.any(full_mask):
                # 如果没有该月数据，设为NaN
                yearly_half_month_value[(year, month, 1)] = np.nan
                yearly_half_month_value[(year, month, 2)] = np.nan
                continue

            # 获取该月所有数据
            month_dates = dates[full_mask]
            month_values = pixel_time_series[full_mask]

            # 计算前半月（1-15日）
            early_mask = month_dates.dt.day <= 15
            early_values = month_values[early_mask]

            # 计算后半月（16日-月底）
            late_mask = month_dates.dt.day > 15
            late_values = month_values[late_mask]

            # 计算均值或总值
            if var_cal_method == 'mean':
                month_early_value = np.nanmean(early_values) if len(early_values) > 0 else np.nan
                month_late_value = np.nanmean(late_values) if len(late_values) > 0 else np.nan
            elif var_cal_method == 'sum':
                month_early_value = np.nansum(early_values) if len(early_values) > 0 else np.nan
                month_late_value = np.nansum(late_values) if len(late_values) > 0 else np.nan

            # 存储结果
            yearly_half_month_value[(year, month, 1)] = month_early_value  # 前半月
            yearly_half_month_value[(year, month, 2)] = month_late_value  # 后半月

        if i == 50 and j==500:
            print(f'yearly_half_month_values:{yearly_half_month_value[(year, 1, 1)]}')

    return i, j, yearly_half_month_value


def pixel_halfmonth_anomaly(
        pixel_time_series,
        dates,
        var_cal_method,
        half_month_keys,
        i, j
):
    """
    对单个像元计算半月尺度 anomaly（含气候态）

    Parameters
    ----------
    pixel_time_series : 1D array (time,)
        单像元日尺度时间序列
    dates : list or array-like of datetime
        与 pixel_time_series 对应的日期
    var_cal_method : str
        'mean' or 'sum'
    half_month_keys : list of (year, month, half)
        时间轴顺序
    i, j : int
        像元行列号（用于回填）

    Returns
    -------
    i, j, anomalies : (int, int, 1D array)
        anomalies 顺序与 half_month_keys 完全一致
    """

    dates = pd.to_datetime(dates)
    years = np.unique(dates.year)

    # ---------- 1. 逐年逐半月计算值 ----------
    yearly_values = {}  # {(year, month, half): value}

    for year in years:
        year_mask = dates.year == year

        for month in range(1, 13):
            month_mask = dates.month == month
            full_mask = year_mask & month_mask

            if not np.any(full_mask):
                yearly_values[(year, month, 1)] = np.nan
                yearly_values[(year, month, 2)] = np.nan
                continue

            month_dates = dates[full_mask]
            month_values = pixel_time_series[full_mask]

            # 前半月
            early_mask = month_dates.day <= 15
            early_vals = month_values[early_mask]

            # 后半月
            late_mask = month_dates.day > 15
            late_vals = month_values[late_mask]

            if var_cal_method == 'mean':
                v1 = np.nanmean(early_vals) if early_vals.size > 0 else np.nan
                v2 = np.nanmean(late_vals) if late_vals.size > 0 else np.nan
            else:  # 'sum'
                v1 = np.nansum(early_vals) if early_vals.size > 0 else np.nan
                v2 = np.nansum(late_vals) if late_vals.size > 0 else np.nan

            yearly_values[(year, month, 1)] = v1
            yearly_values[(year, month, 2)] = v2

    # ---------- 2. 计算半月气候态 ----------
    clim_sum = {}
    clim_cnt = {}

    for (year, month, half), v in yearly_values.items():
        if np.isnan(v):
            continue

        key = (month, half)
        clim_sum[key] = clim_sum.get(key, 0.0) + v
        clim_cnt[key] = clim_cnt.get(key, 0) + 1

    climatology = {
        k: clim_sum[k] / clim_cnt[k]
        for k in clim_sum
    }

    # ---------- 3. 计算 anomaly（按时间顺序） ----------
    anomalies = np.full(len(half_month_keys), np.nan, dtype=np.float32)

    for t, key in enumerate(half_month_keys):
        if key not in yearly_values:
            continue

        v = yearly_values[key]
        clim_key = (key[1], key[2])  # (month, half)

        if np.isnan(v) or clim_key not in climatology:
            anomalies[t] = np.nan
        else:
            anomalies[t] = v - climatology[clim_key]

    return i, j, anomalies


### 去趋势
def detrend_with_lowess_matrix(time_series, i, j, frac):
    if np.all(~np.isfinite(time_series)):
        detrended_full = np.full_like(time_series, np.nan)
    else:
        # 检查有效数据点
        valid_mask = ~np.isnan(time_series)
        valid_data = time_series[valid_mask]

        # 生成时间索引
        y = np.arange(len(valid_data))
        # LOWESS平滑
        smoothed = lowess(valid_data, y, frac=frac, return_sorted=False)
        # 去趋势
        detrended_valid = valid_data - smoothed

        # 重建完整序列
        detrended_full = np.full_like(time_series, np.nan)
        detrended_full[valid_mask] = detrended_valid

        if i == 50 and j == 500:
            print(f'valid_data:{valid_data}')
            print(f'smoothed:{smoothed}')
            print(f'detrended_valid:{detrended_valid}')

    return detrended_full


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
    output_ds.SetGeoTransform(transform)
    output_ds = None
    return True


###################################### 1 数据读取及输出设定 ################################################
######  ============== 需修改的输入设定 ============= #########

startYear = 2001
endYear = 2024
pixel_resolution = 55
climate_var = 'Srad'  ##0-100cmSM / VPD / Ta / Pre / Srad
path = r'I:/Data/ERA5_Land'

folder = f'{path}/ERA5_Land_NH_{pixel_resolution}km_daily/ERA5_Land_NH_{pixel_resolution}km_daily_{climate_var}'

#########################

years_length = endYear - startYear + 1
years = range(startYear, endYear + 1)

tif_files = sorted(glob.glob(os.path.join(folder, '*.tif')))
if not tif_files:
    raise FileNotFoundError("未找到任何 TIF 文件！")

######## 输出设定
output_detrend_tif_path = f'{path}/ERA5_Land_NH_{pixel_resolution}km_daily_deseason_half_month/ERA5_Land_NH_{pixel_resolution}km_half_month_{climate_var}(2001-2024)'

########################### 2 基本信息 ########################
first_tif = tif_files[0]
sample_tif = gdal.Open(first_tif)

if sample_tif is None:
    raise RuntimeError(f"无法打开 TIF 文件：{first_tif}（驱动不支持或文件损坏）")

# 获取地理变换参数
crs = sample_tif.GetProjectionRef()
gt = sample_tif.GetGeoTransform()

# 获取数据尺寸
sample_array = sample_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
rows = sample_array.shape[0]
cols = sample_array.shape[1]
print('原始: rows=', rows, 'cols=', cols)

# 创建行、列索引
row_indices = np.repeat(np.arange(rows), cols)
col_indices = np.tile(np.arange(cols), rows)

# 释放样本文件
sample_tif = None

############################################ 3 时间-堆叠 ###################################################
## 时间
tif_dates = []

for tif_file in tif_files:
    date = extract_date_from_filename(tif_file)
    tif_dates.append(date)

print('前五个日期：', [d.strftime('%Y%m%d') for d in tif_dates[:5]])
print('后五个日期：', [d.strftime('%Y%m%d') for d in tif_dates[-5:]])

## 数据堆叠
data_stack = []

for tif_file in tif_files:
    get_band(tif_file, data_stack)

print('Stack start!')
data_stack = np.stack(data_stack, axis=0)
print('data_stack shape:', data_stack.shape)
print('Stack end!')

########################################### 5 去季节性（半月尺度）###################################################
print(f'{climate_var} deseasonal start')

# 5.0 聚合方式
if climate_var in ('Ta', '0-100cmSM', 'VPD'):
    var_cal_method = 'mean'
elif climate_var in ('Pre', 'Srad'):
    var_cal_method = 'sum'

# 5.1 半月时间轴
half_month_keys = [
    (year, month, half)
    for year in years
    for month in range(1, 13)
    for half in (1, 2)
]

half_time_length = len(half_month_keys)



half_month_output = np.full(
    (half_time_length, rows, cols),
    np.nan,
    dtype=np.float32
)

results = Parallel(n_jobs=15)(
    delayed(pixel_halfmonth_anomaly)(
        data_stack[:, i, j],
        tif_dates,
        var_cal_method,
        half_month_keys,
        i, j
    )
    for i, j in zip(row_indices, col_indices)
)

for i, j, anomalies in results:
    half_month_output[:, i, j] = anomalies

print(f'{climate_var} deseasonal end')


########################################### 6 去趋势 ###################################################
print(f'{climate_var} detrend start')
# 使用LOWESS去趋势
frac_value = 0.4  # LOWESS平滑参数，可根据需要调整

detrended_data = Parallel(n_jobs=15)(
    delayed(detrend_with_lowess_matrix)(
        half_month_output[:, i, j], i, j,
        frac=frac_value
    )
    for i, j in zip(row_indices, col_indices)
)

# 重组为三维数组
detrended_stack = np.array(detrended_data).reshape(rows, cols, half_time_length).transpose(2, 0, 1)
print(f'{climate_var} detrend end')
print(f'detrended_stack shape: {detrended_stack.shape}')

########################################### 7 保存结果 ###################################################

# 保存每个时间步的结果
print("保存结果TIFF文件...")
for t in range(half_time_length):
    year = half_month_keys[t][0]
    month = half_month_keys[t][1]
    half = half_month_keys[t][2]

    output_filename = f"{climate_var}_deseason_{year}{month:02d}_{half}.tif"
    output_path = os.path.join(output_detrend_tif_path, output_filename)

    save_tif_gdal(output_path, detrended_stack[t, :, :], crs, gt)

print("处理完成！")
