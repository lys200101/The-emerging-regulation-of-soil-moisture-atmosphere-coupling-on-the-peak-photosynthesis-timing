
import os
import glob
import datetime
import pandas as pd
from osgeo import gdal
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from joblib import Parallel, delayed, parallel_backend
from matplotlib import colormaps



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



### 计算日气候态（多年日平均）
def calculate_pixel_daily_climatology(pixel_time_series, dates):
    if not hasattr(dates, 'month'):
        dates = pd.Series(dates)

    daily_climatology = {}

    for month in range(1, 13):
        for day in range(1, 32):
            try:
                # 检查日期是否有效
                datetime.datetime(2001, month, day)

                # 筛选该月日的所有数据
                mask = (dates.dt.month == month) & (dates.dt.day == day)
                if np.sum(mask) > 0:
                    # 提取该像元在对应日期的值
                    daily_values = pixel_time_series[mask]
                    # 计算该日的多年均值（忽略NaN）
                    if np.sum(~np.isnan(daily_values)) > 0:
                        climatology = np.nanmean(daily_values)
                        # print(f'month:{month}, day:{day}, daily_values:{pixel_time_series_value}, daily_climatology[key]:{daily_climatology_value}')
                        daily_climatology[(month, day)] = climatology

            except ValueError:
                continue

    return daily_climatology



### 去季节性:减去该日多年均值
def deseasonalize_pixel(pixel_time_series, dates):

    if np.all(~np.isfinite(pixel_time_series)):

        # 创建去季节性后的序列
        deseasonalized_series = np.full_like(pixel_time_series, np.nan)

        return deseasonalized_series

    else:
        # # 计算该像元的日气候态
        # daily_climatology = calculate_pixel_daily_climatology(pixel_time_series, dates)

        # 创建去季节性后的序列
        deseasonalized_series = np.full_like(pixel_time_series, np.nan)

        for i, date in enumerate(dates):
            month = date.month
            day = date.day
            key = (month, day)

            if key in daily_climatology and not np.isnan(pixel_time_series[i]):
                # pixel_time_series_value = pixel_time_series[i]
                # daily_climatology_value = daily_climatology[key]
                # print(f'month:{month}, day:{day}, pixel_time_series[i]:{pixel_time_series_value}, daily_climatology[key]:{daily_climatology_value}')
                # 减去该日的多年平均值
                deseasonalized_series[i] = pixel_time_series[i] - daily_climatology[key]
            else:
                # 如果该日没有气候态数据或原始数据为NaN，保持NaN
                deseasonalized_series[i] = np.nan

    return deseasonalized_series



def deseasonal_parelle(data, dates, n_jobs):
    time_steps, rows, cols = data.shape

    row_indices = np.repeat(np.arange(rows), cols)
    col_indices = np.tile(np.arange(cols), rows)

    deseason_data = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(deseasonalize_pixel)(
            data[:, i, j],
            dates
        )
        for i, j in zip(row_indices, col_indices))

    # 将一维结果重组为三维影像（时间步, 行, 列）
    deseason_data = np.array(deseason_data).reshape(rows, cols, time_steps).transpose(2, 0, 1)

    return deseason_data


### 去趋势
def detrend_with_lowess_matrix(time_series, frac):

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

    return detrended_full


def detrend_stack_parallel(stack, n_jobs, frac=0.4):
    # 获取数据维度（时间步, 行, 列）
    time_steps, rows, cols = stack.shape

    # 生成所有像元的行、列索引（i: 行, j: 列）
    row_indices = np.repeat(np.arange(rows), cols)  # 行索引重复cols次
    col_indices = np.tile(np.arange(cols), rows)  # 列索引平铺rows次

    # print('detrend start2')
    # 并行处理每个像元（提取时间序列 → 去趋势 → 返回结果）
    # with parallel_backend('threading', n_jobs=n_jobs):
    detrended_pixels = Parallel(n_jobs=n_jobs, verbose = 10)(
        delayed(detrend_with_lowess_matrix)(
            stack[:, i, j],  # 提取第(i,j)像元的时间序列（形状：时间步）
            frac=frac
        )
        for i, j in zip(row_indices, col_indices)  # 遍历所有行、列组合
    )

    # 将一维结果重组为三维影像（时间步, 行, 列）
    detrended_stack = np.array(detrended_pixels).reshape(rows, cols, time_steps).transpose(2, 0, 1)

    return detrended_stack



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

print(list(colormaps))



######  ============== 需修改的输入设定 ============= #########

pixel_resolution = 55

climate_var = 'VPD'  ##0-100cmSM / VPD

path = 'I:\Data\ERA5_Land'

folder = f'{path}\ERA5_Land_NH_{pixel_resolution}km_daily\ERA5_Land_NH_{pixel_resolution}km_daily_{climate_var}'


#########################
tif_files = sorted(glob.glob(os.path.join(folder, '*.tif')))
if not tif_files:
    raise FileNotFoundError("未找到任何 TIF 文件！")


######## 输出设定
output_detrend_tif_path = f'{path}\ERA5_Land_NH_{pixel_resolution}km_daily_deseason_detrend\ERA5_Land_NH_{pixel_resolution}km_daily_{climate_var}(2001-2024)'


########################### 2 基本信息 ########################
first_tif = tif_files[0]
sample_tif = gdal.Open(first_tif)

if sample_tif is None:
    raise RuntimeError(f"无法打开 TIF 文件：{sample_tif}（驱动不支持或文件损坏）")

# 获取地理变换参数
crs = sample_tif.GetProjectionRef()
gt = sample_tif.GetGeoTransform()
proj = sample_tif.GetProjection()

# 获取数据尺寸
sample_array = sample_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
rows = sample_array.shape[0]
cols = sample_array.shape[1]
print('原始: rows=', rows, 'cols=', cols)


# 释放样本文件
sample_tif = None


############################################ 3 时间-堆叠 ###################################################
## 时间
tif_dates = []
for tif_file in tif_files:
    date = extract_date_from_filename(tif_file)
    tif_dates.append(date)
dates_pd = pd.Series(tif_dates)  # 转换为pandas Series以便使用dt属性

print('前五个日期：', [d.strftime('%Y%m%d') for d in tif_dates[:5]])
print('后五个日期：', [d.strftime('%Y%m%d') for d in tif_dates[-5:]])

## 数据堆叠
data_stack = []

for tif_file in tif_files:
    get_band(tif_file, data_stack)

print('Stack start!')
data_stack = np.stack(data_stack, axis=0)
# 释放文件
tif_files = None
print('data_stack shape:\n', data_stack.shape)
print('Stack end!')

print(f'{climate_var} deseasonal start')

# 计算该像元的日气候态
daily_climatology = calculate_pixel_daily_climatology(data_stack, dates_pd)


############################ 4 去季节去趋势 ######################################
import gc

mid_row1 = rows // 3
mid_row2 = 2 * rows // 3

mid_col1 = cols // 3
mid_col2 = 2 * cols // 3

parts = [
    # 第一行
    (slice(None), slice(0, mid_row1),      slice(0, mid_col1)),      # 左上
    (slice(None), slice(0, mid_row1),      slice(mid_col1, mid_col2)), # 中上
    (slice(None), slice(0, mid_row1),      slice(mid_col2, None)),      # 右上

    # 第二行
    (slice(None), slice(mid_row1, mid_row2), slice(0, mid_col1)),      # 左中
    (slice(None), slice(mid_row1, mid_row2), slice(mid_col1, mid_col2)), # 中中
    (slice(None), slice(mid_row1, mid_row2), slice(mid_col2, None)),      # 右中

    # 第三行
    (slice(None), slice(mid_row2, None), slice(0, mid_col1)),      # 左下
    (slice(None), slice(mid_row2, None), slice(mid_col1, mid_col2)), # 中下
    (slice(None), slice(mid_row2, None), slice(mid_col2, None)),      # 右下
]


frac = 0.4

data_detrend = np.empty_like(data_stack, dtype=np.float32)

print(f'{climate_var} deseasonal + detrend start')

for i, idx in enumerate(parts):

    print(f'Processing block {i+1}/6')
    print(f'Deseasonal start')

    # ========================
    # 1. 去季节化
    # ========================
    part = deseasonal_parelle(
        data_stack[idx],
        dates_pd,
        n_jobs=18
    )

    # ========================
    # 2. 去趋势
    # ========================
    print(f'Detrend start')
    part = detrend_stack_parallel(
        part,
        n_jobs=18,
        frac=frac
    )

    # 直接写回最终数组
    data_detrend[idx] = part

    print(f'Block {i+1} finished')

    del part
    gc.collect()

print(f'{climate_var} deseasonal + detrend finished')



############################################ 5 导出 ######################################################
### Export data after detrend with date
for k, date in enumerate(tif_dates):
    date_str = date.strftime("%Y%m%d")  # 转成类似 20200101 的格式
    output_path = os.path.join(
        output_detrend_tif_path,
        f'{climate_var}_deseason_{date_str}.tif'
    )

    save_tif_gdal(
        output_path,
        data_detrend[k, :, :],  # 取第 k 张
        crs,
        new_gt  # 使用新的地理变换参数
    )
print(f'{climate_var} export done!')
