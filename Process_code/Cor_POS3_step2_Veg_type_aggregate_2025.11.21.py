
import os
import glob
import datetime
import sys
import pandas as pd
from osgeo import gdal
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from joblib import Parallel, delayed, parallel_backend
from matplotlib import colormaps


######### Function ##########
def aggregate_function(data, aggregation_factor, rows, cols, new_rows, new_cols):

    aggregate_data = np.full((new_rows, new_cols), np.nan, dtype=np.float32)

    row_indices = np.repeat(np.arange(new_rows), new_cols)
    col_indices = np.tile(np.arange(new_cols), new_rows)

    def aggregate_paralle(data, i, j):
        row_start = i * aggregation_factor
        row_end = min((i + 1) * aggregation_factor, rows)
        col_start = j * aggregation_factor
        col_end = min((j + 1) * aggregation_factor, cols)

        window = data[row_start:row_end, col_start:col_end]


        if np.all(np.isnan(window)):
            result = np.nan
        else:
            window_clean = np.where(np.isnan(window), 0, window)
            result = np.mean(window_clean)


        return i, j, result

    results = Parallel(n_jobs=18)(
        delayed(aggregate_paralle)(
            data, i, j
        )
        for i, j in zip(row_indices, col_indices)
    )

    for i, j, result in results:
        aggregate_data[i, j] = result

    return aggregate_data

def clip_by_latitude(gt, rows, lat_min, lat_max):
    """
    根据纬度范围裁剪数据
    返回: (row_start, row_end, new_gt)
    """
    # 计算每行的中心纬度
    row_centers = np.arange(rows) * gt[5] + gt[3] + gt[5] / 2

    # 找到纬度在30-90度范围内的行
    valid_rows = (row_centers >= lat_min) & (row_centers <= lat_max)

    if not np.any(valid_rows):
        raise ValueError(f"在纬度范围 {lat_min}-{lat_max} 内没有找到有效数据")

    # 找到第一个和最后一个有效行
    valid_row_indices = np.where(valid_rows)[0]
    row_start = valid_row_indices[0]
    row_end = valid_row_indices[-1] + 1  # 切片是左闭右开

    # 计算新的左上角坐标
    new_top_left_x = gt[0] + row_start * gt[2]  # 通常gt[2]=0
    new_top_left_y = gt[3] + row_start * gt[5]  # gt[5]是像素高度

    # 创建新的地理变换参数
    new_gt = (
        new_top_left_x,
        gt[1],  # 像素宽度不变
        gt[2],  # 行旋转不变
        new_top_left_y,
        gt[4],  # 列旋转不变
        gt[5]  # 像素高度不变
    )

    # print(f"纬度范围: {lat_min}-{lat_max}°N")
    # print(f"对应的行范围: {row_start} - {row_end - 1}")
    # print(f"原始行数: {rows}, 裁剪后行数: {row_end - row_start}")
    # print(f"原始左上角纬度: {gt[3]:.2f}°N, 新左上角纬度: {new_top_left_y:.2f}°N")

    return row_start, row_end, new_gt


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

######  ============== 需修改的输入/输出设定 ============= #########

aggregation_factor = 5  ### magnification  1 / 5
aggregate_size = 11*aggregation_factor

min_lat_value = 30
max_lat_value = 84

tif = fr'D:\NH_permanent_veg_type_fraction_11km.tif'

fig_output = r'D:\Veg_type'

########################### 2 基本信息 ########################
sample_tif = gdal.Open(tif)

# 获取地理变换参数
crs = sample_tif.GetProjectionRef()
gt = sample_tif.GetGeoTransform()
proj = sample_tif.GetProjection()

# 获取数据尺寸
forest_array = sample_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
rows = forest_array.shape[0]
cols = forest_array.shape[1]
print('rows=', rows, 'cols=', cols)

shrub_array = sample_tif.GetRasterBand(2).ReadAsArray().astype(np.float32)
savanna_array = sample_tif.GetRasterBand(3).ReadAsArray().astype(np.float32)
grass_array = sample_tif.GetRasterBand(4).ReadAsArray().astype(np.float32)
wet_array = sample_tif.GetRasterBand(5).ReadAsArray().astype(np.float32)

# 释放样本文件
sample_tif = None

############################################ 3 聚合 ###################################################
### Aggregate parmeter change
new_rows = rows // aggregation_factor
new_cols = cols // aggregation_factor

# 更新地理变换参数
new_gt = (
    gt[0],  # 左上角x坐标
    gt[1] * aggregation_factor,  # 像素宽度
    gt[2],
    gt[3],  # 左上角y坐标
    gt[4],
    gt[5] * aggregation_factor  # 像素高度
)

aggregated_forest_data = aggregate_function(forest_array, aggregation_factor, rows, cols, new_rows, new_cols)
aggregated_shrub_data = aggregate_function(shrub_array, aggregation_factor, rows, cols, new_rows, new_cols)
aggregated_savanna_data = aggregate_function(savanna_array, aggregation_factor, rows, cols, new_rows, new_cols)
aggregated_grass_data = aggregate_function(grass_array, aggregation_factor, rows, cols, new_rows, new_cols)
aggregated_wet_data = aggregate_function(wet_array, aggregation_factor, rows, cols, new_rows, new_cols)

print(f"Aggregate done!")

# 获取聚合后的样本数据用于后续处理
rows_agg, cols_agg = aggregated_forest_data.shape[0], aggregated_forest_data.shape[1]
print(f'聚合后: rows=', rows_agg, 'cols=', cols_agg)

############################################ 4 判断55km的植被类型 & 哪些11km可以保留 #########################################
def determine_veg_type(args):
    """并行处理植被类型判断"""
    i, j = args
    forest_frac = aggregated_forest_data[i, j]
    shrub_frac = aggregated_shrub_data[i, j]
    savanna_frac = aggregated_savanna_data[i, j]
    grass_frac = aggregated_grass_data[i, j]
    wet_frac = aggregated_wet_data[i, j]

    # 处理无效数据
    if (np.isnan(forest_frac) or np.isnan(shrub_frac) or
            np.isnan(savanna_frac) or np.isnan(grass_frac) or np.isnan(wet_frac)):
        return i, j, np.nan

    total_frac = forest_frac + shrub_frac + savanna_frac + grass_frac + wet_frac

    # 处理除零情况
    if total_frac <= 0:
        return i, j, np.nan

    # 计算比例
    forest_ratio = forest_frac / total_frac
    shrub_ratio = shrub_frac / total_frac
    savanna_ratio = savanna_frac / total_frac
    grass_ratio = grass_frac / total_frac
    wet_ratio = wet_frac / total_frac

    # 找到主导植被类型
    veg_ratios = np.array([forest_ratio, shrub_ratio, savanna_ratio, grass_ratio, wet_ratio])
    max_index = np.argmax(veg_ratios)
    max_value = veg_ratios[max_index]

    # 判断条件：比例大于阈值且是主导类型
    threshold = 0.25

    if forest_frac > threshold and max_index == 0:
        return i, j, 1
    elif shrub_frac > threshold and max_index == 1:
        return i, j, 2
    elif savanna_frac > threshold and max_index == 2:
        return i, j, 3
    elif grass_frac > threshold and max_index == 3:
        return i, j, 4
    elif wet_frac > threshold and max_index == 4:
        return i, j, 5
    else:
        return i, j, np.nan


# 使用并行处理
veg_type_55km = np.full((rows_agg, cols_agg), np.nan)

# 生成所有坐标
coordinates = [(i, j) for i in range(rows_agg) for j in range(cols_agg)]

# 并行处理
results = Parallel(n_jobs=18)(
    delayed(determine_veg_type)(coord) for coord in coordinates
)

# 填充结果
for i, j, result in results:
    veg_type_55km[i, j] = result

row_start, row_end, new_gt = clip_by_latitude(new_gt, rows_agg, lat_min = min_lat_value, lat_max = max_lat_value)
print(f'row_start:{row_start}, row_end:{row_end}')

# 计算裁剪后的尺寸
lon_min = new_gt[0]               ##起始经度
lon_max = new_gt[0] + new_gt[1]*cols_agg  ##起始经度+像元分辨率*列数
lat_max = new_gt[3]       ##起始纬度
lat_min = new_gt[3] + new_gt[5]* (row_end - row_start)  ##起始纬度+像元分辨率*行数数
print(f'经度范围：{lon_min}~{lon_max}°')
print(f'纬度范围：{lat_min}~{lat_max}°N')

veg_type_55km = veg_type_55km[row_start:row_end, :]

### 统计各种森林类型占比
total_count = np.count_nonzero(np.isfinite(veg_type_55km))
print(f'total_count: {total_count}')

forest = veg_type_55km[veg_type_55km== 1]
shrub = veg_type_55km[veg_type_55km== 2]
savanna = veg_type_55km[veg_type_55km== 3]
grass = veg_type_55km[veg_type_55km== 4]
wet = veg_type_55km[veg_type_55km== 5]

forest_count = np.count_nonzero(np.isfinite(forest))
shrub_count = np.count_nonzero(np.isfinite(shrub))
savanna_count = np.count_nonzero(np.isfinite(savanna))
grass_count = np.count_nonzero(np.isfinite(grass))
wet_count = np.count_nonzero(np.isfinite(wet))

forest_ratio = forest_count/total_count * 100
shrub_ratio = shrub_count/total_count * 100
savanna_ratio = savanna_count/total_count * 100
grass_ratio = grass_count/total_count * 100
wet_ratio = wet_count/total_count * 100


############################################ 5 导出 ######################################################
#### 导出tif
output_path = os.path.join(path, rf'NH_veg_type_{aggregate_size}km(Python).tif')

save_tif_gdal(
    output_path,
    veg_type_55km,  # 取第 k 张
    crs,
    new_gt  # 使用新的地理变换参数
)

print('TIF export done!')