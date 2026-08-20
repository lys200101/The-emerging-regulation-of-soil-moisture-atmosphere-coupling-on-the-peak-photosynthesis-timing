
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


tif = fr'D:\NH_permanent_veg_type_fraction_11km.tif'

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


##########################   识别11km植被类型 ###############################################
threshold = 0.25

# 按顺序堆叠
veg_stack = np.stack([
    forest_array,
    shrub_array,
    savanna_array,
    grass_array
], axis=0)      # (4, rows, cols)

# 最大占比
max_fraction = np.max(veg_stack, axis=0)

# 最大值对应的类别
veg_type11km = np.argmax(veg_stack, axis=0) + 1

# 最大占比不足0.25的设为0（Unknown）
veg_type11km[max_fraction < 0.25] = 0

#### 导出tif
output_path = os.path.join(path, 'NH_veg_type_11km(Python).tif')

save_tif_gdal(
    output_path,
    veg_type11km,  # 取第 k 张
    crs,
    gt  # 使用新的地理变换参数
)

print('TIF export done!')

