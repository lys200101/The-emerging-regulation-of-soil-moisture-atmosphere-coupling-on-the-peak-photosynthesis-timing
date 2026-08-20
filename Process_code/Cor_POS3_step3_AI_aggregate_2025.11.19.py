
import os
import sys
from osgeo import gdal
import numpy as np
from joblib import Parallel, delayed


######### Function ##########
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
    new_top_left_x = gt[0]
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

    new_rows = row_end - row_start

    # print(f"纬度范围: {lat_min}-{lat_max}°N")
    # print(f"对应的行范围: {row_start} - {row_end - 1}")
    # print(f"原始行数: {rows}, 裁剪后行数: {row_end - row_start}")
    # print(f"原始左上角纬度: {gt[3]:.2f}°N, 新左上角纬度: {new_top_left_y:.2f}°N")

    return row_start, row_end, new_gt, new_rows


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
min_lat_value = 30
max_lat_value = 84

scale = 55

## Aggregate climate data
tif = fr'D:\ai_v3_yr.tif'

## 标准的pheno data
pheno_tif = fr'D:\FigShare_data\55km\POS_55km\POS1_aggMean_11000m_2001.tif'

####### 输出 #########
tif_output = 'D:\AI'



########################### 2 基本信息 ########################
#####模板tif信息##########
template_data = gdal.Open(pheno_tif)

# 获取地理变换参数
crs_template = template_data.GetProjectionRef()
gt_template = template_data.GetGeoTransform()
proj_template = template_data.GetProjection()

# 获取数据尺寸
template_array = template_data.GetRasterBand(1).ReadAsArray().astype(np.float32)
template_rows, template_cols = template_array.shape
print(f"template_rows= {template_rows}   template_cols= {template_cols}")


####### 数据源信息 #########
ai_tif = gdal.Open(tif)

# 获取地理变换参数
ai_crs = ai_tif.GetProjectionRef()
ai_gt = ai_tif.GetGeoTransform()
ai_proj = ai_tif.GetProjection()

# 获取数据尺寸
ai_array = ai_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
ai_rows, ai_cols = ai_array.shape
print('ai_rows=', ai_rows, 'ai_cols=', ai_cols)


# 释放样本文件
sample_tif = None


############################################ 3 聚合 ###################################################
### Aggregate parmeter change
aggregation_factor = int(round(gt_template[1] / ai_gt[1]))  ##x与y一样


# 计算在源数据中对应模板纬度范围的起始行
ai_start_row = int(round((gt_template[3] - ai_gt[3]) / ai_gt[5])) ##模板数据最大纬度与kb数据最大纬度的差异的行数
ai_end_row = int(round(((gt_template[3] + gt_template[5] * template_rows) - ai_gt[3]) / ai_gt[5])) ##模板数据最小纬度与kb数据最大纬度的差异的行数


row_indices = np.repeat(np.arange(template_rows), template_cols)
col_indices = np.tile(np.arange(template_cols), template_rows)


aggregated_ai_data = np.full((template_rows, template_cols), np.nan)


def ai_climate_region_aggregate(data, i, j):

    if np.count_nonzero(np.isfinite(data)) == 0:
        return np.nan, i, j
    else:
        ai_agg_value = np.nanmean(data)
        return ai_agg_value, i, j


ai_array = np.where((ai_array != 0), ai_array, np.nan)

results = Parallel(n_jobs=18, verbose=10)(
    delayed(ai_climate_region_aggregate)(
        ai_array[ai_start_row + i*aggregation_factor:min(ai_start_row +(i+1)*aggregation_factor, ai_rows),
                j*aggregation_factor:min((j+1)*aggregation_factor, ai_cols)],
        i, j
    ) for i, j in zip(row_indices, col_indices)
)

for ai_agg_value, i, j in results:
    aggregated_ai_data[i, j]= ai_agg_value


######################################## 4 裁剪 #####################################

row_start, row_end, clipped_gt, clipped_rows  = clip_by_latitude(gt_template, template_rows, lat_min = min_lat_value, lat_max = max_lat_value)
print(f'row_start:{row_start}, row_end:{row_end}, clipped_rows:{clipped_rows}')

aggregated_ai_data = aggregated_ai_data[row_start:row_end, :]

# 计算裁剪后的尺寸
lon_min = clipped_gt[0]               ##起始经度
lon_max = clipped_gt[0] + clipped_gt[1]* template_cols  ##起始经度+像元分辨率*列数
lat_max = clipped_gt[3] ##起始纬度
lat_min = clipped_gt[3] + clipped_gt[5]* clipped_rows ##起始纬度+像元分辨率*行数数
print(f'经度范围：{lon_min}~{lon_max}°')
print(f'纬度范围：{lat_min}~{lat_max}°N')


############################################ 5 导出 ######################################################

output_path = os.path.join(tif_output, f'NH30_84_AI_{scale}km.tif')

save_tif_gdal(
    output_path,
    aggregated_ai_data,  # 取第 k 张
    crs_template,
    gt_template  # 使用新的地理变换参数
)

print('TIF export done!')




#### plot and export
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
import matplotlib.gridspec as gridspec


############################################## Plot ###################################################

conditions = [
    (aggregated_ai_data >= 0) & (aggregated_ai_data <= 300),      # 0-0.03 * 10000
    (aggregated_ai_data > 300) & (aggregated_ai_data <= 2000),    # 0.03-0.2 * 10000
    (aggregated_ai_data > 2000) & (aggregated_ai_data <= 3500),   # 0.2-0.35 * 10000
    (aggregated_ai_data > 3500) & (aggregated_ai_data <= 5000),   # 0.35-0.5 * 10000
    (aggregated_ai_data > 5000) & (aggregated_ai_data <= 6500),   # 0.5-0.65 * 10000
    (aggregated_ai_data > 6500)                                   # >0.65 * 10000
]

values = [1, 2, 3, 4, 5, 6]

# 使用np.select
aggregated_ai_data_simple = np.select(conditions, values, default=np.nan)

print('重分类 done！')

output_path = os.path.join(tif_output, f'NH30_84_AI(graident)_{scale}km.tif')

save_tif_gdal(
    output_path,
    aggregated_ai_data_simple,  # 取第 k 张
    crs_template,
    gt_template  # 使用新的地理变换参数
)
print('重分类TIF export done!')



