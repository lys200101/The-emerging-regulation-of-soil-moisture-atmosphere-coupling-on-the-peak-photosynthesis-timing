
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
def extract_date_from_filename(filename, data_belong_to):
    # 提取纯文件名（不含路径和扩展名）
    basename = os.path.basename(filename)
    filename_without_ext = os.path.splitext(basename)[0]

    if data_belong_to == 'climate':
        date_str = filename_without_ext[-8:]

        # 验证日期格式
        if not date_str.isdigit() or len(date_str) != 8:
            raise ValueError(f"文件 {filename} 的最后8位不是有效日期（需为YYYYMMDD）！")

        # 转换为 datetime 对象
        return datetime.datetime.strptime(date_str, "%Y%m%d")

    elif data_belong_to == 'pheno':
        date_str = filename_without_ext[-4:]

        # 验证日期格式
        if not date_str.isdigit() or len(date_str) != 4:
            raise ValueError(f"文件 {filename} 的最后4位不是有效年份（需为YYYY）！")

        # 转换为 datetime 对象
        return datetime.datetime.strptime(date_str, "%Y")



### 读取SM和VPD波段
def get_band(tif_file):
    tif = gdal.Open(tif_file)
    data = tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
    # stack.append(data)
    tif = None  # 及时释放资源
    return data


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
        valid_values = window[np.isfinite(window) & (window!=0) & (window!=-9999) & (window!= 90) & (window!= 300)]

        if len(valid_values) > 0:
            result = np.nanmean(valid_values)
        if len(valid_values) == 0:
            result = np.nan
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

aggregation_factor = 5  ### magnification
aggregate_size = 11*aggregation_factor

min_lat_value = 30
max_lat_value = 84

path = fr'D:'

data_belong_to = 'pheno'  ### climate / pheno

## Aggregate climate data
if data_belong_to == 'climate':
    climate_var = 'Srad'    ##√0-100cmSM / VPD / Ta / Pre / Srad
    folder = fr'{path}\ERA5_Land_NH_11km_daily\ERA5_Land_NH_11km_daily_{climate_var}'
    output_tif_path = fr'{path}\ERA5_Land_NH_{aggregate_size}km_daily\ERA5_Land_NH_{aggregate_size}km_daily_{climate_var}'

## Aggregate pheno data
if data_belong_to == 'pheno' :
    pheno = 'POS'
    folder = fr'{path}\{pheno}_11km'
    output_tif_path = fr'{path}\{pheno}_{aggregate_size}km'

#########################
tif_files = sorted(glob.glob(os.path.join(folder, '*.tif')))
if not tif_files:
    raise FileNotFoundError("未找到任何 TIF 文件！")


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
print('rows=', rows, 'cols=', cols)

# 释放样本文件
sample_tif = None


############################################ 3 时间-堆叠 ###################################################
### Aggregate parmeter change
new_rows = rows // aggregation_factor
new_cols = cols // aggregation_factor
# print(f'聚合后形状（导出形状）: {new_rows} x {new_cols}')

# 更新地理变换参数
new_gt = (
    gt[0],  # 左上角x坐标
    gt[1] * aggregation_factor,  # 像素宽度
    gt[2],
    gt[3],  # 左上角y坐标
    gt[4],
    gt[5] * aggregation_factor  # 像素高度
)

### Aggregate
for tif_file in tif_files:
    ### Time
    date = extract_date_from_filename(tif_file, data_belong_to)

    ### Data
    tif_data = get_band(tif_file)

    print("Aggregate start!")
    ### Aggregate
    aggregated_data = aggregate_function(tif_data, aggregation_factor, rows, cols, new_rows, new_cols)
    print(f"Aggregate done!")

    # 获取聚合后的样本数据用于后续处理
    rows_agg, cols_agg = aggregated_data.shape[0], aggregated_data.shape[1]
    # print(f'聚合后形状（导出形状）: {rows_agg} x {cols_agg}')

    row_start, row_end, new_gt = clip_by_latitude(new_gt, rows_agg, lat_min=min_lat_value, lat_max=max_lat_value)
    print(f'row_start:{row_start}, row_end:{row_end}')

    # 计算裁剪后的尺寸
    lon_min = new_gt[0]  ##起始经度
    lon_max = new_gt[0] + new_gt[1] * cols_agg  ##起始经度+像元分辨率*列数
    lat_max = new_gt[3]  ##起始纬度
    lat_min = new_gt[3] + new_gt[5] * (row_end - row_start)  ##起始纬度+像元分辨率*行数数
    print(f'经度范围：{lon_min}~{lon_max}°')
    print(f'纬度范围：{lat_min}~{lat_max}°N')


    ############################################ 5 导出 ######################################################

    ## Aggregate climate data
    if data_belong_to == 'climate':
        date_str = date.strftime("%Y%m%d")  # 转成类似 20200101 的格式
        output_path = os.path.join(
            output_tif_path,
            f'NH_{climate_var}_{aggregate_size}km_{date_str}.tif'
        )

    ## Aggregate pheno data
    if data_belong_to == 'pheno':
        date_str = date.strftime("%Y")  # 转成类似 20200101 的格式
        output_path = os.path.join(
            output_tif_path,
            f'{pheno}1_aggMean_{aggregate_size}000m_{date_str}.tif'
        )

    save_tif_gdal(
        output_path,
        aggregated_data[row_start: row_end, :],  # 取第 k 张
        crs,
        new_gt  # 使用新的地理变换参数
    )
    if data_belong_to == 'climate':
        print(f'{date} {climate_var} export done!')
    if data_belong_to == 'pheno':
        print(f'{date} {pheno} export done!')

if data_belong_to == 'climate':
    print(f'{climate_var} export done!')
if data_belong_to == 'pheno':
    print(f'{pheno} export done!')