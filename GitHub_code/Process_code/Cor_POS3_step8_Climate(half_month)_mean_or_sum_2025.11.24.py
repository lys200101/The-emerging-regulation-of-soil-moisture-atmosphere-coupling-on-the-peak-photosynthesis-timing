
import os
import glob
import datetime
import numpy as np
import pandas as pd
from osgeo import gdal
from joblib import Parallel, delayed, parallel_backend

######### Functions ##########

def extract_date_from_filename(filename):
    basename = os.path.basename(filename)
    filename_without_ext = os.path.splitext(basename)[0]
    date_str = filename_without_ext[-8:]

    if date_str[-1] == '1':
        year_month = date_str[:6]
        formatted_date = f"{year_month}01"
        return datetime.datetime.strptime(formatted_date, "%Y%m%d")
    elif date_str[-1] == '2':
        year_month = date_str[:6]
        formatted_date = f"{year_month}16"
        return datetime.datetime.strptime(formatted_date, "%Y%m%d")


def load_stack(tif_files):
    """Load a list of TIF files into a 3D numpy array [time, rows, cols] efficiently."""
    stack = []
    for f in tif_files:
        ds = gdal.Open(f)
        band = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
        stack.append(band)
        ds = None
    return np.stack(stack, axis=0)


def save_tif_gdal(output_path, data, crs, transform):
    """Save a 2D numpy array to a GeoTIFF file."""
    rows, cols = data.shape
    driver = gdal.GetDriverByName("GTiff")
    output_ds = driver.Create(output_path, cols, rows, 1, gdal.GDT_Float32)
    if not output_ds:
        raise RuntimeError(f"无法创建输出文件: {output_path}")

    output_band = output_ds.GetRasterBand(1)
    output_band.WriteArray(data, 0, 0)
    output_band.SetNoDataValue(np.nan)

    output_ds.SetProjection(crs)
    output_ds.SetGeoTransform(transform)
    output_ds = None
    return True


###################################### 1. Data I/O & Configuration ################################################

star_year = 2001
end_year = 2024
years_length = end_year - star_year + 1
years = range(star_year, end_year + 1)

test_number = '3'
if test_number == '1':
    interval = 30
elif test_number == '2':
    interval = 60
elif test_number == '3':
    interval = 90

scale = 55

input_path = f'I:\\Data\\ERA5_Land\\ERA5_Land_NH_{scale}km_daily_deseason_half_month'

folder1 = f'{input_path}\\ERA5_Land_NH_{scale}km_half_month_0-100cmSM(2001-2024)'
folder2 = f'{input_path}\\ERA5_Land_NH_{scale}km_half_month_VPD(2001-2024)'
folder3 = f'{input_path}\\ERA5_Land_NH_{scale}km_half_month_Ta(2001-2024)'
folder4 = f'{input_path}\\ERA5_Land_NH_{scale}km_half_month_Pre(2001-2024)'
folder5 = f'{input_path}\\ERA5_Land_NH_{scale}km_half_month_Srad(2001-2024)'

pos_folder = f'D:\\POS_{scale}km'
sos_folder = f'D:\\SOS_{scale}km'

output_path = f'D:\\Climate_data'
os.makedirs(output_sm_tif_path := f'{output_path}\\SM_preseason_mean{test_number}', exist_ok=True)
os.makedirs(output_vpd_tif_path := f'{output_path}\\VPD_preseason_mean{test_number}', exist_ok=True)
os.makedirs(output_ta_tif_path := f'{output_path}\\Ta_preseason_mean{test_number}', exist_ok=True)
os.makedirs(output_pre_tif_path := f'{output_path}\\Pre_preseason_sum{test_number}', exist_ok=True)
os.makedirs(output_srad_tif_path := f'{output_path}\\Srad_preseason_sum{test_number}', exist_ok=True)

tif_files1 = sorted(glob.glob(os.path.join(folder1, '*.tif')))
tif_files2 = sorted(glob.glob(os.path.join(folder2, '*.tif')))
tif_files3 = sorted(glob.glob(os.path.join(folder3, '*.tif')))
tif_files4 = sorted(glob.glob(os.path.join(folder4, '*.tif')))
tif_files5 = sorted(glob.glob(os.path.join(folder5, '*.tif')))
pos_tif_files = sorted(glob.glob(os.path.join(pos_folder, '*.tif')))
sos_tif_files = sorted(glob.glob(os.path.join(sos_folder, '*.tif'))) if sos_folder != 'no' else []

for f, name in zip([tif_files1, tif_files2, tif_files3, tif_files4, tif_files5, pos_tif_files], 
                   ['SM', 'VPD', 'Ta', 'Pre', 'Srad', 'POS']):
    if not f:
        raise FileNotFoundError(f"未找到任何 {name} TIF 文件！")

###################################### 2. Metadata Extraction ################################################

sample_ds = gdal.Open(tif_files1[0])
crs = sample_ds.GetProjectionRef()
gt = sample_ds.GetGeoTransform()
rows = sample_ds.RasterYSize
cols = sample_ds.RasterXSize
sample_ds = None

print(f'Rows: {rows}, Cols: {cols}')

tif_dates = np.array([extract_date_from_filename(f) for f in tif_files1])

###################################### 3. Data Stacking ################################################

print('All stack start!')
sm_stack = load_stack(tif_files1)
vpd_stack = load_stack(tif_files2)
ta_stack = load_stack(tif_files3)
pre_stack = load_stack(tif_files4)
srad_stack = load_stack(tif_files5)

pos_stack = load_stack(pos_tif_files)
sos_stack = load_stack(sos_tif_files) if sos_folder != 'no' else None
print('All stack done!')

###################################### 4. Processing & Computation ################################################

for year in years:
    print(f"正在处理年份：{year}")
    k = year - star_year

    year_mask = (tif_dates >= datetime.datetime(year, 1, 1)) & (tif_dates <= datetime.datetime(year, 12, 31))
    year_idx = np.where(year_mask)[0]
    year_dates = tif_dates[year_idx]

    sm_year = sm_stack[year_idx]
    vpd_year = vpd_stack[year_idx]
    ta_year = ta_stack[year_idx]
    pre_year = pre_stack[year_idx]
    srad_year = srad_stack[year_idx]

    pos_year = pos_stack[k]
    sos_year = sos_stack[k] if sos_stack is not None else None

    # Initialize output matrices for the year
    sm_mean_out = np.full((rows, cols), np.nan)
    vpd_mean_out = np.full((rows, cols), np.nan)
    ta_mean_out = np.full((rows, cols), np.nan)
    pre_sum_out = np.full((rows, cols), np.nan)
    srad_sum_out = np.full((rows, cols), np.nan)

    # Vectorized / Block processing across pixels to avoid per-pixel function overhead
    for i in range(rows):
        for j in range(cols):
            p_val = pos_year[i, j]
            s_val = sos_year[i, j] if sos_year is not None else 0

            if pd.isna(p_val):
                continue

            # Calculate time window dates
            start_date1 = datetime.datetime(year, 1, 1) + datetime.timedelta(days=int(p_val - interval))
            end_date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=int(p_val))

            start_month = start_date1.month
            if start_date1.day <= 15:
                start_date2 = datetime.datetime(year, start_month, 1)
            else:
                start_date2 = datetime.datetime(year, start_month, 16)

            valid_mask = (year_dates >= start_date2) & (year_dates <= end_date)
            if not np.any(valid_mask):
                continue

            sm_mean_out[i, j] = np.nanmean(sm_year[valid_mask, i, j])
            vpd_mean_out[i, j] = np.nanmean(vpd_year[valid_mask, i, j])
            ta_mean_out[i, j] = np.nanmean(ta_year[valid_mask, i, j])
            pre_sum_out[i, j] = np.nansum(pre_year[valid_mask, i, j])
            srad_sum_out[i, j] = np.nansum(srad_year[valid_mask, i, j])

    # Save results
    save_tif_gdal(os.path.join(output_sm_tif_path, f"SM_pearson_mean_{year}.tif"), sm_mean_out, crs, gt)
    save_tif_gdal(os.path.join(output_vpd_tif_path, f"VPD_pearson_mean_{year}.tif"), vpd_mean_out, crs, gt)
    save_tif_gdal(os.path.join(output_ta_tif_path, f"Ta_pearson_mean_{year}.tif"), ta_mean_out, crs, gt)
    save_tif_gdal(os.path.join(output_pre_tif_path, f"Pre_pearson_sum_{year}.tif"), pre_sum_out, crs, gt)
    save_tif_gdal(os.path.join(output_srad_tif_path, f"Srad_pearson_sum_{year}.tif"), srad_sum_out, crs, gt)

    print(f"{year}年结果已保存")