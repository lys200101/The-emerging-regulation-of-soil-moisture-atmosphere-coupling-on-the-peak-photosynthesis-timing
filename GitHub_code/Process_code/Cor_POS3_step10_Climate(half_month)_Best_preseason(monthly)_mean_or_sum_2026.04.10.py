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


######### Functions ##########

### Read time information, extract the date (last 8 digits) from each TIF filename
def extract_date_from_filename(filename):
    # Extract clean filename (without path and extension)
    basename = os.path.basename(filename)
    filename_without_ext = os.path.splitext(basename)[0]

    # Extract the last 8 characters
    date_str = filename_without_ext[-8:]

    if date_str[-1] == '1':
        # Last character is '1', set day to 01
        year_month = date_str[:6]  # Extract Year-Month part
        formatted_date = f"{year_month}01"  # Append 01 as day
        # print(f'formatted_date:{formatted_date}')
        return datetime.datetime.strptime(formatted_date, "%Y%m%d")

    elif date_str[-1] == '2':
        # Last character is '2', set day to 16
        year_month = date_str[:6]  # Extract Year-Month part
        formatted_date = f"{year_month}16"  # Append 16 as day
        # print(f'formatted_date:{formatted_date}')
        return datetime.datetime.strptime(formatted_date, "%Y%m%d")


def get_band(tif_file, stack):
    tif = gdal.Open(tif_file)
    data = tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
    stack.append(data)


def calculate_partial(y, x_idx, covars_list, min_years=12):
    """
    Calculate partial correlation between y and covars_list[x_idx], 
    controlling for all other variables in covars_list.
    """
    # Convert to numpy arrays
    y = np.asarray(y, dtype=float)
    X_all = np.column_stack([np.asarray(c, dtype=float) for c in covars_list])

    # Target independent variable and controlling variables
    x = X_all[:, x_idx]
    Z = np.delete(X_all, x_idx, axis=1)  # Remove x, remaining columns are control variables Z

    # Mask: exclude NaN and Inf values
    valid = np.isfinite(y) & np.isfinite(x) & np.all(np.isfinite(Z), axis=1)

    # Check degrees of freedom (n > k + 2)
    min_required = max(min_years, Z.shape[1] + 2)

    if np.sum(valid) < min_required:
        return np.nan

    x = x[valid]
    y = y[valid]
    Z = Z[valid]

    # Regression residuals approach for partial correlation
    Z_ = np.column_stack([np.ones(len(Z)), Z])

    # Residuals of x ~ Z
    beta_x, _, _, _ = np.linalg.lstsq(Z_, x, rcond=None)
    rx = x - Z_ @ beta_x

    # Residuals of y ~ Z
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

    for win_idx in range(3):  # Corresponding to 30, 60, 90 days
        # Prepare covariate list under current window
        current_covars = [
            sos_stack_pixel,       # 0
            cor_data[win_idx, :],  # 1
            sm_data[win_idx, :],   # 2
            vpd_data[win_idx, :],  # 3
            ta_data[win_idx, :],   # 4
            pre_data[win_idx, :],  # 5
            srad_data[win_idx, :]   # 6
        ]

        # Calculate partial correlation between POS and each climate variable under current window
        for v_idx in range(6):
            # v_idx starts from 0, corresponding index in current_covars is v_idx + 1
            pcor_matrix[v_idx, win_idx] = calculate_partial(
                y=pos_stack_pixel,
                x_idx=v_idx + 1,
                covars_list=current_covars
            )

    best_lens = []
    for v_idx in range(6):
        vals = pcor_matrix[v_idx, :]
        if np.any(np.isfinite(vals)):
            # Pick the index with maximum absolute value; +1 converts index (0,1,2) to length class (1,2,3)
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

    return (i, j, *best_lens)  # Unpack list with * operator


def save_tif_gdal(output_path, data, crs, transform):
    """Save TIFF file, automatically retrieving array shape and applying spatial transformation."""
    rows, cols = data.shape
    driver = gdal.GetDriverByName("GTiff")

    output_ds = driver.Create(
        output_path,
        cols, rows, 1, gdal.GDT_Float32
    )
    if not output_ds:
        raise RuntimeError(f"Unable to create output file: {output_path}")

    output_band = output_ds.GetRasterBand(1)
    output_band.WriteArray(data, 0, 0)
    output_band.SetNoDataValue(np.nan)  # Set NoData value

    output_ds.SetProjection(crs)
    output_ds.SetGeoTransform(transform)  # Apply geo-transform parameters
    output_ds = None
    return True


###################################### 1 Data Loading & Output Settings ################################################
###################### ===================== Input Settings ======================== ########################
#### Input SM and VPD tif files    ### Please modify here carefully ⬇⬇⬇⬇⬇⬇⬇⬇

star_year = 2001
end_year = 2024
years_length = end_year - star_year + 1
print('years_length:', years_length)
years = range(star_year, end_year + 1)

scale = 55

input_same_path = 'D:\CAU\phenology_swc_vpd\paper\Data_and_code_online\FigShare_data'

Basedon = 'Based_on_detrendPheno'  ### Based_on_detrendPheno: Use detrended SOS and POS for partial correlation
                                   ### Based_on_OriginPheno: Use original SOS and POS for partial correlation

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


#### Input POS and SOS
if Basedon == 'Based_on_detrendPheno':
    pos_folder = fr'{input_same_path}\POSdetrend_55km'  # start
    sos_folder = fr'{input_same_path}\SOSdetrend_55km'  # start
elif Basedon == 'Based_on_OriginPheno':
    pos_folder = fr'{input_same_path}\POS_55km'         # start
    sos_folder = fr'{input_same_path}\SOS_55km'         # start


###################### ===================== Output Settings ======================== ########################
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
    raise FileNotFoundError("No tif_files_sm_1 TIF files found!")
if not tif_files_sm_2:
    raise FileNotFoundError("No tif_files2 TIF files found!")
if not tif_files_sm_3:
    raise FileNotFoundError("No tif_files3 TIF files found!")
if not pos_tif_files:
    raise FileNotFoundError("No POS TIF files found!")


####################### 2 Extract TIF Metadata ############################
first_tif = tif_files_sm_1[0]
sample_tif = gdal.Open(first_tif)

if sample_tif is None:
    raise RuntimeError(f"Unable to open TIF file: {sample_tif} (unsupported driver or corrupted file)")

# Get spatial transformation parameters: Projection and pixel size
# Coordinates and projection    Coordinate Reference System (CRS): Spatial reference framework for the data
crs = sample_tif.GetProjectionRef()          # Automatically retrieve input CRS
# print('crs:', crs)
gt = sample_tif.GetGeoTransform()  # GeoTransform: Longitude/Latitude mathematical parameters converting pixel coords to spatial coords
proj = sample_tif.GetProjection()  # Projected coordinates: xy (units in meters)

# Pixel resolution
pixel_width = gt[1]
pixel_height = gt[5]

top_left_x = gt[0]
top_left_y = gt[3]

# Number of rows and columns
sample_tif = sample_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)

rows = sample_tif.shape[0]
cols = sample_tif.shape[1]
print('rows:', rows, 'cols:', cols)

row_indices = np.repeat(np.arange(rows), cols)  # Repeat row indices `cols` times
col_indices = np.tile(np.arange(cols), rows)    # Tile column indices `rows` times

# Calculate spatial extent (handles negative pixel_height)
lon_min = top_left_x
lon_max = top_left_x + cols * pixel_width   # Right longitude boundary
lat_min = top_left_y + rows * pixel_height  # Bottom latitude boundary (Southernmost, smaller value)
lat_max = top_left_y                        # Top latitude boundary (Northernmost, larger value)
print(f"Longitude extent: {lon_min:.6f} -> {lon_max:.6f}")
print(f"Latitude extent: {lat_min:.6f} -> {lat_max:.6f}")


############################ 3 Data Stacking #############################
print('All stack start!')

## Data stacking initialization
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

cor_stack_1 = np.stack(cor_stack_1, axis=0)  # [:, 505:510, 505:510]
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


######################################### 4 Pixel-wise Optimal Preseason Length Calculation ################################################
cor_preseason_len = np.full((rows, cols), np.nan)
sm_preseason_len = np.full((rows, cols), np.nan)
vpd_preseason_len = np.full((rows, cols), np.nan)
ta_preseason_len = np.full((rows, cols), np.nan)
pre_preseason_len = np.full((rows, cols), np.nan)
srad_preseason_len = np.full((rows, cols), np.nan)


print('Calculating optimal preseason time window for each pixel:')
# Parallel processing across all pixels (using 15 CPU threads)
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
print('Pixel-wise optimal time window calculation complete. Starting file exports.')

####################### Save Optimal Preseason Length TIF Files ####################################
output_path0 = os.path.join(ouput_path, "Cor_preseason_length.tif")
save_tif_gdal(
    output_path0,
    cor_preseason_len,
    crs, gt
)

output_path1 = os.path.join(ouput_path, "SM_preseason_length.tif")
save_tif_gdal(
    output_path1,
    sm_preseason_len,
    crs, gt
)

output_path2 = os.path.join(ouput_path, "VPD_preseason_length.tif")
save_tif_gdal(
    output_path2,
    vpd_preseason_len,
    crs, gt
)

output_path3 = os.path.join(ouput_path, "Ta_preseason_length.tif")
save_tif_gdal(
    output_path3,
    ta_preseason_len,
    crs, gt
)

output_path4 = os.path.join(ouput_path, "Pre_preseason_length.tif")
save_tif_gdal(
    output_path4,
    pre_preseason_len,
    crs, gt
)

output_path5 = os.path.join(ouput_path, "Srad_preseason_length.tif")
save_tif_gdal(
    output_path5,
    srad_preseason_len,
    crs, gt
)

print("Optimal preseason length results for Coupling Effect, SM, VPD, Ta, Pre, and Srad saved successfully.")


####################### Calculate Annual Means / Totals using Optimal Preseason ####################################

folder_cor_pvalue_1 = fr'{input_path}\Correlation(SM_VPD_pearson)17_8_1\Pvalue'  #(POS-30) - POS(17_8_1)
folder_cor_pvalue_2 = fr'{input_path}\Correlation(SM_VPD_pearson)17_8_2\Pvalue'  #(POS-60) - POS(17_8_2)
folder_cor_pvalue_3 = fr'{input_path}\Correlation(SM_VPD_pearson)17_8_3\Pvalue'  #(POS-90) - POS(17_8_3)

tif_files_cor_pvalue_1 = sorted(glob.glob(os.path.join(folder_cor_pvalue_1, '*.tif')))
tif_files_cor_pvalue_2 = sorted(glob.glob(os.path.join(folder_cor_pvalue_2, '*.tif')))
tif_files_cor_pvalue_3 = sorted(glob.glob(os.path.join(folder_cor_pvalue_3, '*.tif')))

cor_pvalue_stack_1 = []
cor_pvalue_stack_2 = []
cor_pvalue_stack_3 = []

for tif_file in tif_files_cor_pvalue_1:
    get_band(tif_file, cor_pvalue_stack_1)
for tif_file in tif_files_cor_pvalue_2:
    get_band(tif_file, cor_pvalue_stack_2)
for tif_file in tif_files_cor_pvalue_3:
    get_band(tif_file, cor_pvalue_stack_3)

cor_pvalue_stack_1 = np.stack(cor_pvalue_stack_1, axis=0)  # [:, 505:510, 505:510]
cor_pvalue_stack_2 = np.stack(cor_pvalue_stack_2, axis=0)
cor_pvalue_stack_3 = np.stack(cor_pvalue_stack_3, axis=0)
cor_pvalue_stack = np.stack([cor_pvalue_stack_1[:years_length, :, :], cor_pvalue_stack_2[:years_length, :, :], cor_pvalue_stack_3[:years_length, :, :]], axis=0)


cor_pre_stack = np.full((years_length, rows, cols), np.nan)
cor_pvalue_pre_stack = np.full((years_length, rows, cols), np.nan)
sm_pre_stack = np.full((years_length, rows, cols), np.nan)
vpd_pre_stack = np.full((years_length, rows, cols), np.nan)
ta_pre_stack = np.full((years_length, rows, cols), np.nan)
pre_pre_stack = np.full((years_length, rows, cols), np.nan)
srad_pre_stack = np.full((years_length, rows, cols), np.nan)


def get_preseason_data(i, j,
                       cor_len, cor_data, cor_pvalue_data,
                       sm_len, sm_data,
                       vpd_len, vpd_data,
                       ta_len, ta_data,
                       pre_len, pre_data,
                       srad_len, srad_data):

    def extract_by_len(length_val, data_stack):
        if np.isnan(length_val):
            return np.full(years_length, np.nan)
        idx = int(length_val) - 1  # Crucial: Convert length tags (1, 2, 3) to 0-indexed values (0, 1, 2)
        return data_stack[idx, :]

    cor_preseason_value = extract_by_len(cor_len, cor_data)
    cor_pvalue_preseason_value = extract_by_len(cor_len, cor_pvalue_data)
    sm_preseason_value  = extract_by_len(sm_len, sm_data)
    vpd_preseason_value = extract_by_len(vpd_len, vpd_data)
    ta_preseason_value  = extract_by_len(ta_len, ta_data)
    pre_preseason_value = extract_by_len(pre_len, pre_data)
    srad_preseason_value = extract_by_len(srad_len, srad_data)

    return (i, j, cor_preseason_value, cor_pvalue_preseason_value, sm_preseason_value, vpd_preseason_value,
                  ta_preseason_value, pre_preseason_value, srad_preseason_value)


results = Parallel(n_jobs=15)(
    delayed(get_preseason_data)(
        i, j,
        cor_preseason_len[i, j], cor_stack[:, :, i, j], cor_pvalue_stack[:, :, i, j],
        sm_preseason_len[i, j], sm_stack[:, :, i, j],
        vpd_preseason_len[i, j], vpd_stack[:, :, i, j],
        ta_preseason_len[i, j], ta_stack[:, :, i, j],
        pre_preseason_len[i, j], pre_stack[:, :, i, j],
        srad_preseason_len[i, j], srad_stack[:, :, i, j]
    ) for i, j in zip(row_indices, col_indices)
)

for (i, j, cor_preseason_value, cor_pvalue_preseason_value, sm_preseason_value, vpd_preseason_value,
    ta_preseason_value, pre_preseason_value, srad_preseason_value) in results:
    cor_pre_stack[:, i, j] = cor_preseason_value
    cor_pvalue_pre_stack[:, i, j] = cor_pvalue_preseason_value
    sm_pre_stack[:, i, j] = sm_preseason_value
    vpd_pre_stack[:, i, j] = vpd_preseason_value
    ta_pre_stack[:, i, j] = ta_preseason_value
    pre_pre_stack[:, i, j] = pre_preseason_value
    srad_pre_stack[:, i, j] = srad_preseason_value

print('Preseason match done!')


tif_path = 'D:\climate_in_best_preseason'
for year in years:
    k = year - 2001
    out_put = os.path.join(tif_path, rf'Correlation(SM_VPD_pearson)\Cor_pearson_{year}.tif')
    save_tif_gdal(out_put, cor_pre_stack[k, :, :], crs, gt)

    out_put = os.path.join(tif_path, rf'Correlation(SM_VPD_pearson)\Pvalue\Cor_pearson_pvalue_{year}.tif')
    save_tif_gdal(out_put, cor_pvalue_pre_stack[k, :, :], crs, gt)

    out_put = os.path.join(tif_path, rf'Pre_preseason_sum\Pre_pearson_sum_{year}.tif')
    save_tif_gdal(out_put, pre_pre_stack[k, :, :], crs, gt)

    out_put = os.path.join(tif_path, rf'SM_preseason_mean\SM_pearson_mean_{year}.tif')
    save_tif_gdal(out_put, sm_pre_stack[k, :, :], crs, gt)

    out_put = os.path.join(tif_path, rf'Srad_preseason_sum\Srad_pearson_sum_{year}.tif')
    save_tif_gdal(out_put, srad_pre_stack[k, :, :], crs, gt)

    out_put = os.path.join(tif_path, rf'Ta_preseason_mean\Ta_pearson_mean_{year}.tif')
    save_tif_gdal(out_put, ta_pre_stack[k, :, :], crs, gt)

    out_put = os.path.join(tif_path, rf'VPD_preseason_mean\VPD_pearson_mean_{year}.tif')
    save_tif_gdal(out_put, vpd_pre_stack[k, :, :], crs, gt)