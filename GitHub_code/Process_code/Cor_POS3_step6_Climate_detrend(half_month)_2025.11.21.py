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

### Read temporal information and extract the date from each TIF file (last 8 digits)
def extract_date_from_filename(filename):
    # Extract the filename without the path or extension
    basename = os.path.basename(filename)
    filename_without_ext = os.path.splitext(basename)[0]

    # Extract the last 8 digits
    date_str = filename_without_ext[-8:]

    # Validate the date format
    if not date_str.isdigit() or len(date_str) != 8:
        raise ValueError(f"The last 8 digits of file {filename} are not a valid date (expected YYYYMMDD)!")

    # Convert to a datetime object
    return datetime.datetime.strptime(date_str, "%Y%m%d")



### Read the SM and VPD bands
def get_band(tif_file, stack):
    tif = gdal.Open(tif_file)
    climate_data = tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
    stack.append(climate_data)
    tif = None  # Release the resource promptly


### Calculate half-month climatology
def calculate_half_month_mean_or_sum(pixel_time_series, dates, var_cal_method, i, j):
    """
    Calculate the mean or total value for each half-month of each year.

    Returns: (i, j, yearly_half_month_values)
    yearly_half_month_values: {(year, month, half): value}
    """
    # Ensure that dates are in datetime format
    dates = pd.Series(pd.to_datetime(dates))

    # Get the year range
    years = np.unique(dates.dt.year)

    # Store the values for each half-month of each year
    yearly_half_month_value = {}

    for year in years:
        # Select data for the corresponding year
        year_mask = dates.dt.year == year

        for month in range(1, 13):
            # Select data for the corresponding month
            month_mask = dates.dt.month == month
            full_mask = year_mask & month_mask

            if not np.any(full_mask):
                # Set to NaN if no data are available for the month
                yearly_half_month_value[(year, month, 1)] = np.nan
                yearly_half_month_value[(year, month, 2)] = np.nan
                continue

            # Get all data for the month
            month_dates = dates[full_mask]
            month_values = pixel_time_series[full_mask]

            # Calculate the first half of the month (days 1-15)
            early_mask = month_dates.dt.day <= 15
            early_values = month_values[early_mask]

            # Calculate the second half of the month (days 16-end of month)
            late_mask = month_dates.dt.day > 15
            late_values = month_values[late_mask]

            # Calculate the mean or total value
            if var_cal_method == 'mean':
                month_early_value = np.nanmean(early_values) if len(early_values) > 0 else np.nan
                month_late_value = np.nanmean(late_values) if len(late_values) > 0 else np.nan
            elif var_cal_method == 'sum':
                month_early_value = np.nansum(early_values) if len(early_values) > 0 else np.nan
                month_late_value = np.nansum(late_values) if len(late_values) > 0 else np.nan

            # Store the results
            yearly_half_month_value[(year, month, 1)] = month_early_value  # First half of the month
            yearly_half_month_value[(year, month, 2)] = month_late_value  # Second half of the month

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
    Calculate half-month-scale anomalies for a single pixel (including climatology).

    Parameters
    ----------
    pixel_time_series : 1D array (time,)
        Daily time series for a single pixel
    dates : list or array-like of datetime
        Dates corresponding to pixel_time_series
    var_cal_method : str
        'mean' or 'sum'
    half_month_keys : list of (year, month, half)
        Temporal sequence
    i, j : int
        Pixel row and column indices (used for writing results back)

    Returns
    -------
    i, j, anomalies : (int, int, 1D array)
        Anomalies in exactly the same order as half_month_keys
    """

    dates = pd.to_datetime(dates)
    years = np.unique(dates.year)

    # ---------- 1. Calculate values for each year and half-month ----------
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

            # First half of the month
            early_mask = month_dates.day <= 15
            early_vals = month_values[early_mask]

            # Second half of the month
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

    # ---------- 2. Calculate half-month climatology ----------
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

    # ---------- 3. Calculate anomalies (in chronological order) ----------
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


### Detrending
def detrend_with_lowess_matrix(time_series, i, j, frac):
    if np.all(~np.isfinite(time_series)):
        detrended_full = np.full_like(time_series, np.nan)
    else:
        # Check the valid data points
        valid_mask = ~np.isnan(time_series)
        valid_data = time_series[valid_mask]

        # Generate the time index
        y = np.arange(len(valid_data))
        # LOWESS smoothing
        smoothed = lowess(valid_data, y, frac=frac, return_sorted=False)
        # Detrend
        detrended_valid = valid_data - smoothed

        # Reconstruct the complete time series
        detrended_full = np.full_like(time_series, np.nan)
        detrended_full[valid_mask] = detrended_valid

        if i == 50 and j == 500:
            print(f'valid_data:{valid_data}')
            print(f'smoothed:{smoothed}')
            print(f'detrended_valid:{detrended_valid}')

    return detrended_full


def save_tif_gdal(output_path, data, crs, transform):
    """Save a TIFF file, automatically obtaining the data dimensions and applying the geotransform."""
    rows, cols = data.shape
    driver = gdal.GetDriverByName("GTiff")

    output_ds = driver.Create(
        output_path,
        cols, rows, 1, gdal.GDT_Float32
    )
    if not output_ds:
        raise RuntimeError(f"Failed to create output file: {output_path}")

    output_band = output_ds.GetRasterBand(1)
    output_band.WriteArray(data, 0, 0)
    output_band.SetNoDataValue(np.nan)  # Set NaN as the NoData value

    output_ds.SetProjection(crs)
    output_ds.SetGeoTransform(transform)
    output_ds = None
    return True


###################################### 1 Data Input and Output Settings ################################################
######  ============== Input Settings to Modify ============= #########

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
    raise FileNotFoundError("No TIF files were found!")

######## Output Settings
output_detrend_tif_path = f'{path}/ERA5_Land_NH_{pixel_resolution}km_daily_deseason_half_month/ERA5_Land_NH_{pixel_resolution}km_half_month_{climate_var}(2001-2024)'

########################### 2 Basic Information ########################
first_tif = tif_files[0]
sample_tif = gdal.Open(first_tif)

if sample_tif is None:
    raise RuntimeError(f"Failed to open TIF file: {first_tif} (unsupported driver or corrupted file)")

# Get the geotransform parameters
crs = sample_tif.GetProjectionRef()
gt = sample_tif.GetGeoTransform()

# Get the data dimensions
sample_array = sample_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
rows = sample_array.shape[0]
cols = sample_array.shape[1]
print('Original: rows=', rows, 'cols=', cols)

# Create row and column indices
row_indices = np.repeat(np.arange(rows), cols)
col_indices = np.tile(np.arange(cols), rows)

# Release the sample file
sample_tif = None

############################################ 3 Temporal Stacking ###################################################
## Time
tif_dates = []

for tif_file in tif_files:
    date = extract_date_from_filename(tif_file)
    tif_dates.append(date)

print('First five dates:', [d.strftime('%Y%m%d') for d in tif_dates[:5]])
print('Last five dates:', [d.strftime('%Y%m%d') for d in tif_dates[-5:]])

## Data stacking
data_stack = []

for tif_file in tif_files:
    get_band(tif_file, data_stack)

print('Stack start!')
data_stack = np.stack(data_stack, axis=0)
print('data_stack shape:', data_stack.shape)
print('Stack end!')

########################################### 5 Deseasonalization (half-month scale) ###################################################
print(f'{climate_var} deseasonal start')

# 5.0 Aggregation method
if climate_var in ('Ta', '0-100cmSM', 'VPD'):
    var_cal_method = 'mean'
elif climate_var in ('Pre', 'Srad'):
    var_cal_method = 'sum'

# 5.1 Half-month temporal axis
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


########################################### 6 Detrending ###################################################
print(f'{climate_var} detrend start')
# Detrend using LOWESS
frac_value = 0.4  # LOWESS smoothing parameter, which can be adjusted as needed

detrended_data = Parallel(n_jobs=15)(
    delayed(detrend_with_lowess_matrix)(
        half_month_output[:, i, j], i, j,
        frac=frac_value
    )
    for i, j in zip(row_indices, col_indices)
)

# Reshape into a three-dimensional array
detrended_stack = np.array(detrended_data).reshape(rows, cols, half_time_length).transpose(2, 0, 1)
print(f'{climate_var} detrend end')
print(f'detrended_stack shape: {detrended_stack.shape}')

########################################### 7 Save Results ###################################################

# Save the results for each time step
print("Saving result TIFF files...")
for t in range(half_time_length):
    year = half_month_keys[t][0]
    month = half_month_keys[t][1]
    half = half_month_keys[t][2]

    output_filename = f"{climate_var}_deseason_{year}{month:02d}_{half}.tif"
    output_path = os.path.join(output_detrend_tif_path, output_filename)

    save_tif_gdal(output_path, detrended_stack[t, :, :], crs, gt)

print("Processing completed!")