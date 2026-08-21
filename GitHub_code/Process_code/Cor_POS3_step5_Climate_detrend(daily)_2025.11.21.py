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



### Calculate daily climatology (multi-year daily mean)
def calculate_pixel_daily_climatology(pixel_time_series, dates):
    if not hasattr(dates, 'month'):
        dates = pd.Series(dates)

    daily_climatology = {}

    for month in range(1, 13):
        for day in range(1, 32):
            try:
                # Check whether the date is valid
                datetime.datetime(2001, month, day)

                # Select all data for the corresponding month and day
                mask = (dates.dt.month == month) & (dates.dt.day == day)
                if np.sum(mask) > 0:
                    # Extract the pixel values for the corresponding date
                    daily_values = pixel_time_series[mask]
                    # Calculate the multi-year mean for the day (ignoring NaN)
                    if np.sum(~np.isnan(daily_values)) > 0:
                        climatology = np.nanmean(daily_values)
                        # print(f'month:{month}, day:{day}, daily_values:{pixel_time_series_value}, daily_climatology[key]:{daily_climatology_value}')
                        daily_climatology[(month, day)] = climatology

            except ValueError:
                continue

    return daily_climatology



### Deseasonalization: subtract the multi-year daily mean
def deseasonalize_pixel(pixel_time_series, dates):

    if np.all(~np.isfinite(pixel_time_series)):

        # Create the deseasonalized time series
        deseasonalized_series = np.full_like(pixel_time_series, np.nan)

        return deseasonalized_series

    else:
        # # Calculate the daily climatology for the pixel
        # daily_climatology = calculate_pixel_daily_climatology(pixel_time_series, dates)

        # Create the deseasonalized time series
        deseasonalized_series = np.full_like(pixel_time_series, np.nan)

        for i, date in enumerate(dates):
            month = date.month
            day = date.day
            key = (month, day)

            if key in daily_climatology and not np.isnan(pixel_time_series[i]):
                # pixel_time_series_value = pixel_time_series[i]
                # daily_climatology_value = daily_climatology[key]
                # print(f'month:{month}, day:{day}, pixel_time_series[i]:{pixel_time_series_value}, daily_climatology[key]:{daily_climatology_value}')
                # Subtract the multi-year daily mean
                deseasonalized_series[i] = pixel_time_series[i] - daily_climatology[key]
            else:
                # If no climatological value is available for the day or the original data is NaN, keep NaN
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

    # Reshape the one-dimensional results into a three-dimensional array (time steps, rows, columns)
    deseason_data = np.array(deseason_data).reshape(rows, cols, time_steps).transpose(2, 0, 1)

    return deseason_data



### Detrending
def detrend_with_lowess_matrix(time_series, frac):

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

    return detrended_full


def detrend_stack_parallel(stack, n_jobs, frac=0.4):
    # Get the data dimensions (time steps, rows, columns)
    time_steps, rows, cols = stack.shape

    # Generate row and column indices for all pixels (i: row, j: column)
    row_indices = np.repeat(np.arange(rows), cols)  # Repeat each row index cols times
    col_indices = np.tile(np.arange(cols), rows)  # Tile the column indices rows times

    # print('detrend start2')
    # Process each pixel in parallel (extract time series → detrend → return results)
    # with parallel_backend('threading', n_jobs=n_jobs):
    detrended_pixels = Parallel(n_jobs=n_jobs, verbose = 10)(
        delayed(detrend_with_lowess_matrix)(
            stack[:, i, j],  # Extract the time series of pixel (i,j) (shape: time steps)
            frac=frac
        )
        for i, j in zip(row_indices, col_indices)  # Iterate over all row-column combinations
    )

    # Reshape the one-dimensional results into a three-dimensional array (time steps, rows, columns)
    detrended_stack = np.array(detrended_pixels).reshape(rows, cols, time_steps).transpose(2, 0, 1)

    return detrended_stack



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
    output_ds.SetGeoTransform(transform)  # Apply the adjusted geotransform parameters
    output_ds = None
    return True




###################################### 1 Data Input and Output Settings ################################################

print(list(colormaps))



######  ============== Input Settings to Modify ============= #########

pixel_resolution = 55

climate_var = 'VPD'  ##0-100cmSM / VPD

path = 'I:\Data\ERA5_Land'

folder = f'{path}\ERA5_Land_NH_{pixel_resolution}km_daily\ERA5_Land_NH_{pixel_resolution}km_daily_{climate_var}'


#########################
tif_files = sorted(glob.glob(os.path.join(folder, '*.tif')))
if not tif_files:
    raise FileNotFoundError("No TIF files were found!")


######## Output Settings
output_detrend_tif_path = f'{path}\ERA5_Land_NH_{pixel_resolution}km_daily_deseason_detrend\ERA5_Land_NH_{pixel_resolution}km_daily_{climate_var}(2001-2024)'


########################### 2 Basic Information ########################
first_tif = tif_files[0]
sample_tif = gdal.Open(first_tif)

if sample_tif is None:
    raise RuntimeError(f"Failed to open TIF file: {sample_tif} (unsupported driver or corrupted file)")

# Get the geotransform parameters
crs = sample_tif.GetProjectionRef()
gt = sample_tif.GetGeoTransform()
proj = sample_tif.GetProjection()

# Get the data dimensions
sample_array = sample_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
rows = sample_array.shape[0]
cols = sample_array.shape[1]
print('Original: rows=', rows, 'cols=', cols)


# Release the sample file
sample_tif = None


############################################ 3 Temporal Stacking ###################################################
## Time
tif_dates = []
for tif_file in tif_files:
    date = extract_date_from_filename(tif_file)
    tif_dates.append(date)
dates_pd = pd.Series(tif_dates)  # Convert to a pandas Series to use the dt attribute

print('First five dates:', [d.strftime('%Y%m%d') for d in tif_dates[:5]])
print('Last five dates:', [d.strftime('%Y%m%d') for d in tif_dates[-5:]])

## Data stacking
data_stack = []

for tif_file in tif_files:
    get_band(tif_file, data_stack)

print('Stack start!')
data_stack = np.stack(data_stack, axis=0)
# Release the files
tif_files = None
print('data_stack shape:\n', data_stack.shape)
print('Stack end!')

print(f'{climate_var} deseasonal start')

# Calculate the daily climatology for the pixel
daily_climatology = calculate_pixel_daily_climatology(data_stack, dates_pd)


############################ 4 Deseasonalization and Detrending ######################################
import gc

mid_row1 = rows // 3
mid_row2 = 2 * rows // 3

mid_col1 = cols // 3
mid_col2 = 2 * cols // 3

parts = [
    # First row
    (slice(None), slice(0, mid_row1),      slice(0, mid_col1)),      # Upper left
    (slice(None), slice(0, mid_row1),      slice(mid_col1, mid_col2)), # Upper middle
    (slice(None), slice(0, mid_row1),      slice(mid_col2, None)),      # Upper right

    # Second row
    (slice(None), slice(mid_row1, mid_row2), slice(0, mid_col1)),      # Middle left
    (slice(None), slice(mid_row1, mid_row2), slice(mid_col1, mid_col2)), # Middle middle
    (slice(None), slice(mid_row1, mid_row2), slice(mid_col2, None)),      # Middle right

    # Third row
    (slice(None), slice(mid_row2, None), slice(0, mid_col1)),      # Lower left
    (slice(None), slice(mid_row2, None), slice(mid_col1, mid_col2)), # Lower middle
    (slice(None), slice(mid_row2, None), slice(mid_col2, None)),      # Lower right
]


frac = 0.4

data_detrend = np.empty_like(data_stack, dtype=np.float32)

print(f'{climate_var} deseasonal + detrend start')

for i, idx in enumerate(parts):

    print(f'Processing block {i+1}/6')
    print(f'Deseasonal start')

    # ========================
    # 1. Deseasonalization
    # ========================
    part = deseasonal_parelle(
        data_stack[idx],
        dates_pd,
        n_jobs=18
    )

    # ========================
    # 2. Detrending
    # ========================
    print(f'Detrend start')
    part = detrend_stack_parallel(
        part,
        n_jobs=18,
        frac=frac
    )

    # Directly write back to the final array
    data_detrend[idx] = part

    print(f'Block {i+1} finished')

    del part
    gc.collect()

print(f'{climate_var} deseasonal + detrend finished')



############################################ 5 Export ######################################################
### Export data after detrending with date
for k, date in enumerate(tif_dates):
    date_str = date.strftime("%Y%m%d")  # Convert to a format such as 20200101
    output_path = os.path.join(
        output_detrend_tif_path,
        f'{climate_var}_deseason_{date_str}.tif'
    )

    save_tif_gdal(
        output_path,
        data_detrend[k, :, :],  # Select the kth layer
        crs,
        new_gt  # Apply the updated geotransform parameters
    )
print(f'{climate_var} export done!')