import os
import glob
import datetime
import pandas as pd
from osgeo import gdal
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from joblib import Parallel, delayed, parallel_backend
from matplotlib import colormaps


######### Functions ##########

### Extract temporal information from the filename
def extract_date_from_filename(filename, data_belong_to):

    # Extract the filename without the path or extension
    basename = os.path.basename(filename)
    filename_without_ext = os.path.splitext(basename)[0]

    if data_belong_to == 'climate':

        date_str = filename_without_ext[-8:]

        # Validate the date format
        if not date_str.isdigit() or len(date_str) != 8:
            raise ValueError(
                f"The last 8 characters of file {filename} "
                f"are not a valid date (expected YYYYMMDD)."
            )

        # Convert the date string to a datetime object
        return datetime.datetime.strptime(date_str, "%Y%m%d")

    elif data_belong_to == 'pheno':

        date_str = filename_without_ext[-4:]

        # Validate the year format
        if not date_str.isdigit() or len(date_str) != 4:
            raise ValueError(
                f"The last 4 characters of file {filename} "
                f"are not a valid year (expected YYYY)."
            )

        # Convert the year string to a datetime object
        return datetime.datetime.strptime(date_str, "%Y")


### Read the first raster band
def get_band(tif_file):

    tif = gdal.Open(tif_file)

    data = (
        tif.GetRasterBand(1)
        .ReadAsArray()
        .astype(np.float32)
    )

    # Release the raster resource
    tif = None

    return data


def aggregate_function(
    data,
    aggregation_factor,
    rows,
    cols,
    new_rows,
    new_cols
):

    aggregate_data = np.full(
        (new_rows, new_cols),
        np.nan,
        dtype=np.float32
    )

    row_indices = np.repeat(
        np.arange(new_rows),
        new_cols
    )

    col_indices = np.tile(
        np.arange(new_cols),
        new_rows
    )

    def aggregate_paralle(data, i, j):

        row_start = i * aggregation_factor
        row_end = min(
            (i + 1) * aggregation_factor,
            rows
        )

        col_start = j * aggregation_factor
        col_end = min(
            (j + 1) * aggregation_factor,
            cols
        )

        window = data[
            row_start:row_end,
            col_start:col_end
        ]

        # Select valid values and exclude predefined invalid values
        valid_values = window[
            np.isfinite(window) &
            (window != 0) &
            (window != -9999) &
            (window != 90) &
            (window != 300)
        ]

        if len(valid_values) > 0:
            result = np.nanmean(valid_values)

        if len(valid_values) == 0:
            result = np.nan

        return i, j, result

    results = Parallel(n_jobs=18)(
        delayed(aggregate_paralle)(
            data,
            i,
            j
        )
        for i, j in zip(row_indices, col_indices)
    )

    for i, j, result in results:
        aggregate_data[i, j] = result

    return aggregate_data


def clip_by_latitude(gt, rows, lat_min, lat_max):
    """
    Clip the data to the specified latitude range.

    Returns:
        (row_start, row_end, new_gt)
    """

    # Calculate the center latitude of each row
    row_centers = (
        np.arange(rows) * gt[5] +
        gt[3] +
        gt[5] / 2
    )

    # Identify rows within the specified latitude range
    valid_rows = (
        (row_centers >= lat_min) &
        (row_centers <= lat_max)
    )

    if not np.any(valid_rows):
        raise ValueError(
            f"No valid data found within the latitude range "
            f"{lat_min}-{lat_max}°N."
        )

    # Find the first and last valid rows
    valid_row_indices = np.where(valid_rows)[0]

    row_start = valid_row_indices[0]
    row_end = valid_row_indices[-1] + 1  # Slicing uses a half-open interval

    # Calculate the new upper-left coordinates
    new_top_left_x = (
        gt[0] +
        row_start * gt[2]
    )  # Usually gt[2] = 0

    new_top_left_y = (
        gt[3] +
        row_start * gt[5]
    )  # gt[5] is the pixel height

    # Create the updated geotransform parameters
    new_gt = (
        new_top_left_x,
        gt[1],  # Keep the pixel width unchanged
        gt[2],  # Keep the row rotation unchanged
        new_top_left_y,
        gt[4],  # Keep the column rotation unchanged
        gt[5]   # Keep the pixel height unchanged
    )

    # print(f"Latitude range: {lat_min}-{lat_max}°N")
    # print(f"Corresponding row range: {row_start} - {row_end - 1}")
    # print(f"Original number of rows: {rows}, "
    #       f"clipped number of rows: {row_end - row_start}")
    # print(f"Original upper-left latitude: {gt[3]:.2f}°N, "
    #       f"new upper-left latitude: {new_top_left_y:.2f}°N")

    return row_start, row_end, new_gt


def save_tif_gdal(output_path, data, crs, transform):
    """Save a GeoTIFF file with the input dimensions and georeferencing information."""

    rows, cols = data.shape

    driver = gdal.GetDriverByName("GTiff")

    output_ds = driver.Create(
        output_path,
        cols,
        rows,
        1,
        gdal.GDT_Float32
    )

    if not output_ds:
        raise RuntimeError(
            f"Failed to create output file: {output_path}"
        )

    output_band = output_ds.GetRasterBand(1)

    output_band.WriteArray(
        data,
        0,
        0
    )

    output_band.SetNoDataValue(
        np.nan
    )  # Set NaN as the NoData value

    output_ds.SetProjection(crs)

    output_ds.SetGeoTransform(
        transform
    )  # Apply the updated geotransform parameters

    output_ds = None

    return True


######################################
# 1. Data Input and Output Settings
######################################

###### ============== Input/Output Settings to Modify ============== ######

aggregation_factor = 5  ### Aggregation factor
aggregate_size = 11 * aggregation_factor

min_lat_value = 30
max_lat_value = 84

path = fr'D:'

data_belong_to = 'pheno'  ### climate / pheno


## Aggregate climate data
if data_belong_to == 'climate':

    climate_var = 'Srad'
    ## Available variables:
    ## 0-100cmSM / VPD / Ta / Pre / Srad

    folder = (
        fr'{path}\ERA5_Land_NH_11km_daily'
        fr'\ERA5_Land_NH_11km_daily_{climate_var}'
    )

    output_tif_path = (
        fr'{path}\ERA5_Land_NH_{aggregate_size}km_daily'
        fr'\ERA5_Land_NH_{aggregate_size}km_daily_{climate_var}'
    )


## Aggregate phenology data
if data_belong_to == 'pheno':

    pheno = 'POS'

    folder = fr'{path}\{pheno}_11km'

    output_tif_path = fr'{path}\{pheno}_{aggregate_size}km'


#########################

tif_files = sorted(
    glob.glob(
        os.path.join(folder, '*.tif')
    )
)

if not tif_files:
    raise FileNotFoundError(
        "No TIF files were found!"
    )


###########################
# 2. Basic Information
###########################

first_tif = tif_files[0]

sample_tif = gdal.Open(first_tif)

if sample_tif is None:
    raise RuntimeError(
        f"Failed to open TIF file: {first_tif} "
        f"(unsupported driver or corrupted file)."
    )


# Get the georeferencing information
crs = sample_tif.GetProjectionRef()
gt = sample_tif.GetGeoTransform()
proj = sample_tif.GetProjection()


# Get the raster dimensions
sample_array = (
    sample_tif.GetRasterBand(1)
    .ReadAsArray()
    .astype(np.float32)
)

rows = sample_array.shape[0]
cols = sample_array.shape[1]

print(
    'rows =',
    rows,
    'cols =',
    cols
)


# Release the input raster
sample_tif = None


############################################
# 3. Temporal Processing and Aggregation
############################################

### Calculate the dimensions after aggregation

new_rows = rows // aggregation_factor
new_cols = cols // aggregation_factor

# print(
#     f'Aggregated dimensions: '
#     f'{new_rows} x {new_cols}'
# )


# Update the geotransform parameters
new_gt = (
    gt[0],  # Upper-left x-coordinate
    gt[1] * aggregation_factor,  # Pixel width
    gt[2],
    gt[3],  # Upper-left y-coordinate
    gt[4],
    gt[5] * aggregation_factor  # Pixel height
)


### Process and aggregate each TIF file

for tif_file in tif_files:

    ### Extract temporal information
    date = extract_date_from_filename(
        tif_file,
        data_belong_to
    )


    ### Read raster data
    tif_data = get_band(tif_file)


    print("Aggregation started!")


    ### Aggregate the data
    aggregated_data = aggregate_function(
        tif_data,
        aggregation_factor,
        rows,
        cols,
        new_rows,
        new_cols
    )

    print("Aggregation completed!")


    # Get the dimensions of the aggregated data
    # for subsequent processing
    rows_agg = aggregated_data.shape[0]
    cols_agg = aggregated_data.shape[1]

    # print(
    #     f'Aggregated dimensions: '
    #     f'{rows_agg} x {cols_agg}'
    # )


    row_start, row_end, new_gt = clip_by_latitude(
        new_gt,
        rows_agg,
        lat_min=min_lat_value,
        lat_max=max_lat_value
    )

    print(
        f'row_start: {row_start}, '
        f'row_end: {row_end}'
    )


    # Calculate the spatial extent after clipping
    lon_min = new_gt[0]
    lon_max = (
        new_gt[0] +
        new_gt[1] * cols_agg
    )

    lat_max = new_gt[3]

    lat_min = (
        new_gt[3] +
        new_gt[5] * (row_end - row_start)
    )

    print(
        f'Longitude range: {lon_min} to {lon_max}°'
    )

    print(
        f'Latitude range: {lat_min} to {lat_max}°N'
    )


    ############################################
    # 4. Export
    ############################################

    ## Export climate data
    if data_belong_to == 'climate':

        date_str = date.strftime(
            "%Y%m%d"
        )  # Format as YYYYMMDD

        output_path = os.path.join(
            output_tif_path,
            f'NH_{climate_var}_{aggregate_size}km_{date_str}.tif'
        )


    ## Export phenology data
    if data_belong_to == 'pheno':

        date_str = date.strftime(
            "%Y"
        )  # Format as YYYY

        output_path = os.path.join(
            output_tif_path,
            f'{pheno}1_aggMean_{aggregate_size}000m_{date_str}.tif'
        )


    save_tif_gdal(
        output_path,
        aggregated_data[row_start:row_end, :],
        crs,
        new_gt  # Apply the updated geotransform parameters
    )


    if data_belong_to == 'climate':
        print(
            f'{date} {climate_var} export completed!'
        )

    if data_belong_to == 'pheno':
        print(
            f'{date} {pheno} export completed!'
        )


if data_belong_to == 'climate':

    print(
        f'{climate_var} export completed!'
    )

if data_belong_to == 'pheno':

    print(
        f'{pheno} export completed!'
    )