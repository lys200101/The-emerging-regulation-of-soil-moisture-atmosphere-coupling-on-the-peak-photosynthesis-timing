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


######### Functions ##########

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
    Clip the data to the specified latitude range.

    Returns:
        (row_start, row_end, new_gt)
    """

    # Calculate the center latitude of each row
    row_centers = np.arange(rows) * gt[5] + gt[3] + gt[5] / 2

    # Identify rows within the specified latitude range
    valid_rows = (row_centers >= lat_min) & (row_centers <= lat_max)

    if not np.any(valid_rows):
        raise ValueError(
            f"No valid data found within the latitude range "
            f"{lat_min}-{lat_max}°N"
        )

    # Find the first and last valid rows
    valid_row_indices = np.where(valid_rows)[0]
    row_start = valid_row_indices[0]
    row_end = valid_row_indices[-1] + 1  # Slicing uses a half-open interval

    # Calculate the new upper-left coordinates
    new_top_left_x = gt[0] + row_start * gt[2]  # Usually gt[2] = 0
    new_top_left_y = gt[3] + row_start * gt[5]  # gt[5] is the pixel height

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
    # print(f"Original number of rows: {rows}, clipped number of rows: {row_end - row_start}")
    # print(f"Original upper-left latitude: {gt[3]:.2f}°N, "
    #       f"new upper-left latitude: {new_top_left_y:.2f}°N")

    return row_start, row_end, new_gt


def save_tif_gdal(output_path, data, crs, transform):
    """Save a GeoTIFF file with the input dimensions and georeferencing information."""

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
    output_ds.SetGeoTransform(transform)  # Apply the updated geotransform parameters

    output_ds = None
    return True


######################################
# 1. Data Input and Output Settings
######################################

###### ============== Input/Output Settings to Modify ============== ######

aggregation_factor = 5  ### Aggregation factor: 1 / 5
aggregate_size = 11 * aggregation_factor

min_lat_value = 30
max_lat_value = 84

tif = fr'D:\NH_permanent_veg_type_fraction_11km.tif'

fig_output = r'D:\Veg_type'


###########################
# 2. Basic Information
###########################

sample_tif = gdal.Open(tif)

# Get the georeferencing information
crs = sample_tif.GetProjectionRef()
gt = sample_tif.GetGeoTransform()
proj = sample_tif.GetProjection()

# Get the raster dimensions
forest_array = sample_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
rows = forest_array.shape[0]
cols = forest_array.shape[1]

print('rows=', rows, 'cols=', cols)

shrub_array = sample_tif.GetRasterBand(2).ReadAsArray().astype(np.float32)
savanna_array = sample_tif.GetRasterBand(3).ReadAsArray().astype(np.float32)
grass_array = sample_tif.GetRasterBand(4).ReadAsArray().astype(np.float32)
wet_array = sample_tif.GetRasterBand(5).ReadAsArray().astype(np.float32)

# Release the input raster
sample_tif = None


############################################
# 3. Aggregation
############################################

### Aggregation parameter settings

new_rows = rows // aggregation_factor
new_cols = cols // aggregation_factor

# Update the geotransform parameters
new_gt = (
    gt[0],  # Upper-left x-coordinate
    gt[1] * aggregation_factor,  # Pixel width
    gt[2],
    gt[3],  # Upper-left y-coordinate
    gt[4],
    gt[5] * aggregation_factor  # Pixel height
)

aggregated_forest_data = aggregate_function(
    forest_array, aggregation_factor, rows, cols, new_rows, new_cols
)

aggregated_shrub_data = aggregate_function(
    shrub_array, aggregation_factor, rows, cols, new_rows, new_cols
)

aggregated_savanna_data = aggregate_function(
    savanna_array, aggregation_factor, rows, cols, new_rows, new_cols
)

aggregated_grass_data = aggregate_function(
    grass_array, aggregation_factor, rows, cols, new_rows, new_cols
)

aggregated_wet_data = aggregate_function(
    wet_array, aggregation_factor, rows, cols, new_rows, new_cols
)

print("Aggregation completed!")


# Get the dimensions of the aggregated data for subsequent processing
rows_agg, cols_agg = (
    aggregated_forest_data.shape[0],
    aggregated_forest_data.shape[1]
)

print(f'Aggregated dimensions: rows={rows_agg}, cols={cols_agg}')


############################################
# 4. Determine the 55 km Vegetation Type
#    and Identify Valid 11 km Pixels
############################################

def determine_veg_type(args):
    """Determine the dominant vegetation type for each aggregated pixel."""

    i, j = args

    forest_frac = aggregated_forest_data[i, j]
    shrub_frac = aggregated_shrub_data[i, j]
    savanna_frac = aggregated_savanna_data[i, j]
    grass_frac = aggregated_grass_data[i, j]
    wet_frac = aggregated_wet_data[i, j]

    # Handle invalid data
    if (np.isnan(forest_frac) or np.isnan(shrub_frac) or
            np.isnan(savanna_frac) or np.isnan(grass_frac) or
            np.isnan(wet_frac)):
        return i, j, np.nan

    total_frac = (
        forest_frac +
        shrub_frac +
        savanna_frac +
        grass_frac +
        wet_frac
    )

    # Handle zero or negative total fractions
    if total_frac <= 0:
        return i, j, np.nan

    # Calculate the relative proportion of each vegetation type
    forest_ratio = forest_frac / total_frac
    shrub_ratio = shrub_frac / total_frac
    savanna_ratio = savanna_frac / total_frac
    grass_ratio = grass_frac / total_frac
    wet_ratio = wet_frac / total_frac

    # Identify the dominant vegetation type
    veg_ratios = np.array([
        forest_ratio,
        shrub_ratio,
        savanna_ratio,
        grass_ratio,
        wet_ratio
    ])

    max_index = np.argmax(veg_ratios)
    max_value = veg_ratios[max_index]

    # Assign the dominant vegetation type if its fraction exceeds the threshold
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


# Initialize the vegetation type array
veg_type_55km = np.full(
    (rows_agg, cols_agg),
    np.nan
)

# Generate all pixel coordinates
coordinates = [
    (i, j)
    for i in range(rows_agg)
    for j in range(cols_agg)
]

# Process all pixels in parallel
results = Parallel(n_jobs=18)(
    delayed(determine_veg_type)(coord)
    for coord in coordinates
)

# Fill the results into the vegetation type array
for i, j, result in results:
    veg_type_55km[i, j] = result


# Clip the data to the specified latitude range
row_start, row_end, new_gt = clip_by_latitude(
    new_gt,
    rows_agg,
    lat_min=min_lat_value,
    lat_max=max_lat_value
)

print(f'row_start: {row_start}, row_end: {row_end}')


# Calculate the spatial extent after clipping
lon_min = new_gt[0]
lon_max = new_gt[0] + new_gt[1] * cols_agg

lat_max = new_gt[3]
lat_min = new_gt[3] + new_gt[5] * (row_end - row_start)

print(f'Longitude range: {lon_min} to {lon_max}°')
print(f'Latitude range: {lat_min} to {lat_max}°N')


# Apply the latitude-based clipping
veg_type_55km = veg_type_55km[row_start:row_end, :]


### Calculate the proportion of each vegetation type

total_count = np.count_nonzero(np.isfinite(veg_type_55km))
print(f'total_count: {total_count}')

forest = veg_type_55km[veg_type_55km == 1]
shrub = veg_type_55km[veg_type_55km == 2]
savanna = veg_type_55km[veg_type_55km == 3]
grass = veg_type_55km[veg_type_55km == 4]
wet = veg_type_55km[veg_type_55km == 5]

forest_count = np.count_nonzero(np.isfinite(forest))
shrub_count = np.count_nonzero(np.isfinite(shrub))
savanna_count = np.count_nonzero(np.isfinite(savanna))
grass_count = np.count_nonzero(np.isfinite(grass))
wet_count = np.count_nonzero(np.isfinite(wet))

forest_ratio = forest_count / total_count * 100
shrub_ratio = shrub_count / total_count * 100
savanna_ratio = savanna_count / total_count * 100
grass_ratio = grass_count / total_count * 100
wet_ratio = wet_count / total_count * 100


############################################
# 5. Export
############################################

#### Export the vegetation type raster

output_path = os.path.join(
    path,
    rf'NH_veg_type_{aggregate_size}km(Python).tif'
)

save_tif_gdal(
    output_path,
    veg_type_55km,  # Export the vegetation type map
    crs,
    new_gt  # Apply the updated geotransform parameters
)

print('TIF export completed!')