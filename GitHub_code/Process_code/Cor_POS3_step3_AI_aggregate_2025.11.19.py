import os
import sys
from osgeo import gdal
import numpy as np
from joblib import Parallel, delayed


######### Functions ##########

def clip_by_latitude(gt, rows, lat_min, lat_max):
    """
    Clip the data to the specified latitude range.

    Returns:
        (row_start, row_end, new_gt, new_rows)
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
    new_top_left_x = gt[0]
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

    new_rows = row_end - row_start

    # print(f"Latitude range: {lat_min}-{lat_max}°N")
    # print(f"Corresponding row range: {row_start} - {row_end - 1}")
    # print(f"Original number of rows: {rows}, "
    #       f"clipped number of rows: {row_end - row_start}")
    # print(f"Original upper-left latitude: {gt[3]:.2f}°N, "
    #       f"new upper-left latitude: {new_top_left_y:.2f}°N")

    return row_start, row_end, new_gt, new_rows


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

min_lat_value = 30
max_lat_value = 84

scale = 55

## Aggregate climate data
tif = fr'D:\ai_v3_yr.tif'

## Standard phenology data
pheno_tif = fr'D:\FigShare_data\55km\POS_55km\POS1_aggMean_11000m_2001.tif'

####### Output settings #########

tif_output = 'D:\AI'


###########################
# 2. Basic Information
###########################

##### Template raster information #####

template_data = gdal.Open(pheno_tif)

# Get the georeferencing information
crs_template = template_data.GetProjectionRef()
gt_template = template_data.GetGeoTransform()
proj_template = template_data.GetProjection()

# Get the raster dimensions
template_array = (
    template_data.GetRasterBand(1)
    .ReadAsArray()
    .astype(np.float32)
)

template_rows, template_cols = template_array.shape

print(
    f"template_rows = {template_rows}   "
    f"template_cols = {template_cols}"
)


####### Source data information #########

ai_tif = gdal.Open(tif)

# Get the georeferencing information
ai_crs = ai_tif.GetProjectionRef()
ai_gt = ai_tif.GetGeoTransform()
ai_proj = ai_tif.GetProjection()

# Get the raster dimensions
ai_array = (
    ai_tif.GetRasterBand(1)
    .ReadAsArray()
    .astype(np.float32)
)

ai_rows, ai_cols = ai_array.shape

print('ai_rows =', ai_rows, 'ai_cols =', ai_cols)


# Release the input raster
sample_tif = None


############################################
# 3. Aggregation
############################################

### Calculate the aggregation factor

aggregation_factor = int(
    round(gt_template[1] / ai_gt[1])
)  # The x- and y-resolution are assumed to be identical


# Calculate the starting row in the source data
# corresponding to the latitude range of the template
ai_start_row = int(
    round((gt_template[3] - ai_gt[3]) / ai_gt[5])
)

ai_end_row = int(
    round(
        (
            gt_template[3] +
            gt_template[5] * template_rows -
            ai_gt[3]
        ) / ai_gt[5]
    )
)


row_indices = np.repeat(
    np.arange(template_rows),
    template_cols
)

col_indices = np.tile(
    np.arange(template_cols),
    template_rows
)


aggregated_ai_data = np.full(
    (template_rows, template_cols),
    np.nan
)


def ai_climate_region_aggregate(data, i, j):

    if np.count_nonzero(np.isfinite(data)) == 0:
        return np.nan, i, j

    else:
        ai_agg_value = np.nanmean(data)
        return ai_agg_value, i, j


# Replace zero values with NaN
ai_array = np.where(
    (ai_array != 0),
    ai_array,
    np.nan
)


results = Parallel(
    n_jobs=18,
    verbose=10
)(
    delayed(ai_climate_region_aggregate)(
        ai_array[
            ai_start_row + i * aggregation_factor:
            min(
                ai_start_row + (i + 1) * aggregation_factor,
                ai_rows
            ),
            j * aggregation_factor:
            min(
                (j + 1) * aggregation_factor,
                ai_cols
            )
        ],
        i,
        j
    )
    for i, j in zip(row_indices, col_indices)
)


for ai_agg_value, i, j in results:
    aggregated_ai_data[i, j] = ai_agg_value


########################################
# 4. Clipping
########################################

row_start, row_end, clipped_gt, clipped_rows = clip_by_latitude(
    gt_template,
    template_rows,
    lat_min=min_lat_value,
    lat_max=max_lat_value
)

print(
    f'row_start: {row_start}, '
    f'row_end: {row_end}, '
    f'clipped_rows: {clipped_rows}'
)


aggregated_ai_data = aggregated_ai_data[
    row_start:row_end,
    :
]


# Calculate the spatial extent after clipping
lon_min = clipped_gt[0]
lon_max = clipped_gt[0] + clipped_gt[1] * template_cols

lat_max = clipped_gt[3]
lat_min = clipped_gt[3] + clipped_gt[5] * clipped_rows

print(f'Longitude range: {lon_min} to {lon_max}°')
print(f'Latitude range: {lat_min} to {lat_max}°N')


############################################
# 5. Export
############################################

output_path = os.path.join(
    tif_output,
    f'NH30_84_AI_{scale}km.tif'
)

save_tif_gdal(
    output_path,
    aggregated_ai_data,  # Export the aggregated AI data
    crs_template,
    gt_template  # Apply the geotransform parameters
)

print('TIF export completed!')


############################################
# Plot and Export
############################################

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
import matplotlib.gridspec as gridspec


##############################################
# Plot
##############################################

conditions = [
    (aggregated_ai_data >= 0) &
    (aggregated_ai_data <= 300),       # 0-0.03 * 10000

    (aggregated_ai_data > 300) &
    (aggregated_ai_data <= 2000),      # 0.03-0.2 * 10000

    (aggregated_ai_data > 2000) &
    (aggregated_ai_data <= 3500),      # 0.2-0.35 * 10000

    (aggregated_ai_data > 3500) &
    (aggregated_ai_data <= 5000),      # 0.35-0.5 * 10000

    (aggregated_ai_data > 5000) &
    (aggregated_ai_data <= 6500),      # 0.5-0.65 * 10000

    (aggregated_ai_data > 6500)        # >0.65 * 10000
]

values = [1, 2, 3, 4, 5, 6]


# Reclassify the AI data
aggregated_ai_data_simple = np.select(
    conditions,
    values,
    default=np.nan
)

print('Reclassification completed!')


output_path = os.path.join(
    tif_output,
    f'NH30_84_AI(gradient)_{scale}km.tif'
)


save_tif_gdal(
    output_path,
    aggregated_ai_data_simple,  # Export the reclassified AI data
    crs_template,
    gt_template  # Apply the geotransform parameters
)

print('Reclassified TIF export completed!')