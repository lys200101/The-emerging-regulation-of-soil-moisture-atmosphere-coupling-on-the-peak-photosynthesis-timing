
import os
import glob
import datetime
import sys

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
from duckdb.experimental.spark.sql.functions import isnan
from matplotlib.pyplot import subplot
from mpl_toolkits.basemap import Basemap
from osgeo import gdal
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
import pingouin as pg
from scipy.stats import pearsonr, alpha
from joblib import Parallel, delayed, parallel_backend
import matplotlib as mpl
from matplotlib import colormaps
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.model_selection import train_test_split
import pymannkendall as mk
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import shap
from sklearn.metrics import r2_score
import gc
from sklearn.preprocessing import StandardScaler
from joblib import parallel_backend

import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

######### Function #########
### Read SM and VPD bands
def get_SPEIband(tif, stack, spei_length):
    tif_data = gdal.Open(tif)
    tif_array = tif_data.GetRasterBand(spei_length).ReadAsArray().astype(np.float32)
    stack.append(tif_array)

def get_band(tif_file, stack):
    tif = gdal.Open(tif_file)
    data = tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
    stack.append(data)


def save_tif_gdal(output_path, data, rows, cols, crs, transform):

        driver = gdal.GetDriverByName("GTiff")

        # Create output dataset (overwrite mode)
        output_ds = driver.Create(
            output_path,
            cols,  # Width (number of columns)
            rows,  # Height (number of rows)
            1,  # Number of bands
            gdal.GDT_Float32  # Default data type (can be modified according to requirements)
        )
        if not output_ds:
            raise RuntimeError(f"Unable to create output file: {output_path}")

        output_band = output_ds.GetRasterBand(1)  # Single band index is 1
        output_band.WriteArray(data, 0, 0)  # Write data (0,0 indicates top-left starting corner)

        output_ds.SetProjection(crs)

        output_ds.SetGeoTransform(transform)

        output_ds = None  # Close dataset (Required! Otherwise the file may be corrupted)
        return True


def cal_partial(i, j, vars, y, pos_stack, cor_matrix, sm_matrix, vpd_matrix, ta_matrix, pre_matrix, srad_matrix):
# def cal_partial(i, j, vars, y, pos_stack, cor_matrix, sm_matrix, vpd_matrix):

    # mask = (~np.isnan(pos_stack[:, i, j]) &
    #         ~np.isnan(cor_matrix[:, i, j]) &
    #         ~np.isnan(sm_matrix[:, i, j]) &
    #         ~np.isnan(vpd_matrix[:, i, j]) &
    #         ~np.isnan(ta_matrix[:, i, j]) &
    #         ~np.isnan(pre_matrix[:, i, j]) &
    #         ~np.isnan(srad_matrix[:, i, j]))\

    all_vars = np.stack([
        pos_stack[:, i, j],
        cor_matrix[:, i, j],
        sm_matrix[:, i, j],
        vpd_matrix[:, i, j],
        ta_matrix[:, i, j],
        pre_matrix[:, i, j],
        srad_matrix[:, i, j]
    ], axis=0)
    # mask: time points where all variables are not NaN
    mask = np.all(np.isfinite(all_vars), axis=0)   # If ~ is added before np, invalid values are returned

    if mask.sum() > 3: # Count number of valid values

        pos = np.array(pos_stack[:, i, j][mask])

        cor_time_series = np.array(cor_matrix[:, i, j][mask])
        sm_time_series = np.array(sm_matrix[:, i, j][mask])
        vpd_time_series = np.array(vpd_matrix[:, i, j][mask])
        ta_time_series = np.array(ta_matrix[:, i, j][mask])
        pre_time_series = np.array(pre_matrix[:, i, j][mask])
        srad_time_series = np.array(srad_matrix[:, i, j][mask])

        pixel_data = pd.DataFrame({
        'POS': pos,
        'cor': cor_time_series,   # cor
        'sm': sm_time_series,     # sm
        'vpd': vpd_time_series,   # vpd
        # })
        'ta': ta_time_series,     # ta
        'pre': pre_time_series,   # pre
        'srad': srad_time_series  # srad
        })

        # print('pixel_data:\n', pixel_data)

        # partial_coefficient[i, j] = partial_cor['r'].iloc[0].round(4)
        # partial_coefficient_p[i, j] = partial_cor['p-val'].iloc[0].round(4)

        res_list = []
        for var in vars:
            # Perform partial correlation for each variable with other variables
            res = pg.partial_corr(pixel_data, y=y, x=var, covar=[item for item in vars if item != var])
            res['var'] = var
            # print('res:\n', res)
            res_list.append(res)
        result_df = pd.concat(res_list, axis=0)
        # print('result_df:\n', result_df)

        # Sort and deduplicate: sort by absolute value of correlation coefficient from largest to smallest
        # top_vars = result_df.reindex(result_df['r'].abs().sort_values(ascending=False).index)
        top_vars = result_df.sort_values(by='r', key=lambda x: x.abs(), ascending=False)
        # print('top_vars:', top_vars)

        # Get the first variable (use drop_duplicates to ensure no duplicates)
        top_vars_unique = top_vars.drop_duplicates(subset='var').head(1)

        # Display top 3 variables
        # print("Top 3 strongest partial correlations:")
        # print(top_vars_unique[['var', 'r', 'p-val']])

        varname = top_vars_unique['var'][0]
        # print('top var:', varname)
        r = top_vars_unique['r'][0]
        p = top_vars_unique['p-val'][0]

        sm_pcor = result_df.loc[result_df['var'] == 'sm', 'r'].iloc[0]
        vpd_pcor = result_df.loc[result_df['var'] == 'vpd', 'r'].iloc[0]
        cor_pcor = result_df.loc[result_df['var'] == 'cor', 'r'].iloc[0]
        ta_pcor = result_df.loc[result_df['var'] == 'ta', 'r'].iloc[0]
        pre_pcor = result_df.loc[result_df['var'] == 'pre', 'r'].iloc[0]
        srad_pcor = result_df.loc[result_df['var'] == 'srad', 'r'].iloc[0]


    else:
        varname = np.nan
        r = np.nan
        p = np.nan

        sm_pcor = np.nan
        vpd_pcor = np.nan
        cor_pcor = np.nan
        ta_pcor = np.nan
        pre_pcor = np.nan
        srad_pcor = np.nan

    return (i, j, varname, r, p, sm_pcor, vpd_pcor, cor_pcor, ta_pcor, pre_pcor, srad_pcor)



###################################### 1 Data Reading and Output Settings ################################################
###########################  ==== Input Settings ==== #################################
star_year = 2001
end_year = 2024
years_length = end_year - star_year + 1
print('years_length:', years_length)

y_list = ['POS drought normal diff']#, 'POS slope', 'POS std']  'POS drought normal diff' / 'POS wet normal diff' / 'POS corhigh corlow diff'

months_before_pos = 2

climate_test_number = spei_length = months_before_pos

drought_distinguish_way = 3

pheno = 'pos'

Basedon = 'Based_on_detrendPheno'  ### Based_on_detrendPheno means using detrended SOS and POS for partial correlation
                                   ### Based_on_OriginPheno means using original SOS and POS for partial correlation

Outlier = 'No'
SigCorPvalue = 'No'
analyze_by = 'All'  ## 'All'  /  'advance PPT'  /  'delay PPT'

drought_or_wet_times = 2  ### !!! Key modification
spei_strength = -1

veg_type = 'All'     ### Forest / Shrub / Savanna / Grass
ai_type = 'All'      ### Arid / Semi-Arid / Dry sub-humid / Humid
cor_type = 'All'     ### Cor(-0.6~-0.5) / Cor(-0.5~-0.4) / Cor(-0.4~-0.3) / Cor(-0.3~-0.2)

ML_model_in_spatio = 'RF'  ## RF / XGBoost
test_size = 0.1

scale = 11

#### Input tif path for Climate
input_same_path = rf'D:\FigShare_data'

folder_cor = fr'{input_same_path}\{scale}km\Correlation(SM_VPD_pearson){cor_test_number}'  # (POS-30) - POS

folder_cor_pvalue = fr'{input_same_path}\{scale}km\Correlation(SM_VPD_pearson){cor_test_number}\Pvalue'  # (POS-30) - POS

folder_sm = fr'{input_same_path}\{scale}km\SM_preseason_mean{climate_test_number}'  # (POS-30) - POS

folder_vpd = fr'{input_same_path}\{scale}km\VPD_preseason_mean{climate_test_number}'  # (POS-30) - POS

folder_ta = fr'{input_same_path}\{scale}km\Ta_preseason_mean{climate_test_number}'

folder_pre = fr'{input_same_path}\{scale}km\Pre_preseason_sum{climate_test_number}'

folder_srad = fr'{input_same_path}\{scale}km\Srad_preseason_sum{climate_test_number}'

#### Input tif path for POS
pos_origin_folder = fr'{input_same_path}\{scale}km\POS_55km'  # start
sos_origin_folder = fr'{input_same_path}\{scale}km\SOS_55km'  # start

pos_detrend_folder = fr'{input_same_path}\{scale}km\POSdetrend_55km'  # start
sos_detrend_folder = fr'{input_same_path}\{scale}km\SOSdetrend_55km'  # start


#### Input tif path for drought years identified by SPEI
spei_strength_input_path = rf'{input_same_path}\{scale}km\NH_SPEI{spei_length}_{spei_length}monthBeforePOS'
drought_path = fr'{input_same_path}\{scale}km\drought_event(POS_SPEI{spei_length}_threshold10%_way3)'

#### Input tif path for AI
ai_tif_file = rf'{input_same_path}\{scale}km\AI\NH30_84_AI(graident)_55km.tif'

#### Input tif path for Vegetation Types
veg_type_file = rf"{input_same_path}\{scale}km\Veg_type\NH_veg_type_55km(Python).tif"

#### Input tif path for Coupling Gradient
cor_mean_file = fr'{input_same_path}\{scale}km\3Cor_mean_slope\mean\SM_VPD_Cor17_8_0\Cor_mean_55km_All.tif'  # SOS - POS


###########################  ==== Output Settings ==== #################################
output_path = fr'D:\Result'

####################################### 2 Data Reading #################################################
tif_files_cor = sorted(glob.glob(os.path.join(folder_cor, '*.tif')))
tif_files_cor_pvalue = sorted(glob.glob(os.path.join(folder_cor_pvalue, '*.tif')))
tif_files_sm = sorted(glob.glob(os.path.join(folder_sm, '*.tif')))
tif_files_vpd = sorted(glob.glob(os.path.join(folder_vpd, '*.tif')))
tif_files_ta = sorted(glob.glob(os.path.join(folder_ta, '*.tif')))
tif_files_pre = sorted(glob.glob(os.path.join(folder_pre, '*.tif')))
tif_files_srad = sorted(glob.glob(os.path.join(folder_srad, '*.tif')))

pos_origin_tif_files = sorted(glob.glob(os.path.join(pos_origin_folder, '*.tif')))
sos_origin_tif_files = sorted(glob.glob(os.path.join(sos_origin_folder, '*.tif')))
pos_detrend_tif_files = sorted(glob.glob(os.path.join(pos_detrend_folder, '*.tif')))
sos_detrend_tif_files = sorted(glob.glob(os.path.join(sos_detrend_folder, '*.tif')))

drought_year_tif_files = sorted(glob.glob(os.path.join(drought_path, '*.tif')))
spei_strength_year_tif_files = sorted(glob.glob(os.path.join(spei_strength_input_path, '*.tif')))



###################################################
if cor_test_number == '(Preseason)':
    first_tif = tif_files_sm_1[0]
else:
    first_tif = tif_files_sm[0]
sample_tif = gdal.Open(first_tif)

if sample_tif is None:
    raise RuntimeError(f"Unable to open TIF file: {sample_tif} (Unsupported driver or corrupted file)")

# Get geotransform parameters: projection, pixel size
# Coordinates and projection    Spatial Reference System: Spatial reference frame of the data
crs = sample_tif.GetProjectionRef()          # Automatically retrieve input CRS
# print('crs:', crs)
gt = sample_tif.GetGeoTransform()  # Geographic coordinates: Longitude & Latitude. Mathematical transformation parameters to convert pixel coordinates into actual geographic coordinates.
proj = sample_tif.GetProjection()  # Projected coordinates: xy (units in m)

# Pixels
pixel_width = gt[1]
pixel_height = gt[5]

top_left_x = gt[0]
top_left_y = gt[3]

# Number of rows and columns
sample_tif = sample_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)

rows = sample_tif.shape[0]
cols = sample_tif.shape[1]
print('rows:', rows, 'cols:', cols)

row_indices = np.repeat(np.arange(rows), cols)  # Repeat row indices 'cols' times
col_indices = np.tile(np.arange(cols), rows)  # Tile column indices 'rows' times

# Calculate longitude and latitude boundaries (correcting for negative pixel_height)
lon_min = top_left_x
lon_max = top_left_x + cols * pixel_width  # Right boundary longitude
lat_min = top_left_y + rows * pixel_height  # Bottom boundary latitude (southernmost point, may be smaller)
lat_max = top_left_y  # Top boundary latitude (northernmost point, may be larger)
print(f"Longitude range: {lon_min:.6f} -> {lon_max:.6f}")
print(f"Latitude range: {lat_min:.6f} -> {lat_max:.6f}")


############################################ 3 Time-Stacking ###################################################

## Data Stacking
cor_stack = []

cor_pvalue_stack = []

sm_stack = []

vpd_stack = []

ta_stack = []

pre_stack = []

srad_stack = []

for tif_file in tif_files_cor:
    get_band(tif_file, cor_stack)

for tif_file in tif_files_cor_pvalue:
    get_band(tif_file, cor_pvalue_stack)

for tif_file in tif_files_sm:
    get_band(tif_file, sm_stack)

for tif_file in tif_files_vpd:
    get_band(tif_file, vpd_stack)

for tif_file in tif_files_ta:
    get_band(tif_file, ta_stack)

for tif_file in tif_files_pre:
    get_band(tif_file, pre_stack)

for tif_file in tif_files_srad:
    get_band(tif_file, srad_stack)


cor_stack = np.stack(cor_stack, axis=0)

cor_pvalue_stack = np.stack(cor_pvalue_stack, axis=0)

sm_stack = np.stack(sm_stack, axis=0)
vpd_stack = np.stack(vpd_stack, axis=0)

ta_stack = np.stack(ta_stack, axis=0)

pre_stack = np.stack(pre_stack, axis=0)

srad_stack = np.stack(srad_stack, axis=0)


pos_origin_stack = []
sos_origin_stack = []

pos_detrend_stack = []
sos_detrend_stack = []

drought_year_stack = []
spei_strength_year_stack = []

for tif_file in pos_origin_tif_files:
    get_band(tif_file, pos_origin_stack)
for tif_file in sos_origin_tif_files:
    get_band(tif_file, sos_origin_stack)

for tif_file in pos_detrend_tif_files:
    get_band(tif_file, pos_detrend_stack)
for tif_file in sos_detrend_tif_files:
    get_band(tif_file, sos_detrend_stack)

for tif_file in drought_year_tif_files:
    get_band(tif_file, drought_year_stack)
for tif_file in spei_strength_year_tif_files:
    get_band(tif_file, spei_strength_year_stack)



ai_tif = gdal.Open(ai_tif_file)
ai_type_data = ai_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
print('AI shape:', ai_type_data.shape)

veg_type_tif = gdal.Open(veg_type_file)
veg_type_data = veg_type_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
print('veg_type_data shape:', veg_type_data.shape)

#### Coupling gradient data
cor_mean_file = gdal.Open(cor_mean_file)
cor_mean_data = cor_mean_file.GetRasterBand(1).ReadAsArray().astype(np.float32)
print('cor_mean_data shape:', cor_mean_data.shape)


print('Stack start')

pos_origin_stack = np.stack(pos_origin_stack, axis=0)
pos_detrend_stack = np.stack(pos_detrend_stack, axis=0)

sos_origin_stack = np.stack(sos_origin_stack, axis=0)
sos_detrend_stack = np.stack(sos_detrend_stack, axis=0)

drought_year_stack = np.stack(drought_year_stack, axis=0)

spei_strength_year_stack = np.stack(spei_strength_year_stack, axis=0)

print('Stack end!')


##### ======== 4 Identify pixels experiencing compound drought events annually and calculate POS changes caused by drought or wet conditions ============ #######
time_length = drought_year_stack.shape[0]

drought_event = np.where(((drought_year_stack==1) & (spei_strength_year_stack <= spei_strength)), drought_year_stack, np.nan)
wet_event = np.where(((drought_year_stack==2) & (spei_strength_year_stack >= -spei_strength)), drought_year_stack, np.nan)

##### 4.1 Count occurrences and number of pixels
drought_event_count = np.nansum(drought_event, axis=0)
wet_event_count = np.nansum(wet_event, axis=0)
print('drought_event_count1:', drought_event_count[26, 501])


#########============== 5 Pixel-by-Pixel Standardization ===============###########
### 5.1 Mask with or without outlier removal
if Outlier == 'Yes':
    def Outlier_array_IQR(x, i, j, qmin, qmax):
        """
        Compute average of the 25-th to 75-th percentile of the data across specified zonal.
        Spatial statistics after the removal of outliers by quantiles
        https://medium.com/@prashant.nair2050/hands-on-outlier-detection-and-treatment-in-python-using-1-5-iqr-rule-f9ff1961a414
        """
        if not type(x) is np.ndarray:
            x = np.asarray(x, dtype=np.float32)
        # x1 = x1[~np.isnan(x)]
        x_flatten = x[np.isfinite(x)]
        if len(x_flatten) < years_length/2:
            # If there is no valid data, return all NaNs directly
            return np.full_like(x, np.nan), i, j
        else:
            # Remove Outliers
            upper_quartile, lower_quartile = np.percentile(x_flatten, [qmax, qmin])
            IQR = (upper_quartile - lower_quartile)

            # from scipy.stats import iqr
            # x1 = x.copy()
            # x1 = np.where(x!=fillvalue, x1, np.nan)
            # IQR = iqr(x1, nan_policy='omit')

            lower_range = lower_quartile - (1.5 * IQR)
            upper_range = upper_quartile + (1.5 * IQR)

            # maxv = np.max(x_flatten)
            # minv = np.min(x_flatten)
            valid_mask = np.logical_and(x <= upper_range, x >= lower_range)
            x_masked = np.where(valid_mask, x, np.nan)

            if (len(np.isfinite(x_masked)) > (years_length/2)) & (len(np.isfinite(x_masked)) <= years_length):
                return x_masked, i, j  # IQR, lower_range, upper_range, minv, maxv
            else:
                return np.full_like(x, np.nan), i, j


    outlier_pos_stack = np.full((years_length, rows, cols), np.nan)

    outlier_sos_stack = np.full((years_length, rows, cols), np.nan)
    outlier_cor_stack = np.full((years_length, rows, cols), np.nan)
    outlier_sm_stack = np.full((years_length, rows, cols), np.nan)
    outlier_vpd_stack = np.full((years_length, rows, cols), np.nan)
    outlier_ta_stack = np.full((years_length, rows, cols), np.nan)
    outlier_pre_stack = np.full((years_length, rows, cols), np.nan)
    outlier_srad_stack = np.full((years_length, rows, cols), np.nan)

    if Basedon == 'Based_on_detrendPheno':
        pos_input_stack = pos_detrend_stack
        sos_input_stack = sos_detrend_stack
    elif Basedon == 'Based_on_originPheno':
        pos_input_stack = pos_origin_stack
        sos_input_stack = sos_origin_stack

    ### IQR removal of POS outliers
    results = Parallel(n_jobs=15, verbose=10)(
        delayed(Outlier_array_IQR)(
            pos_input_stack[:, i, j], i, j, 25, 75
        ) for i, j in zip(row_indices, col_indices)
    )

    for data_mask, i, j in results:
        outlier_pos_stack[:, i, j] = data_mask

    ### IQR removal of SOS outliers
    results = Parallel(n_jobs=15, verbose=10)(
        delayed(Outlier_array_IQR)(
            sos_input_stack[:, i, j], i, j, 25, 75
        ) for i, j in zip(row_indices, col_indices)
    )

    for data_mask, i, j in results:
        outlier_sos_stack[:, i, j] = data_mask

    ### IQR removal of Cor outliers
    results = Parallel(n_jobs=15, verbose=10)(
        delayed(Outlier_array_IQR)(
            cor_stack[:, i, j], i, j, 25, 75
        ) for i, j in zip(row_indices, col_indices)
    )

    for data_mask, i, j in results:
        outlier_cor_stack[:, i, j] = data_mask

    ### IQR removal of SM outliers
    results = Parallel(n_jobs=15, verbose=10)(
        delayed(Outlier_array_IQR)(
            sm_stack[:, i, j], i, j, 25, 75
        ) for i, j in zip(row_indices, col_indices)
    )

    for data_mask, i, j in results:
        outlier_sm_stack[:, i, j] = data_mask

    ### IQR removal of VPD outliers
    results = Parallel(n_jobs=15, verbose=10)(
        delayed(Outlier_array_IQR)(
            vpd_stack[:, i, j], i, j, 25, 75
        ) for i, j in zip(row_indices, col_indices)
    )

    for data_mask, i, j in results:
        outlier_vpd_stack[:, i, j] = data_mask

    ### IQR removal of Ta outliers
    results = Parallel(n_jobs=15, verbose=10)(
        delayed(Outlier_array_IQR)(
            ta_stack[:, i, j], i, j, 25, 75
        ) for i, j in zip(row_indices, col_indices)
    )

    for data_mask, i, j in results:
        outlier_ta_stack[:, i, j] = data_mask

    ### IQR removal of Pre outliers
    results = Parallel(n_jobs=15, verbose=10)(
        delayed(Outlier_array_IQR)(
            pre_stack[:, i, j], i, j, 25, 75
        ) for i, j in zip(row_indices, col_indices)
    )

    for data_mask, i, j in results:
        outlier_pre_stack[:, i, j] = data_mask

    ### IQR removal of Srad outliers
    results = Parallel(n_jobs=15, verbose=10)(
        delayed(Outlier_array_IQR)(
            srad_stack[:, i, j], i, j, 25, 75
        ) for i, j in zip(row_indices, col_indices)
    )

    for data_mask, i, j in results:
        outlier_srad_stack[:, i, j] = data_mask

    ### Mask after outlier removal
    mask = (np.isfinite(outlier_pos_stack) & np.isfinite(outlier_sos_stack) &
            np.isfinite(outlier_cor_stack) &
            np.isfinite(outlier_sm_stack) & np.isfinite(outlier_vpd_stack) &
            np.isfinite(outlier_ta_stack) & np.isfinite(outlier_pre_stack) & np.isfinite(outlier_srad_stack))


elif Outlier == 'No':
    if Basedon == 'Based_on_detrendPheno':
        ## Mask without outlier removal
        mask = (np.isfinite(pos_detrend_stack) & np.isfinite(sos_detrend_stack) &
                np.isfinite(cor_stack) &
                np.isfinite(sm_stack) & np.isfinite(vpd_stack) &
                np.isfinite(ta_stack) & np.isfinite(pre_stack) & np.isfinite(srad_stack))
    elif Basedon == 'Based_on_originPheno':
        ## Mask without outlier removal
        mask = (np.isfinite(pos_origin_stack) & np.isfinite(sos_origin_stack) &
                np.isfinite(cor_stack) &
                np.isfinite(sm_stack) & np.isfinite(vpd_stack) &
                np.isfinite(ta_stack) & np.isfinite(pre_stack) & np.isfinite(srad_stack))

if SigCorPvalue == 'Yes':
    mask = (cor_pvalue_stack <= 0.1) & np.isfinite(cor_pvalue_stack) & mask

vaild_year_length = np.sum(mask, axis=0)

space_mask = vaild_year_length > (years_length / 2)  # Shape: (rows, cols)


### 4.2 Valid mask + compute cor mean
space_mask_3d = space_mask[np.newaxis, :, :]  # (1, rows, cols)
space_mask_3d = np.repeat(space_mask_3d, years_length, axis=0)  # (years_length, rows, cols)

cor_stack_masked = np.where(space_mask_3d, cor_stack, np.nan)

cor_mean = np.nanmean(cor_stack_masked, axis=0)  # Shape: (rows, cols)

### 4.3 Time series standardization
def standardize_data(data_stack, mask_3d):
    """Temporal standardization"""

    def standardize_func(data_pixel, i, j):
        original_nan_mask = np.isnan(data_pixel)

        spatio_mean = np.nanmean(data_pixel)
        spatio_std = np.nanstd(data_pixel)

        standardized_pixel_data = (data_pixel - spatio_mean) / spatio_std

        standardized_pixel_data[original_nan_mask] = np.nan

        return standardized_pixel_data, i, j

    data_stack = np.where(mask_3d, data_stack, np.nan)

    standardized_data = np.full((years_length, rows, cols), np.nan)

    results = Parallel(n_jobs=15, verbose=10)(
        delayed(standardize_func)(
            data_stack[:, i, j],
            i, j
        ) for i, j in zip(row_indices, col_indices)
    )

    for standardized_pixel_data, i, j in results:
        standardized_data[:, i, j] = standardized_pixel_data

    return standardized_data

if Outlier == 'Yes':
    pos_standardized = standardize_data(outlier_pos_stack, space_mask_3d)
    sos_standardized = standardize_data(outlier_sos_stack, space_mask_3d)
    cor_standardized = standardize_data(outlier_cor_stack, space_mask_3d)
    sm_standardized = standardize_data(outlier_sm_stack, space_mask_3d)
    vpd_standardized = standardize_data(outlier_vpd_stack, space_mask_3d)
    ta_standardized = standardize_data(outlier_ta_stack, space_mask_3d)
    pre_standardized = standardize_data(outlier_pre_stack, space_mask_3d)
    srad_standardized = standardize_data(outlier_srad_stack, space_mask_3d)
if Outlier == 'No':
    if Basedon == 'Based_on_detrendPheno':
        pos_standardized = standardize_data(pos_detrend_stack, space_mask_3d)
        sos_standardized = standardize_data(sos_detrend_stack, space_mask_3d)
    elif Basedon == 'Based_on_originPheno':
        pos_standardized = standardize_data(pos_origin_stack, space_mask_3d)
        sos_standardized = standardize_data(sos_origin_stack, space_mask_3d)
    cor_standardized = standardize_data(cor_stack, space_mask_3d)
    sm_standardized = standardize_data(sm_stack, space_mask_3d)
    vpd_standardized = standardize_data(vpd_stack, space_mask_3d)
    ta_standardized = standardize_data(ta_stack, space_mask_3d)
    pre_standardized = standardize_data(pre_stack, space_mask_3d)
    srad_standardized = standardize_data(srad_stack, space_mask_3d)

### 5.3 Compute mean/slope/std for valid pixels ###
def drought_wet_normal_year_diff(data_stack, drought_wet_normal_year_stack):
    ######### First, isolate drought, wet, and normal years respectively ##########
    drought_year_mask = (drought_wet_normal_year_stack == 1)
    wet_year_mask = (drought_wet_normal_year_stack == 2)
    normal_year_mask = (drought_wet_normal_year_stack == 0)

    ######### Calculate means for drought, wet, and normal year data, then calculate differences #########
    data_in_drought_year = np.where(drought_year_mask, data_stack, np.nan)
    data_in_wet_year = np.where(wet_year_mask, data_stack, np.nan)
    data_in_normal_year = np.where(normal_year_mask, data_stack, np.nan)

    data_in_drought_mean = np.nanmean(data_in_drought_year, axis=0)
    data_in_wet_mean = np.nanmean(data_in_wet_year, axis=0)
    data_in_normal_mean = np.nanmean(data_in_normal_year, axis=0)

    data_diff_in_drought_noraml_year = data_in_drought_mean - data_in_normal_mean
    data_diff_in_wet_noraml_year = data_in_wet_mean - data_in_normal_mean

    return data_diff_in_drought_noraml_year, data_diff_in_wet_noraml_year



### POS:
pos_stack = np.where(space_mask_3d, pos_origin_stack, np.nan)
pos_diff_drought_normal_origin, pos_diff_wet_normal_origin = drought_wet_normal_year_diff(pos_stack, drought_year_stack)
pos_diff_drought_normal, pos_diff_wet_normal = drought_wet_normal_year_diff(pos_standardized, drought_year_stack)

print(f'pos_diff_drought_normal valid pixel count: {np.count_nonzero(np.isfinite(pos_diff_drought_normal))}')
print(f'pos_diff_wet_normal valid pixel count: {np.count_nonzero(np.isfinite(pos_diff_wet_normal))}')

### SOS:
sos_diff_drought_normal, sos_diff_wet_normal = drought_wet_normal_year_diff(sos_standardized, drought_year_stack)

### Cor
cor_diff_drought_normal, cor_diff_wet_normal = drought_wet_normal_year_diff(cor_standardized, drought_year_stack)

### SM slope
sm_diff_drought_normal, sm_diff_wet_normal = drought_wet_normal_year_diff(sm_standardized, drought_year_stack)

### VPD slope
vpd_diff_drought_normal, vpd_diff_wet_normal = drought_wet_normal_year_diff(vpd_standardized, drought_year_stack)

### Ta slope
ta_diff_drought_normal, ta_diff_wet_normal = drought_wet_normal_year_diff(ta_standardized, drought_year_stack)

### Pre slope
pre_diff_drought_normal, pre_diff_wet_normal = drought_wet_normal_year_diff(pre_standardized, drought_year_stack)

### Srad slope
srad_diff_drought_normal, srad_diff_wet_normal = drought_wet_normal_year_diff(srad_standardized, drought_year_stack)

del cor_standardized, sm_standardized, vpd_standardized, ta_standardized, pre_standardized, srad_standardized


########################### 6 Preseason Pearson+Partial ################
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches


## a or b
def plot_cor_mean_or_slope_and_pvalue_forAllvegType(plot_data, colorbarmin, colorbarmax, drought_or_wet, name, ax):

    fig = ax.get_figure()
    if name == 'All':
        height = 0.18
        wspace = 0.05
    else:
        height = 0.25
        wspace = 0.15
    gs_inner = ax.get_subplotspec().subgridspec(2, 2,
                                                width_ratios=[5, 0.8],
                                                height_ratios=[5, 0.3],
                                                hspace=height, wspace=wspace)

    # Hide parent ax as it serves only as a placeholder
    ax.axis('off')

    plots = []  # Store plot objects for each subplot

    # Create three actual inner sub-axes
    ax1 = fig.add_subplot(gs_inner[0, 0])  # Map
    ax2 = fig.add_subplot(gs_inner[0, 1])  # Latitudinal profile
    if drought_or_wet == 'drought':
        if name in ['All', 'Grass', 'Humid', 'Cor(lt-0.4)']:
            ax3 = fig.add_subplot(gs_inner[1, :])  # Colorbar

    plot_data_mean = np.nanmean(plot_data)
    valid_mask = np.isfinite(plot_data)
    plot_data_count_sum = np.sum(valid_mask)

    plot_data_gte0 = np.where(plot_data>=0, plot_data, np.nan)
    plot_data_gte0_mean = np.nanmean(plot_data_gte0)
    plot_data_gte0_sum = np.sum(plot_data>=0)

    plot_data_lt0 = np.where(plot_data < 0, plot_data, np.nan)
    plot_data_lt0_mean = np.nanmean(plot_data_lt0)
    plot_data_lt0_sum = np.sum(plot_data < 0)

    advance_percent = (plot_data_lt0_sum / plot_data_count_sum) * 100
    delay_percent = (plot_data_gte0_sum / plot_data_count_sum) * 100

    ########### Subplot 1: Spatial Distribution #################
    ax1.set_box_aspect(1)  # Force map axis aspect ratio to square so its diameter fills grid height
    ax1.axis('off')
    ### Create map
    m = Basemap(ax=ax1,
                projection='npstere',   # North Polar Stereographic Projection
                boundinglat=30,         # Lowest visible latitude (currently 30°N)
                lon_0=0,                # Central meridian (180: Pacific centered; 90: Asia centered)
                resolution='l')

    # Generate grid coordinates
    lons = np.linspace(-180, 180, cols, endpoint=False)
    lats = np.linspace(lat_max , lat_min , rows)
    lons, lats = np.meshgrid(lons, lats)

    # Set latitude and longitude ticks/gridlines
    m.drawparallels(np.arange(30, 90, 30),
                    labels=[0, 0, 0, 0],
                    linewidth=0.5,
                    # fontsize=8,
                    color="black")

    m.drawmeridians(np.arange(0, 360, 60),
                    latmax=90,  # Ensure meridian lines converge at the North Pole
                    labels=[0, 0, 0, 0],  # labels=[left, right, top, bottom] toggles meridian label visibility
                    linewidth=0.5,
                    # fontsize=8,
                    color="gray")

    # # Fill continents
    # m.fillcontinents(color='white', lake_color='white', zorder=1)

    # map_boundary = m.drawmapboundary(linewidth=0)  # Hide boundary line


    ### Plot data
    # Color mapping
    color_list = ['#c51b7d', '#de77ae', '#f1b6da', '#fde0ef',
                  '#e6f5d0', '#b8e186', '#7fbc41', '#4d9221']  # PiYG / 8
    cmap = mpl.colors.ListedColormap(color_list)
    bins = np.linspace(colorbarmin, colorbarmax, 9)
    norm = mpl.colors.BoundaryNorm(bins, cmap.N)

    plot = m.pcolormesh(lons, lats, plot_data, cmap=cmap, norm=norm, shading='nearest', latlon=True,
                        zorder=1)  # Avoid tearing at polar region

    plots.append(plot)  # Save plot object

    ax1.set_frame_on(False)

    ### Draw boundaries
    terrence = r'D:\CAU\Border\NH_terrence\NH_Terrence_30_90'
    m.readshapefile(terrence, 'NH_Terrence', drawbounds=False)
    for shape in m.NH_Terrence:
        # Convert list to numpy array for easier calculation
        points = np.array(shape)
        x, y = points[:, 0], points[:, 1]

        # Core logic: Calculate projected distance between adjacent points
        # If the distance between two consecutive points on the projection plane spikes drastically,
        # it indicates a loop line crossing the center of the polar projection.
        dist = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)

        # Set a threshold (projected coordinates are usually large, e.g., 1e5 scale)
        # If distance between adjacent points exceeds 1/10th of map diameter, classify as an abnormal jump
        threshold = (ax1.get_xlim()[1] - ax1.get_xlim()[0]) * 0.1

        # Find indices of jump points
        break_indices = np.where(dist > threshold)[0]

        if len(break_indices) == 0:
            # No jumps detected; plot the continuous line directly
            ax1.plot(x, y, color='black', linewidth=0.3, zorder=3)
        else:
            # Jumps detected; split line segments and plot individually
            # This removes artifact lines crossing the map center while keeping valid boundaries
            start_idx = 0
            for break_idx in break_indices:
                ax1.plot(x[start_idx:break_idx + 1], y[start_idx:break_idx + 1],
                         color='black', linewidth=0.3, zorder=3)
                start_idx = break_idx + 1
            # Plot the final segment
            ax1.plot(x[start_idx:], y[start_idx:], color='black', linewidth=0.3, zorder=3)

    # x_pole, y_pole = m(0, 90)
    # ax1.scatter(x_pole, y_pole, color='black', s=5, zorder=3)

    ### Outer boundary clipping
    from matplotlib.patches import Circle

    x0, x1 = ax1.get_xlim()
    y0, y1 = ax1.get_ylim()

    center = [(x0 + x1) / 2, (y0 + y1) / 2]
    radius = (x1 - x0) / 2

    clip_circle = Circle(center, radius, transform=ax1.transData)

    for artist in ax1.collections + ax1.lines + ax1.patches:
        artist.set_clip_path(clip_circle)

    boundary_circle = Circle(
        center,
        radius,
        transform=ax1.transData,
        facecolor='none',
        edgecolor='black',  # Color
        linewidth=1,
        clip_on=False,
        zorder=4  # Place on top layer
    )

    ax1.add_patch(boundary_circle)

    # if name == 'All':
    #     if drought_or_wet == 'drought':
    #         ax1.set_title(f'(a)', pad=10, fontweight='bold')
    #     elif drought_or_wet == 'wet':
    #         ax1.set_title(f'(b)', pad=10, fontweight='bold')
    # elif name == 'Forest' or name == 'Arid':
    #     if drought_or_wet == 'drought':
    #         ax1.set_title(f'(a) {name}', pad=10, fontweight='bold')
    #     elif drought_or_wet == 'wet':
    #         ax1.set_title(f'(b)', pad=10, fontweight='bold')
    # elif name == 'Shrub' or name == 'Semi-arid':
    #     if drought_or_wet == 'drought':
    #         ax1.set_title(f'(d) {name}', pad=10, fontweight='bold')
    #     elif drought_or_wet == 'wet':
    #         ax1.set_title(f'(e)', pad=10, fontweight='bold')
    # elif name == 'Savanna' or name == 'Dry sub-humid':
    #     if drought_or_wet == 'drought':
    #         ax1.set_title(f'(g) {name}', pad=10, fontweight='bold')
    #     elif drought_or_wet == 'wet':
    #         ax1.set_title(f'(h)', pad=10, fontweight='bold')
    # elif name == 'Grass' or name == 'Humid':
    #     if drought_or_wet == 'drought':
    #         ax1.set_title(f'(j) {name}', pad=10, fontweight='bold')
    #     elif drought_or_wet == 'wet':
    #         ax1.set_title(f'(k)', pad=10, fontweight='bold')

    # ax1.set_title(f'({word}) {name}', pad=10, fontweight='bold')
    if name == 'All':
        h = 0.24
        v = 0.815
    else:
        h = 0.22
        v = 0.8
    ax1.text(h, v,
             f'Mean = {plot_data_mean:.1f}\n'
             f'Advance = {plot_data_lt0_mean:.1f} ({advance_percent:.1f}%)\n'
             f'Delay = {plot_data_gte0_mean:.1f} ({delay_percent:.1f}%)',
             transform=ax1.transAxes,  # Use relative coordinates for positioning
             multialignment='center',   # Vertical alignment
             fontsize=6)


    ########### Subplot 2: Latitudinal Trend ###########

    # Use actual latitude values as y-axis
    lat_centers = lats[:, 0]

    plot_data_lat = np.nanmean(plot_data, axis=1)

    # if drought_or_wet == 'drought':
    ax2.axvline(x=0, color='gray', linestyle='--', linewidth=1)

    # elif drought_or_wet == 'wet':
    #     ax2.axvline(x=-0.2, color='gray', linestyle='--', linewidth=1)

    ax2.plot(plot_data_lat, lat_centers, color='red', linewidth=1, alpha=0.8)

    ax2.tick_params(axis='both', length=2, pad=3)#,which='major', )

    # if drought_or_wet == 'drought':
    ax2.set_xlim(-6, 6)
    ax2.set_xticks(np.arange(-4, 4.1, 4))
    ax2.set_xticklabels(['-4 ', '0', '4'])  # Manually set tick labels
    # elif drought_or_wet == 'wet':
    #     ax2.set_xlim(-0.5, 0)
    #     ax2.set_xticks(np.arange(-0.4, 0.01, 0.2))
    #     ax2.set_xticklabels(['-0.4', '-0.2', '0'], rotation=45)  # Manually set tick labels

    ax2.set_ylim(30, 90)
    ax2.set_yticks(np.arange(30, 91, 10))
    ax2.set_yticklabels(f'{int(x)}°' for x in np.arange(30, 91, 10))



    ########### Subplot 3: Colorbar ###########
    ### Generate Colorbar (placed at bottom)
    if drought_or_wet == 'drought':
        if name in ['All', 'Grass', 'Humid', 'Cor(lt-0.4)']:
            cbar = fig.colorbar(plots[0], cax=ax3, orientation='horizontal')

            cbar.set_ticks(bins)

            cbar.set_label('PPT difference (days)', labelpad=3)
            cbar.set_ticklabels([f'{int(x)}' for x in bins])


    plt.tight_layout()

    # Get bottom-left corner position of current ax1
    pos1 = ax1.get_position()

    pos2 = ax2.get_position()

    if drought_or_wet == 'drought':
        if name in ['All', 'Grass', 'Humid', 'Cor(lt-0.4)']:
            pos3 = ax3.get_position()

    # Reposition ax1
    if name == 'All':
        xpos = 0.06
        ax1.set_position([
            pos1.x0 - xpos,  # Keep left side fixed offset
            pos2.y0,  # Align bottom with ax2
            pos2.height,
            pos2.height
        ])  # [left, bottom, width, height]
    else:
        xpos = 0.02
        ax1.set_position([
            pos1.x0 - xpos,  # Shift left only
            pos1.y0 ,  # Maintain original bottom
            pos1.width,  # Maintain original width
            pos1.height  # Maintain original height
        ])

    pos1_new = ax1.get_position()
    pos2_new = ax2.get_position()
    # print(f'ax1 height = {pos1_new.height}\n'
    #       f'ax1 width = {pos1_new.width}\n'
    #       f'ax2 height = {pos2_new.height}\n')
    if drought_or_wet == 'drought':
        if name in ['All', 'Grass', 'Humid', 'Cor(lt-0.4)']:
            pos1_new = ax1.get_position()
            ax3.set_position([
                pos1_new.x0+0.11,
                pos3.y0-0.01,
                pos2.x1 - pos1_new.x0,
                pos3.height])

    # plt.tight_layout()

    # plt.show()


## c
def plot_cor_mean_or_slope_lt0_sig_and_lt0_nosig_and_gt0_forAllvegType(drought_data, wet_data, colorbarmin, colorbarmax, name, ax):

    bins = np.arange(colorbarmin, colorbarmax+1, 1)
    count_drought_data, _ = np.histogram(drought_data, bins=bins)
    count_wet_data, _ = np.histogram(wet_data, bins=bins)

    ### Plot
    fig = ax.get_figure()
    gs_inner = ax.get_subplotspec().subgridspec(
        2, 1,
        height_ratios=[5, 0.3],
        hspace=0.15
    )

    ax1 = fig.add_subplot(gs_inner[0])

    ax.axis('off')

    # Set broken/truncated y-axis
    # if data_type == 'cor slope':
    #     ax.axis('off')
    #     bax = brokenaxes(
    #         ylims=((0, 5000), (10000, 11000)),
    #         hspace=0.1,
    #         height_ratios=[1, 5],  # Upper plot gets 1 share, lower plot gets 5 shares
    #         subplot_spec=sub_gs
    #     )
    #
    #     # Set bar positions
    #     bin_centers = (bins[:-1] + bins[1:]) / 2
    #     print(f'bin_centers:{bin_centers}')
    #
    #     total_width = 0.007  # Total width allocated to bars within one tick unit
    #     n = 2  # Number of categories
    #     width = total_width / n  # Single bar width
    #
    #     bax.bar(bin_centers - width / 2, count_lt0_sig, width=width, color='#b2182b', label='Cor sig-strong')
    #     bax.bar(bin_centers + width / 2, count_lt0_nosig, width=width, color='#fddbc7',
    #             label='Cor nonsig-strong')
    #     bax.bar(bin_centers - width / 2, count_gte0_nosig, width=width, color='#d1e5f0',
    #             label='Cor nonsig-weaken')
    #     bax.bar(bin_centers + width / 2, count_gte0_sig, width=width, color='#2166ac', label='Cor sig-weaken')
    #
    #
    #     ticks = np.arange(colorbarmin, colorbarmax + 0.0001, 0.007)
    #     labels = [f'{(c * 10):.2f}' for c in ticks]
    #
    #     bax.set_xlim(colorbarmin, colorbarmax)
    #
    #     for ax_part in bax.axs:
    #         ax_part.set_xticks(ticks)
    #         ax_part.set_xticklabels(labels)
    #
    #     bax.set_xlabel('VPD-SM coupling trend (per decade)', labelpad=20) # Distance between label and ticks
    #
    #     bax.set_ylabel('Frequency', labelpad=31)  # Distance between label and ticks

    #     # Colorbar
    #     bax.legend(
    #         loc='upper right',
    #         bbox_to_anchor=(1.2, 1),
    #         ncol=1,
    #         frameon=False,  # Toggle legend frame visibility
    #         handlelength=1,
    #         handleheight=1
    #     )


    # Do not set broken y-axis
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    # Set bar positions
    bin_centers = (bins[:-1] + bins[1:]) / 2
    print(f'bin_centers:{bin_centers}')

    total_width = 1  # Total width allocated to bars within one tick unit
    n = 2  # Number of categories
    width = total_width / n  # Single bar width

    ax1.bar(bin_centers - width / 2, count_drought_data, width=width, color='#c51b7d', alpha = 0.5, label='drought - normal')
    ax1.bar(bin_centers + width / 2, count_wet_data, width=width, color='#4d9221', alpha = 0.5, label='wet - normal')

    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)

    ax1.tick_params(axis='both', length=2, pad=3)#,which='major', )

    ticks = np.arange(colorbarmin, colorbarmax + 0.0001, 3)
    ax1.set_xlim(colorbarmin, colorbarmax)
    ax1.set_xticks(ticks)
    ax1.tick_params(axis='x', labelsize=9)

    ax1.set_xlabel('PPT difference (days)', labelpad=5) # Distance between label and ticks

    if  name == 'All':
        ax1.set_ylim(0, 2000)
        ax1.set_yticks(np.arange(0, 2000.1, 500))
        ax1.set_yticklabels(f'{int(x * 0.01)}' for x in np.arange(0, 2000.1, 500))
    elif name in ['Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']:
        ax1.set_ylim(0, 1000)
        ax1.set_yticks(np.arange(0, 1000.1, 200))
        ax1.set_yticklabels(f'{int(x * 0.01)}' for x in np.arange(0, 1000.1, 200))
    else:
        ax1.set_ylim(0, 700)
        ax1.set_yticks(np.arange(0, 700.1, 200))
        ax1.set_yticklabels(f'{int(x * 0.01)}' for x in np.arange(0, 700.1, 200))

    if name == 'All':
        x_pos = -0.15
        y_pos = 1.15
    else:
        x_pos = -0.15
        y_pos = 1.12

    ax1.text(
        x_pos,  # x = tick position (data coordinates)
        y_pos,  # y = slightly below (axes coordinates)
        r'$×10^{2}$',  # Scientific notation multiplier
        transform=ax1.transAxes,
        ha='left',  # Left-aligned to prevent truncation
        va='top',
        rotation=0,
        clip_on=False
    )


    # elif name in ['Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']:
    #     ax.set_ylim(0, 2500)
    #     ax.set_yticks(np.arange(0, 2500.1, 500))


    # Colorbar / Legend
    if name == 'All':
        c = 0.5
    else:
        c = 0.45
    if name in ['All', 'Grass', 'Humid', 'Cor(lt-0.4)']:
        ax1.legend(
            loc='lower center',
            bbox_to_anchor=(c, -0.4),
            ncol=2,
            columnspacing=0.4,
            frameon=False,  # Toggle legend frame visibility
            handlelength=0.8,
            handleheight=0.8,
            fontsize = 8
        )

    ax1.set_ylabel('Frequency', labelpad=3)  # Distance between label and ticks

    # if name == 'Forest' or name == 'Arid' or name == 'All':
    #     word = 'c'
    # elif name == 'Shrub' or name == 'Semi-arid':
    #     word = 'f'
    # elif name == 'Savanna' or name == 'Dry sub-humid':
    #     word = 'i'
    # elif name == 'Grass' or name == 'Humid':
    #     word = 'l'
    # ax.set_title(f'({word})', pad=10, fontweight='bold')

    # elif data_type == 'cor mean':
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #
    #     total_width = 0.2  # Total width allocated to bars within one tick unit
    #     n = 2  # Number of categories
    #     width = total_width / n  # Single bar width
    #
    #     bin_centers = (bins[:-1] + bins[1:]) / 2
    #
    #     color_list = ['#a50f15', '#de2d26', '#fb6a4a', '#fc9272',
    #                   '#fcbba1', '#fee5d9', '#9ecae1']
    #
    #     # Plot bars individually to ensure strict color mapping
    #     for j in range(len(count_mean)):
    #         ax.bar(bin_centers[j], count_mean[j], width=width,
    #                 color=color_list[j], edgecolor='none')
    #
    #     ax.set_xlim(colorbarmin, colorbarmax)
    #     ax.set_xticks(np.arange(colorbarmin, colorbarmax, 0.2))
    #
    #     ax.set_xlabel('VPD-SM coupling', labelpad=3) # Distance between label and ticks
    #     if  name in ['Forest', 'Shrub', 'Savanna', 'Grass']:
    #         ax.set_ylim(0, 2500)
    #         ax.set_yticks(np.arange(0, 2500.1, 500))
    #     elif name in ['Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']:
    #         ax.set_ylim(0, 4000)
    #         ax.set_yticks(np.arange(0, 4000.1, 1000))
    #
    #     # ax.axs[0].set_yticks([5000, 6000, 7500])
    #     # ax.axs[1].set_yticks([0, 1000, 2000, 3500])
    #
    #     ax.set_ylabel('Frequency', labelpad=3)  # Distance between label and ticks

    # plt.show()

## d & e & S26-28: Driving factors of Drought/Wet and normal diff
def Partial_ML_calculate_and_plot(pos_data, sos_data, cor_data, sm_data, vpd_data, ta_data, pre_data, srad_data, grade_by, veg_type, ax):

    var_list = ['POS mean', 'POS slope', 'POS std', 'POS drought normal diff', 'POS wet normal diff',
                'POS corhigh corlow diff',

                'SOS mean', 'SOS slope', 'SOS std', 'SOS drought normal diff', 'SOS wet normal diff',
                'SOS corhigh corlow diff',

                'Cor mean', 'Cor slope', 'Cor std', 'Cor drought normal diff', 'Cor wet normal diff',
                'Cor corhigh corlow diff',

                'SM mean', 'SM slope', 'SM std', 'SM drought normal diff', 'SM wet normal diff',
                'SM corhigh corlow diff',

                'VPD mean', 'VPD slope', 'VPD std', 'VPD drought normal diff', 'VPD wet normal diff',
                'VPD corhigh corlow diff',

                'Ta mean', 'Ta slope', 'Ta std', 'Ta drought normal diff', 'Ta wet normal diff',
                'Ta corhigh corlow diff',

                'Pre mean', 'Pre slope', 'Pre std', 'Pre drought normal diff', 'Pre wet normal diff',
                'Pre corhigh corlow diff',

                'Srad mean', 'Srad slope', 'Srad std', 'Srad drought normal diff', 'Srad wet normal diff',
                'Srad corhigh corlow diff']  ## Modifying the order here changes the order of Subplot 1

    if grade_by != 'All':
        plt.rcParams.update({
            'font.family': 'Arial',

            'mathtext.fontset': 'custom',

            'mathtext.rm': 'Arial',  # Regular
            'mathtext.it': 'Arial:italic',  # Italic
            'mathtext.bf': 'Arial:bold',  # Bold

            # Optional (recommended)
            'mathtext.default': 'regular',  # Prevent automatic italicization

            'font.size': 10,
            'axes.titlesize': 10,
            'axes.labelsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            # 'text.usetex': False,  # Do not use external LaTeX
        })

    ### Classify vegetation types / AI gradients
    var_mask = (np.isfinite(pos_data) & np.isfinite(sos_data) &
                np.isfinite(cor_data) & np.isfinite(sm_data) & np.isfinite(vpd_data) &
                np.isfinite(ta_data) & np.isfinite(pre_data) & np.isfinite(srad_data)
                )

    if grade_by == 'All':
        types = ['All']
        codes = [1]

        fig = ax.figure
        gs_inner = ax.get_subplotspec().subgridspec(1, 4,
                               width_ratios=[1, 1, 1, 0.1],  # Width ratios for the three columns
                               height_ratios=[1],  # Reserved for colorbar
                               hspace=0.3, wspace=0.2)

        ax.axis('off')

    elif grade_by == 'Veg':
        if veg_type == 'All':
            types = ['Forest', 'Shrub', 'Savanna', 'Grass']
            codes = [1, 2, 3, 4]

            fig = plt.figure(figsize=(8.2, 14))
            gs = gridspec.GridSpec(4, 4,
                                   width_ratios=[1, 1, 1, 0.1],  # Width ratios for the three columns
                                   height_ratios=[1, 1, 1, 1],  # Reserved for colorbar
                                   hspace=0.3, wspace=0.2)
        elif veg_type == 'Grass':
            types = ['Grass']
            codes = [4]

            fig = ax.figure
            gs_inner = ax.get_subplotspec().subgridspec(1, 4,
                                                        width_ratios=[1, 1, 1, 0.1],  # Width ratios for the three columns
                                                        height_ratios=[1],  # Reserved for colorbar
                                                        hspace=0.3, wspace=0.2)
            ax.axis('off')

    elif grade_by == 'AI':
        types = ['Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']
        codes = [2, 3, 5, 6]

        fig = plt.figure(figsize=(8.2, 14))
        gs = gridspec.GridSpec(4, 4,
                               width_ratios=[1, 1, 1, 0.1],  # Width ratios for the three columns
                               height_ratios=[1, 1, 1, 1],  # Reserved for colorbar
                               hspace=0.3, wspace=0.2)

    elif grade_by == 'Cor mean':
        types = ['Cor(-0.1~0)', 'Cor(-0.2~-0.1)', 'Cor(-0.3~-0.2)',
                 'Cor(-0.4~-0.3)', 'Cor(lt-0.4)']
        codes = [0, 1, 2, 3, 4]

        fig = plt.figure(figsize=(8.2, 15.5))
        gs = gridspec.GridSpec(5, 4,
                               width_ratios=[1, 1, 1, 0.1],  # Width ratios for the three columns
                               height_ratios=[1, 1, 1, 1, 1],  # Reserved for colorbar
                               hspace=0.3, wspace=0.2)



    for i, (type, code) in enumerate(zip(types, codes)):
        if grade_by == 'All':
            ax1 = fig.add_subplot(gs_inner[0, 0])
            ax2 = fig.add_subplot(gs_inner[0, 1])
            ax3 = fig.add_subplot(gs_inner[0, 2])
            ax4 = fig.add_subplot(gs_inner[0, 3])
        elif grade_by == 'Veg' and veg_type == 'Grass':
            ax1 = fig.add_subplot(gs_inner[0, 0])
            ax2 = fig.add_subplot(gs_inner[0, 1])
            ax3 = fig.add_subplot(gs_inner[0, 2])
            ax4 = fig.add_subplot(gs_inner[0, 3])
        else:
            ax1 = plt.subplot(gs[i, 0])
            ax2 = plt.subplot(gs[i, 1])
            ax3 = plt.subplot(gs[i, 2])
            ax4 = plt.subplot(gs[i, 3])

        if grade_by == 'Veg':
            mask = (veg_type_data == code) & var_mask

        elif grade_by == 'AI':
            if code == 3:
                mask = ((ai_type_data == code) | (ai_type_data == 4)) & var_mask

            else:
                mask = (ai_type_data == code) & var_mask


        elif grade_by == 'Cor mean':
            if code == 4:
                mask = (
                        (cor_mean_data <= -0.4) & var_mask )
            else:
                mask = (
                        (cor_mean_data <= -(code*0.1)) & (cor_mean_data > -((code+1)*0.1)) & var_mask )

        elif grade_by == 'All':
            mask = var_mask

        valid_indices = np.where(mask.flatten())[0]

        ########## Spatial scale
        ## Standardized data preparation: drought/wet/normal year diff ###
        standardized_data_drought = pd.DataFrame({
            'POS drought normal diff': pos_diff_drought_normal.flatten()[valid_indices],

            'SOS drought normal diff': sos_diff_drought_normal.flatten()[valid_indices],

            'Cor drought normal diff': cor_diff_drought_normal.flatten()[valid_indices],

            'SM drought normal diff': sm_diff_drought_normal.flatten()[valid_indices],

            'VPD drought normal diff': vpd_diff_drought_normal.flatten()[valid_indices],

            'Ta drought normal diff': ta_diff_drought_normal.flatten()[valid_indices],

            'Pre drought normal diff': pre_diff_drought_normal.flatten()[valid_indices],

            'Srad drought normal diff': srad_diff_drought_normal.flatten()[valid_indices]
        })

        standardized_data_wet = pd.DataFrame({
            'POS wet normal diff': pos_diff_wet_normal.flatten()[valid_indices],

            'SOS wet normal diff': sos_diff_wet_normal.flatten()[valid_indices],

            'Cor wet normal diff': cor_diff_wet_normal.flatten()[valid_indices],

            'SM wet normal diff': sm_diff_wet_normal.flatten()[valid_indices],

            'VPD wet normal diff': vpd_diff_wet_normal.flatten()[valid_indices],

            'Ta wet normal diff': ta_diff_wet_normal.flatten()[valid_indices],

            'Pre wet normal diff': pre_diff_wet_normal.flatten()[valid_indices],

            'Srad wet normal diff': srad_diff_wet_normal.flatten()[valid_indices],
        })

        # standardized_data_corhigh_corlow= pd.DataFrame({
        #     'POS corhigh corlow diff': pos_diff_corhigh_corlow.flatten()[valid_indices],
        #
        #     'SOS corhigh corlow diff': sos_diff_corhigh_corlow.flatten()[valid_indices],
        #
        #     'Cor corhigh corlow diff': cor_diff_corhigh_corlow.flatten()[valid_indices],
        #
        #     'SM corhigh corlow diff': sm_diff_corhigh_corlow.flatten()[valid_indices],
        #
        #     'VPD corhigh corlow diff': vpd_diff_corhigh_corlow.flatten()[valid_indices],
        #
        #     'Ta corhigh corlow diff': ta_diff_corhigh_corlow.flatten()[valid_indices],
        #
        #     'Pre corhigh corlow diff': pre_diff_corhigh_corlow.flatten()[valid_indices],
        #
        #     'Srad corhigh corlow diff': srad_diff_corhigh_corlow.flatten()[valid_indices]
        # })

        standardized_data_drought = standardized_data_drought.dropna().reset_index()
        standardized_data_wet = standardized_data_wet.dropna().reset_index()
        # standardized_data_corhigh_corlow = standardized_data_corhigh_corlow.dropna().reset_index()

        # print('standardized_data_drought:\n', standardized_data_drought.iloc[:20, :])
        # print('standardized_data_wet:\n', standardized_data_wet.iloc[:20, :])

        # sys.exit()

        ######## Partial correlation & ML ###########
        def seq_pcorr(df, y, x_list):
            """
            Perform sequential partial correlation analysis.

            Parameters:
            - df (DataFrame): Input DataFrame containing the variables.
            - y (str): Dependent variable.
            - x_list (list): List of independent variables.

            Returns:
            - result_df (DataFrame): DataFrame containing the sequential partial correlation results.
            """
            res_list = []
            # print('df type:', df.dtypes)
            # print('y:', y)
            # print('x_list:', x_list)
            for var in x_list:
                # Perform partial correlation for each variable with other variables
                res = pg.partial_corr(df, y=y, x=var, covar=[item for item in x_list if item != var])
                res['var'] = var
                res_list.append(res)
            result_df = pd.concat(res_list, axis=0)
            return result_df

        results_dict = {}  # save results

        for y in y_list :

            ### ============= 1. Partial correlation analysis ============== ###
            print('Start partial analyze!')
            # if y == 'POS mean' :
            #     # x_list = ['SOS mean', 'SOS std', 'Cor mean', 'Cor std',
            #     #           'SM mean', 'SM std', 'VPD mean', 'VPD std', 'Ta mean', 'Ta std', 'Pre mean', 'Pre std', 'Srad mean', 'Srad std']
            #     x_list = ['SOS mean', 'Cor mean',
            #               'SM mean', 'VPD mean', 'Ta mean', 'Pre mean', 'Srad mean']
            #     df_pcorr = seq_pcorr(standardized_data_drought, y, x_list)[['var', 'r', 'p-val']].reset_index(drop=True).rename(
            #         columns={'r': 'r_pcorr', 'p-val': 'p_pcorr'})
            #     ml_data = standardized_data_drought
            #
            # elif y == 'POS std':
            #     x_list = ['SOS std', 'Cor std',
            #               'SM std', 'VPD std', 'Ta std', 'Pre std', 'Srad std']
            #     df_pcorr = seq_pcorr(outlier_standardized_data, y, x_list)[['var', 'r', 'p-val']].reset_index(drop=True).rename(
            #         columns={'r': 'r_pcorr', 'p-val': 'p_pcorr'})
            #     ml_data = outlier_standardized_data
            #
            #
            # elif y == 'POS slope':
            #     x_list = ['SOS slope', 'SOS std',
            #               'Cor slope', 'Cor std',
            #               'SM slope', 'SM std', 'VPD slope', 'VPD std',
            #               'Ta slope', 'Ta std', 'Pre slope', 'Pre std',
            #               'Srad slope', 'Srad std']
            #     df_pcorr = seq_pcorr(outlier_standardized_data, y, x_list)[['var', 'r', 'p-val']].reset_index(drop=True).rename(
            #         columns={'r': 'r_pcorr', 'p-val': 'p_pcorr'})
            #     ml_data = outlier_standardized_data


            if y == 'POS drought normal diff':
                x_list = ['SOS drought normal diff',
                          'Cor drought normal diff',
                          'SM drought normal diff',
                          'VPD drought normal diff',
                          'Ta drought normal diff',
                          'Pre drought normal diff',
                          'Srad drought normal diff']
                df_pcorr = seq_pcorr(standardized_data_drought, y, x_list)[['var', 'r', 'p-val']].reset_index(drop=True).rename(
                    columns={'r': 'r_pcorr', 'p-val': 'p_pcorr'})

            elif y == 'POS wet normal diff':
                x_list = ['SOS wet normal diff',
                          'Cor wet normal diff',
                          'SM wet normal diff',
                          'VPD wet normal diff',
                          'Ta wet normal diff',
                          'Pre wet normal diff',
                          'Srad wet normal diff']
                df_pcorr = seq_pcorr(standardized_data_wet, y, x_list)[['var', 'r', 'p-val']].reset_index(drop=True).rename(
                    columns={'r': 'r_pcorr', 'p-val': 'p_pcorr'})

            # elif y == 'POS corhigh corlow diff':
            #     x_list = ['SOS corhigh corlow diff',
            #               'Cor corhigh corlow diff',
            #               'SM corhigh corlow diff',
            #               'VPD corhigh corlow diff',
            #               'Ta corhigh corlow diff',
            #               'Pre corhigh corlow diff',
            #               'Srad corhigh corlow diff']
            #     df_pcorr = seq_pcorr(standardized_data_corhigh_corlow, y, x_list)[['var', 'r', 'p-val']].reset_index(drop=True).rename(
            #         columns={'r': 'r_pcorr', 'p-val': 'p_pcorr'})
            #     ml_data = standardized_data_corhigh_corlow



            df_pcorr = df_pcorr.rename(columns={'var': 'feature'})
            df_pcorr['Significance'] = ''

            for x in x_list:
                p_value = df_pcorr[df_pcorr['feature'] == x]['p_pcorr'].iloc[0]
                if p_value < 0.01:
                    significance = '**'  # p < 0.01 mark **
                elif p_value < 0.05:
                    significance = '*'  # 0.01 <= p < 0.05 mark *
                else:
                    significance = ''  # p >= 0.05 no mark

                df_pcorr.loc[df_pcorr['feature'] == x, 'Significance'] = significance

            df_pcorr['Abs_Coefficient'] = df_pcorr['r_pcorr'].abs()
            print('df_pcorr:\n', df_pcorr)

            # sys.exit()

            ###### ================= 2. ML =================== #######
            if y == 'POS drought normal diff':
                ml_data = standardized_data_drought
            elif y == 'POS wet normal diff':
                ml_data = standardized_data_wet
            # elif y == 'POS corhigh corlow diff':
            #     ml_data = standardized_data_corhigh_corlow

            X = ml_data[x_list]
            Y = ml_data[y]

            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
            print("test split done")


            def calculate_shap(xgb_model, x_train, y_train, x_test, y_test):

                # Step 1: Calculate MSE and R2 for train and test data
                mse_train = mean_squared_error(y_train, xgb_model.predict(x_train))
                r2_train = xgb_model.score(x_train, y_train)
                print(f'Train R2: {r2_train}, MSE: {mse_train}')

                y_train_pred = xgb_model.predict(x_test)
                r2_test = r2_score(y_test, y_train_pred)
                mse_test = mean_squared_error(y_test, y_train_pred)
                print(f'Test R2: {r2_test}, MSE: {mse_test}')

                # Step 2: Calculate SHAP values
                explainer = shap.TreeExplainer(xgb_model)
                print("SHAP explainer done")
                shap_values = explainer(x_train)
                print("SHAP values calculation done")

                # Convert SHAP values into a DataFrame
                shap_values_df = pd.DataFrame(shap_values.values, columns=x_train.columns, index=x_train.index)
                shap_values_df.columns = ['shap_' + col for col in x_train.columns]

                shap_values_df = pd.concat(
                    [x_train.reset_index(drop=True),
                     pd.DataFrame(shap_values.values,
                                  columns=[f'shap_{c}' for c in x_train.columns])],
                    axis=1
                )

                # Step 3: Consolidate feature importance and metrics
                df_xgb_model = pd.DataFrame(xgb_model.feature_importances_, index=x_list,
                                            columns=["importance"]).reset_index().rename(
                    columns={'index': 'var'})
                print("Feature importances calculated")
                print(df_xgb_model)

                df_xgb_model["model_mse"] = mse_train
                df_xgb_model["model_r2"] = r2_train
                df_xgb_model["model_test_mse"] = mse_test
                df_xgb_model["model_test_r2"] = r2_test

                # Step 4: Calculate mean absolute SHAP values
                mean_abs_shap_values = abs(shap_values.values).mean(0)
                print(f'Mean absolute SHAP values: {mean_abs_shap_values}')

                df_shap = pd.DataFrame([x_list, mean_abs_shap_values], index=["var", "shapvalue"]).T
                print("SHAP values calculated")

                pos_range = y_train.max() - y_train.min()
                shap_sum = shap_values.values.sum(axis=1)  # sum of SHAP per pixel

                print("POS_mean spatial range (days):", pos_range)
                print("SHAP sum range (days):", shap_sum.max() - shap_sum.min())

                # Step 5: Merge SHAP values and feature importance
                df_merge = pd.merge(df_shap, df_xgb_model, on="var")
                # print('df_merge1:\n', df_merge)
                df_merge = df_merge.rename(columns={'var': 'feature'})
                # df_merge['importance'] = df_merge['importance'] / df_merge['importance'].sum()  # Normalize feature importance
                # df_merge['shapvalue'] = df_merge['shapvalue'] / df_merge['shapvalue'].sum()  # Normalize SHAP importance
                # df_merge['mean'] = (df_merge['importance'] + df_merge['shapvalue']) / 2  # Calculate mean importance

                print('df_merge:\n', df_merge)

                # # Step 6: Sort features by SHAP value
                # df_merge = df_merge.sort_values(by='shapvalue', ascending=False)

                return shap_values_df, df_merge ,r2_test

            ############################### XGBoost ########################################
            if ML_model_in_spatio == 'XGBoost':
                ## K-fold cross-validation and grid search to find the optimal parameters ##
                xgb_model = xgb.XGBRegressor(random_state=42)
                # Define parameter grid
                param_grid = {'n_estimators': [50, 100, 150, 200, 250, 300],
                              'max_depth': [5, 10, 15],
                              'learning_rate': [0.05, 0.1],
                              'subsample': [0.8, 0.9, 1],
                              'colsample_bytree': [0.8, 0.9, 1]}
                # Define K-fold cross-validation (K-Fold)
                kfold = KFold(n_splits=10, shuffle=True, random_state=42)
                # Use grid search to find optimal parameters
                grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='r2', cv=kfold,
                                           verbose=10, n_jobs=15, error_score='raise')

                # Fit the model
                grid_search.fit(x_train, y_train)

                print(f'XGboost model done')

                # Output optimal parameters and optimal score
                print(f"Best parameters: {grid_search.best_params_}")
                print(f"Best R2 score: {grid_search.best_score_}")

                Best_parameters = pd.DataFrame(grid_search.best_params_, index=[0])
                print(f'Best_parameters:{Best_parameters}')
                Best_parameters['mean R2 score'] = round(grid_search.best_score_, 4)
                Best_parameters.to_csv(
                    fr'D:\CAU\phenology_swc_vpd\Global_test4\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_{ML_model_in_spatio}_TestSize{test_size}_Best_parameters_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).csv')
                #
                # if grade_by == 'Veg':
                #     Best_parameters.to_csv(fr'D:\CAU\phenology_swc_vpd\Global_test4\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_VegType({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).csv')
                # elif grade_by == 'AI':
                #     Best_parameters.to_csv(fr'D:\CAU\phenology_swc_vpd\Global_test4\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_AI({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).csv')
                # elif grade_by == 'Cor mean':
                #     Best_parameters.to_csv(fr'D:\CAU\phenology_swc_vpd\Global_test4\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_Cormean({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).csv')
                # elif grade_by == 'All':
                #     Best_parameters.to_csv(fr'D:\CAU\phenology_swc_vpd\Global_test4\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_All_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).csv')
                print(f'Best_parameters done')

                # ## Train the model using optimal parameters ##
                # xgboost = grid_search.best_estimator_

                ######################################################
                # #
                Best_parameters = pd.read_csv(
                    fr'D:\CAU\phenology_swc_vpd\Global_test4\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_{ML_model_in_spatio}_TestSize{test_size}_Best_parameters_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).csv')
                # if grade_by == 'Veg':
                #     Best_parameters = pd.read_csv(fr'D:\CAU\phenology_swc_vpd\Global_test4\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_VegType({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).csv')
                # elif grade_by == 'AI':
                #     Best_parameters = pd.read_csv(fr'D:\CAU\phenology_swc_vpd\Global_test4\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_AI({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).csv')
                # elif grade_by == 'Cor mean':
                #     Best_parameters = pd.read_csv(fr'D:\CAU\phenology_swc_vpd\Global_test4\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_Cormean({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).csv')
                # elif grade_by == 'All':
                #     Best_parameters = pd.read_csv(fr'D:\CAU\phenology_swc_vpd\Global_test4\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_All_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).csv')

                n_estimators = int(Best_parameters['n_estimators'][0])
                max_depth = int(Best_parameters['max_depth'][0])
                colsample_bytree = float(Best_parameters['colsample_bytree'][0])
                subsample = float(Best_parameters['subsample'][0])
                learning_rate = float(Best_parameters['learning_rate'][0])

                xgboost = XGBRegressor(n_estimators=n_estimators,
                                         max_depth=max_depth,
                                         colsample_bytree=colsample_bytree,
                                         subsample=subsample,
                                         learning_rate=learning_rate,
                                         random_state=42,
                                         n_jobs=20)

                xgboost.fit(x_train,y_train)
                print(f"xgb_model:", xgboost)


                shap_values_df, ModelImportance_and_ShapValueMean, r2_test = calculate_shap(xgboost, x_train, y_train, x_test, y_test)


            ############################ RF #########################################
            elif ML_model_in_spatio == 'RF':
                ## K-fold cross-validation and grid search for optimal parameters ##
                rf_model = RandomForestRegressor(random_state=42, n_jobs=15)
                # Define parameter grid
                param_grid = {
                    'n_estimators': [50, 100, 150, 200, 250, 300],
                    'max_depth': [5, 10, 15, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7]
                }
                # Define K-fold cross-validation (K-Fold)
                kfold = KFold(n_splits=10, shuffle=True, random_state=42)
                # Use grid search to find optimal parameters
                grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, scoring='r2', cv=kfold,
                                           verbose=10, n_jobs=15, error_score='raise')
                # Fit the model
                grid_search.fit(x_train, y_train)

                print(f'RF model done')

                print(f"Best parameters: {grid_search.best_params_}")
                print(f"Best R2 score: {grid_search.best_score_}")

                Best_parameters = pd.DataFrame(grid_search.best_params_, index=[0])
                print(f'Best_parameters:{Best_parameters}')
                Best_parameters['mean R2 score'] = round(grid_search.best_score_, 4)
                if analyze_by == 'All':
                    Best_parameters.to_csv(
                        fr'\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_{ML_model_in_spatio}_TestSize{test_size}_Best_parameters_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).csv')
                elif analyze_by != 'All':
                    Best_parameters.to_csv(
                        fr'D:\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEI{spei_strength}_Cor{cor_test_number}_{ML_model_in_spatio}_TestSize{test_size}_BestParameters_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_analyzeBy({analyze_by}).csv')

                # if grade_by == 'Veg':
                #     Best_parameters.to_csv(
                #         fr'D:\CAU\phenology_swc_vpd\Global_test4\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_RF_Best_parameters_VegType({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).csv')
                # elif grade_by == 'AI':
                #     Best_parameters.to_csv(
                #         fr'D:\CAU\phenology_swc_vpd\Global_test4\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_RF_Best_parameters_AI({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).csv')
                # elif grade_by == 'Cor mean':
                #     Best_parameters.to_csv(
                #         fr'D:\CAU\phenology_swc_vpd\Global_test4\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_RF_Best_parameters_Cormean({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).csv')
                # elif grade_by == 'All':
                #     Best_parameters.to_csv(
                #         fr'D:\CAU\phenology_swc_vpd\Global_test4\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_RF_Best_parameters_All_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).csv')
                print(f'Best_parameters done')

                # ## 使用最优参数训练模型 ##
                # rf = grid_search.best_estimator_

                ######################################################
                # #
                if analyze_by == 'All':
                    Best_parameters = pd.read_csv(
                        fr'D:\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_{ML_model_in_spatio}_TestSize{test_size}_Best_parameters_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).csv')
                elif analyze_by != 'All':
                    Best_parameters = pd.read_csv(
                        fr'D:\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEI{spei_strength}_Cor{cor_test_number}_{ML_model_in_spatio}_TestSize{test_size}_BestParameters_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_analyzeBy({analyze_by}).csv')

                # if grade_by == 'Veg':
                #     Best_parameters = pd.read_csv(
                #         fr'D:\CAU\phenology_swc_vpd\Global_test4\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_RF_Best_parameters_VegType({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).csv')
                # elif grade_by == 'AI':
                #     Best_parameters = pd.read_csv(
                #         fr'D:\CAU\phenology_swc_vpd\Global_test4\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_RF_Best_parameters_AI({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).csv')
                # elif grade_by == 'Cor mean':
                #     Best_parameters = pd.read_csv(
                #         fr'D:\CAU\phenology_swc_vpd\Global_test4\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_RF_Best_parameters_Cormean({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).csv')
                # elif grade_by == 'All':
                #     Best_parameters = pd.read_csv(
                #         fr'D:\CAU\phenology_swc_vpd\Global_test4\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_RF_Best_parameters_All_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).csv')

                n_estimators = int(Best_parameters['n_estimators'][0])
                max_depth = int(Best_parameters['max_depth'][0])
                min_samples_split = int(Best_parameters['min_samples_split'][0])
                min_samples_leaf = int(Best_parameters['min_samples_leaf'][0])
                max_features = Best_parameters['max_features'][0]

                if isinstance(max_features, str):
                    max_features = max_features
                else:
                    max_features = float(max_features)

                rf = RandomForestRegressor(n_estimators=n_estimators,
                                         max_depth=max_depth,
                                         min_samples_split=min_samples_split,
                                         min_samples_leaf=min_samples_leaf,
                                         max_features=max_features,
                                         random_state=42,
                                         n_jobs=20)

                rf.fit(x_train, y_train)
                print(f"rf_model:", rf)

                shap_values_df, ModelImportance_and_ShapValueMean, r2_test = calculate_shap(rf, x_train, y_train, x_test, y_test)


            print('shap_values_df:\n', shap_values_df)
            print('ModelImportance_and_ShapValueMean:\n', ModelImportance_and_ShapValueMean)

            Best_parameters['r2_test'] = r2_test
            if analyze_by == 'All':
                Best_parameters.to_csv(
                    fr'D:\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_{ML_model_in_spatio}_TestSize{test_size}_Best_parameters_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).csv')
            elif analyze_by != 'All':
                Best_parameters.to_csv(
                    fr'D:\Tif_of_fig\6Partial\In spatio\SM_VPD_Cor17\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEI{spei_strength}_Cor{cor_test_number}_{ML_model_in_spatio}_TestSize{test_size}_BestParameters_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_analyzeBy({analyze_by}).csv')

            results_dict[y] = {
                'df_pcorr': df_pcorr,
                'importance_and_shap_values_mean': ModelImportance_and_ShapValueMean,
                'shap_values_df': shap_values_df,
                'x_list': x_list,
                'y': y
            }


            # srad_std_shap = shap_values_df.values[:, -1]
            print(f"SHAP值统计:")
            print(f"  最小值: {np.nanmin(shap_values_df.values):.1f}")
            print(f"  最大值: {np.nanmax(shap_values_df.values):.1f}")
            print(f"  平均值: {np.nanmean(shap_values_df.values):.1f}")

            #
            in_range_mask = (shap_values_df >= -1) & (shap_values_df <= 1) & (~np.isnan(shap_values_df))
            total_valid = np.sum(~np.isnan(shap_values_df))
            in_range_count = np.sum(in_range_mask)

            # sys.exit()

        ######## ================= 3. plot ====================== #######

        #### Plotting Driver Analysis Results ####
        for y, results in results_dict.items():
            # Use stored results for plotting
            df_pcorr = results['df_pcorr']
            ModelImportance_and_ShapValueMean = results['importance_and_shap_values_mean']
            shap_values_df = results['shap_values_df']
            x_list = results['x_list']

            # ========== 1. Define "unique y-axis order" (based on SHAP / importance) ==========
            feature_order = ModelImportance_and_ShapValueMean['feature'].tolist()
            print(f'feature_order:{feature_order}')

            # # ========== 2. Create figure and axes ==========
            # fig, axes = plt.subplots(1, 3, figsize=(10, 8), sharey=False)
            # ax1, ax2, ax3 = axes
            # plt.subplots_adjust(wspace=0.05)

            # ========== 3. Subplot 1: Partial correlation ==========
            df_pcorr_plot = df_pcorr.copy()
            df_pcorr_plot['feature'] = pd.Categorical(
                df_pcorr_plot['feature'],
                categories=feature_order,
                ordered=True
            )
            df_pcorr_plot = df_pcorr_plot.sort_values('feature')
            print(f'df_pcorr_plot:{df_pcorr_plot}')

            colors = ['#4393c3' if r < 0 else '#d6604d' for r in df_pcorr_plot['r_pcorr']]

            sns.barplot(
                x='Abs_Coefficient',
                y='feature',
                data=df_pcorr_plot,
                palette=colors,
                width=0.4,
                ax=ax1,
                alpha=0.8
            )

            ax1.tick_params(axis='both', length=2, pad=3)  # ,which='major', )

            for i, (r, sig) in enumerate(zip(df_pcorr_plot['Abs_Coefficient'],
                                             df_pcorr_plot['Significance'])):
                ax1.text(r + 0.04, i, sig, va='center')#, fontsize=12)

            # if type == 'Forest' or type == 'Arid' or type == 'All':
            #     word = 'a'
            #     if type == 'All':
            #         ytitle = 1.1
            #     elif type == 'Forest':
            #         ytitle = 1.13
            #     elif type == 'Arid':
            #         ytitle = 1.13
            # elif type == 'Shrub' or type == 'Semi-arid':
            #     word = 'd'
            #     ytitle = 1.07
            # elif type == 'Savanna' or type == 'Dry sub-humid':
            #     word = 'g'
            #     ytitle = 1.07
            # elif type == 'Grass' or type == 'Humid':
            #     word = 'j'
            #     ytitle = 1.07
            # ax1.set_title(f'({word}) {type}', y=ytitle)
            if type in ['All', 'Grass', 'Humid', 'Cor(lt-0.4)']:
                ax1.set_xlabel(f'Partial correlation')
            else:
                ax1.set_xlabel('')

            ax1.set_xlim(0, 0.8)
            ax1.set_xticks(np.arange(0, 0.81, 0.2))
            ax1.set_xticklabels([f"{int(x)}" if x.is_integer() else f"{x:.1f}" for x in np.arange(0, 0.81, 0.2)])

            ax1.set_ylabel('')
            ax1.set_yticklabels(['SOS', 'Coupling', 'SM', 'VPD', 'Ta', 'Pre', 'Srad'])
            # ax1.set_yticklabels(['Cor', 'SM', 'VPD', 'Ta', 'Pre', 'Srad'])
            ax1.tick_params(axis='y', length=0)

            # legend_handles = [
            #     mpatches.Patch(color='#4393c3', label='Negative', alpha=0.8),
            #     mpatches.Patch(color='#d6604d', label='Positive', alpha=0.8)
            # ]
            # ax1.legend(handles=legend_handles, frameon=False, fontsize=8, loc='lower right')
            from matplotlib.lines import Line2D

            legend_handles = [
                Line2D([0], [0], linestyle='none', marker='s', markerfacecolor='#4393c3',
                       markeredgecolor='none', markersize=6, label='Negative', alpha=0.8),
                Line2D([0], [0], linestyle='none', marker='s', markerfacecolor='#d6604d',
                       markeredgecolor='none', markersize=6, label='Positive', alpha=0.8)
            ]

            ax1.legend(handles=legend_handles, frameon=False, loc='lower right', fontsize=8 ,
                       handletextpad=0.5, handlelength=1.0)  # 缩短句柄长度使其看起来更像纯方块

            # ========== 4. ax 2：ML feature importance ==========
            importance_plot = ModelImportance_and_ShapValueMean.copy()

            importance_plot['feature'] = pd.Categorical(
                importance_plot['feature'],
                categories=feature_order,
                ordered=True
            )
            importance_plot = importance_plot.sort_values('feature')
            print(f'importance_plot:{importance_plot}')

            sns.barplot(
                x='importance',
                y='feature',
                data=importance_plot,
                color='#4393c3',
                alpha = 0.8,
                width=0.4,
                ax=ax2
            )

            ax2.tick_params(axis='both', length=2, pad=3)  # ,which='major', )

            # if type == 'Forest' or type == 'Arid' or type == 'All':
            #     word = 'b'
            #     if type == 'All':
            #         ytitle = 1.1
            #     elif type == 'Forest':
            #         ytitle = 1.13
            # elif type == 'Shrub' or type == 'Semi-arid':
            #     word = 'e'
            #     ytitle = 1.07
            # elif type == 'Savanna' or type == 'Dry sub-humid':
            #     word = 'h'
            #     ytitle = 1.07
            # elif type == 'Grass' or type == 'Humid':
            #     word = 'k'
            #     ytitle = 1.07
            # ax2.set_title(f'({word}) {type}', y=ytitle)
            if type in ['All', 'Grass', 'Humid', 'Cor(lt-0.4)']:
                ax2.set_xlabel(f'{ML_model_in_spatio} importance')
            else:
                ax2.set_xlabel('')

            if grade_by == 'All':
                x_min = 0
                x_max = 0.45
                interval = 0.1
            else:
                x_min = 0
                x_max = 0.8
                interval = 0.2
            ax2.set_xlim(x_min, x_max)
            ax2.set_xticks(np.arange(x_min, x_max + 0.01, interval))
            ax2.set_xticklabels([f"{int(x)}" if x.is_integer() else f"{x:.1f}" for x in np.arange(x_min, x_max + 0.01, interval)])

            ax2.set_ylabel('')
            ax2.tick_params(axis='y', length=0, labelright=False, labelleft=False)

            # ========== 5. ax 3：SHAP beeswarm（关键：用 order 控制顺序） ==========
            plt.sca(ax3)
            ### 1 SHAP value scatter plot
            # —— Construct SHAP Explanation (ordered by x_list) ——
            shap_values = shap.Explanation(
                values=shap_values_df[[f'shap_{c}' for c in x_list]].values,
                data=shap_values_df[x_list].values,
                feature_names=x_list
            )

            # —— Map feature_order to SHAP index order ——
            order_idx = [x_list.index(f) for f in feature_order if f in x_list]
            print(f'order_idx:{order_idx}')

            shap.plots.beeswarm(
                shap_values,
                order=order_idx,
                max_display=len(order_idx),
                plot_size=None,  # Controlled by external ax
                show=False,
                color_bar=False
            )

            # Change point colors
            new_cmap = plt.get_cmap('coolwarm')

            for coll in ax3.collections:
                coll.set_sizes([5])
                # Map existing color array to the new cmap
                # SHAP internally usually maps feature values between [0, 1]
                coll.set_cmap(new_cmap)

            # Point size
            for coll in ax3.collections:
                coll.set_sizes([5])

            ax3.tick_params(axis='both', length=2, pad=3)  # ,which='major', )

            fp = ax2.get_xticklabels()[0].get_fontproperties()
            ax3.set_xlim(-1.5, 1.5)
            ax3.set_xticks(np.arange(-1.5, 1.51, 0.5))
            ax3.set_xticklabels([f"{int(x)}" if x.is_integer() else f"{x:.1f}" for x in np.arange(-1.5, 1.51, 0.5)], fontproperties=fp)

            # if type == 'Forest' or type == 'Arid' or type == 'All':
            #     word = 'c'
            #     if type == 'All':
            #         ytitle = 1.1
            #     elif type == 'Forest':
            #         ytitle = 1.13
            # elif type == 'Shrub' or type == 'Semi-arid':
            #     word = 'f'
            #     ytitle = 1.07
            # elif type == 'Savanna' or type == 'Dry sub-humid':
            #     word = 'i'
            #     ytitle = 1.07
            # elif type == 'Grass' or type == 'Humid':
            #     word = 'l'
            #     ytitle = 1.07
            # ax3.set_title(f'({word}) {type}', y=ytitle)
            fp = ax2.xaxis.label.get_fontproperties()
            if type in ['All', 'Grass', 'Humid', 'Cor(lt-0.4)']:
                ax3.set_xlabel(f'{ML_model_in_spatio} SHAP value', fontproperties = fp)#, labelpad=4)
            else:
                ax3.set_xlabel('')
            ax3.tick_params(axis='y', labelleft=False)

            ### 2 Shap abs mean

            ax3_twin = ax3.twiny()
            ax3_twin.set_ylim(ax1.get_ylim())

            shapvalue_mean = ModelImportance_and_ShapValueMean.copy()

            shapvalue_mean['feature'] = pd.Categorical(
                shapvalue_mean['feature'],
                categories=feature_order[::-1],
                ordered=True
            )
            shapvalue_mean = shapvalue_mean.sort_values('feature')
            print(f'shapvalue_mean:{shapvalue_mean}')
            sns.barplot(
                x='shapvalue',
                y='feature',
                data=shapvalue_mean,
                color='gray',
                width=0.4,
                alpha=0.3,
                ax=ax3_twin,
                zorder=0
            )

            # Set top X-axis limits and tick labels
            ax3_twin.set_xlim(0, 0.3)  # Leave margin for bar chart
            ax3_twin.set_xticks(np.arange(0, 0.301, 0.1))  # Leave margin for bar chart
            ax3_twin.set_xticklabels('0' if x == 0 else
                                     f'{x:.1f}' for x in np.arange(0, 0.301, 0.1))  # Leave margin for bar chart
            ax3_twin.tick_params(axis='x', length=2, pad=1)
            if type in ['All', 'Forest', 'Arid', 'Cor(-0.1~0)']:
                ax3_twin.set_xlabel(f'mean |{ML_model_in_spatio} SHAP value|', labelpad = 6)
            elif veg_type == 'Grass' and grade_by == 'Veg':
                ax3_twin.set_xlabel(f'mean |{ML_model_in_spatio} SHAP value|', labelpad=6)
            else:
                ax3_twin.set_xlabel('')


            ### ================ 6. ax 4：ax3的color ================ ###
            norm = mpl.colors.Normalize(vmin=0, vmax=1)

            cb = fig.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=new_cmap),
                cax=ax4,
                orientation='vertical'
            )

            cb.ax.tick_params(
                axis='y',
                length=0,
                pad=2
            )
            cb.set_ticks([0, 1])
            cb.set_ticklabels(['Low', 'High'])
            cb.set_label('Feature value', labelpad=-7)
            cb.outline.set_visible(False)

            pos = ax4.get_position()

            new_width = pos.width * 0.5

            ax4.set_position([pos.x0-0.01, pos.y0, new_width, pos.height])

            # ========== 6. Unified Spine/Border Style ==========
            for ax in [ax1, ax2, ax3]:
                for spine in ax.spines.values():
                    spine.set_linewidth(1)
                    spine.set_edgecolor('black')

            # ---------------------- Unified Y-Axis Alignment ----------------------
            # 1. Use ax1 as baseline
            y_min, y_max = ax1.get_ylim()
            y_lim = (min(y_min, y_max), max(y_min, y_max))  # Ensure ascending order

            # 2. Get y-ticks from ax1
            yticks = ax1.get_yticks()

            # 3. Apply iteratively to ax2 and ax3
            for ax in [ax3_twin]:
                # Set identical ylim (ensuring ascending order)
                ax.set_ylim(y_lim)

                # Set identical ticks
                ax.set_yticks(yticks)

    if grade_by in ['Veg', 'AI', 'Cor mean']:
        # plt.tight_layout()
        if analyze_by == 'All':
            fig_path = rf'D:\CAU\phenology_swc_vpd\Global_test5\Fig\Fig 6 Driver of POS difference in drought event\{ML_model_in_spatio}_by_all_drought_event\{grade_by}\DroughtEventTimes({drought_or_wet_times})_SPEIstrength({spei_strength})_SM_VPD_Cor{cor_test_number}_Global_{grade_by}_{ML_model_in_spatio}_TestSize({test_size})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).png'
        elif analyze_by == 'advance PPT':
            fig_path = rf'D:\CAU\phenology_swc_vpd\Global_test5\Fig\Fig 6 Driver of POS difference in drought event\{ML_model_in_spatio}_by_advancedPPT_drought_event\{grade_by}\DroughtEventTimes({drought_or_wet_times})_SPEIstrength({spei_strength})_SM_VPD_Cor{cor_test_number}_Global_{grade_by}_{ML_model_in_spatio}_TestSize({test_size})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).png'
        elif analyze_by == 'delay PPT':
            fig_path = rf'D:\CAU\phenology_swc_vpd\Global_test5\Fig\Fig 6 Driver of POS difference in drought event\{ML_model_in_spatio}_by_delayedPPT_drought_event\{grade_by}\DroughtEventTimes({drought_or_wet_times})_SPEIstrength({spei_strength})_SM_VPD_Cor{cor_test_number}_Global_{grade_by}_{ML_model_in_spatio}_TestSize({test_size})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue}).png'
        plt.savefig(
            fig_path,
            dpi=300, bbox_inches='tight')
        print(f'Fig save to：{fig_path}')

    # plt.show()


### S23-25: Distribution of Drought/Wet and normal diff
def plot_pos(drought_difference_data, wet_difference_data, colorbarmin, colorbarmax, grade_by, ax):

    if grade_by != 'All':
        # Uniformly set all font sizes and styles
        plt.rcParams.update({
            'font.family': 'Arial',

            'mathtext.fontset': 'custom',

            'mathtext.rm': 'Arial',  # Regular
            'mathtext.it': 'Arial:italic',  # Italic
            'mathtext.bf': 'Arial:bold',  # Bold

            # Optional (recommended)
            'mathtext.default': 'regular',  # Prevent automatic italicization

            'font.size': 10,
            'axes.titlesize': 10,
            'axes.labelsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            # 'text.usetex': False,  # Do not use external LaTeX
        })

    if grade_by == 'All':
        fig = ax.figure
        gs = ax.get_subplotspec().subgridspec(1, 3,
                                            width_ratios=[5, 5, 4],  # Width ratios for the three columns
                                            height_ratios=[1],  # Reserved for colorbar
                                            wspace=0.25)
        ax.axis('off')

        drought_data_list = [drought_difference_data]
        wet_data_list = [wet_difference_data]
        labels = ['All']

    elif grade_by == 'Veg' or grade_by == 'AI':

        fig = plt.figure(figsize=(8.6, 12.5))
        gs = gridspec.GridSpec(4, 4,
                               width_ratios=[5, 5, 0.04, 4],  # Width ratios for the columns
                               height_ratios=[1, 1, 1, 1],  # Reserved for colorbar
                               hspace=0.25, wspace=0.25)

        if grade_by == 'Veg':
            drought_data_list = [np.where(veg_type_data == i, drought_difference_data, np.nan) for i in [1, 2, 3, 4]]
            wet_data_list = [np.where(veg_type_data == i, wet_difference_data, np.nan) for i in [1, 2, 3, 4]]
            labels = ['Forest', 'Shrub', 'Savanna', 'Grass']
        elif grade_by == 'AI':
            drought_data_list = [
                np.where(ai_type_data == 2, drought_difference_data, np.nan),  # Arid
                np.where((ai_type_data == 3) | (ai_type_data == 4), drought_difference_data, np.nan),  # Semi-Arid (merge 3 and 4)
                np.where(ai_type_data == 5, drought_difference_data, np.nan),  # Dry sub-humid
                np.where(ai_type_data == 6, drought_difference_data, np.nan)  # Humid
            ]
            wet_data_list = [
                np.where(ai_type_data == 2, wet_difference_data, np.nan),  # Arid
                np.where((ai_type_data == 3) | (ai_type_data == 4), wet_difference_data, np.nan),
                # Semi-Arid (merge 3 and 4)
                np.where(ai_type_data == 5, wet_difference_data, np.nan),  # Dry sub-humid
                np.where(ai_type_data == 6, wet_difference_data, np.nan)  # Humid
            ]
            labels = ['Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']

    elif grade_by == 'Cor mean':

        fig = plt.figure(figsize=(8.6, 15.5))
        gs = gridspec.GridSpec(5, 4,
                               width_ratios=[5, 5, 0.03, 4],  # Width ratios for the columns
                               height_ratios=[1, 1, 1, 1, 1],  # Reserved for colorbar
                               hspace=0.25, wspace=0.25)

        drought_data_list = [
            np.where((cor_mean_data >= -0.1) & (cor_mean_data < 0), drought_difference_data, np.nan),
            np.where((cor_mean_data >= -0.2) & (cor_mean_data < -0.1), drought_difference_data, np.nan),
            np.where((cor_mean_data >= -0.3) & (cor_mean_data < -0.2), drought_difference_data, np.nan),
            np.where((cor_mean_data >= -0.4) & (cor_mean_data < -0.3), drought_difference_data, np.nan),
            np.where((cor_mean_data < -0.4), drought_difference_data, np.nan)
        ]
        wet_data_list = [
            np.where((cor_mean_data >= -0.1) & (cor_mean_data < 0), wet_difference_data, np.nan),
            np.where((cor_mean_data >= -0.2) & (cor_mean_data < -0.1), wet_difference_data, np.nan),
            np.where((cor_mean_data >= -0.3) & (cor_mean_data < -0.2), wet_difference_data, np.nan),
            np.where((cor_mean_data >= -0.4) & (cor_mean_data < -0.3), wet_difference_data, np.nan),
            np.where((cor_mean_data < -0.4), wet_difference_data, np.nan)
        ]

        labels = [  # 'Cor(<-0.5)',
            'Cor(-0.1~0)', 'Cor(-0.2~-0.1)', 'Cor(-0.3~-0.2)',
            'Cor(-0.4~-0.3)', 'Cor(lt-0.4)']

    plots = []  # Store plot objects for each subplot



    for i, (drought_data, wet_data, name)  in enumerate(zip(drought_data_list, wet_data_list, labels)):
        if grade_by == 'All':
            pos_ax1 = fig.add_subplot(gs[0, 0])  # Map
            pos_ax2 = fig.add_subplot(gs[0, 1])  # Latitudinal profile
            pos_ax3 = fig.add_subplot(gs[0, 2])  # Colorbar spanning columns
        else:
            pos_ax1 = plt.subplot(gs[i, 0])
            pos_ax2 = plt.subplot(gs[i, 1])
            pos_ax3 = plt.subplot(gs[i, 3])

        print(f'{name} -- valid pixel count for drought-normal POS difference: {np.count_nonzero(np.isfinite(drought_data))}')
        print(f'{name} -- percentage of valid pixels for drought-normal POS difference: {(np.count_nonzero(np.isfinite(drought_data))/np.count_nonzero(np.isfinite(pos_diff_drought_normal)))*100}%')
    #
        ###### Subplot 1: Spatial distribution of POS difference between drought and normal years
        plot_cor_mean_or_slope_and_pvalue_forAllvegType(drought_data,  colorbarmin, colorbarmax, 'drought', name, pos_ax1)
        plot_cor_mean_or_slope_and_pvalue_forAllvegType(wet_data,  colorbarmin, colorbarmax,'wet', name, pos_ax2)
        plot_cor_mean_or_slope_lt0_sig_and_lt0_nosig_and_gt0_forAllvegType(drought_data, wet_data, -12, 12, name, pos_ax3)

    if grade_by != 'All':
        # plt.tight_layout()
        # plt.savefig(fr'D:\CAU\phenology_swc_vpd\Global_test4\Fig\5Drought_event\POS_diff\SPEI{spei_length}\{grade_by}\POSmean_difference(drought and normal)_way3.png', dpi=600, bbox_inches='tight')
        if analyze_by == 'All':
            plt.savefig(fr'D:\Fig\Fig 6 Driver of POS difference in drought event\{ML_model_in_spatio}_by_all_drought_event\{grade_by}\DroughtEventTimes({drought_or_wet_times})_SPEIstrength({spei_strength})_POSmean_difference(drought and normal)_way3_CorSig({SigCorPvalue}).png', dpi=300, bbox_inches='tight')
        elif analyze_by == 'advance PPT':
            plt.savefig(
                fr'D:\Fig\Fig 6 Driver of POS difference in drought event\{ML_model_in_spatio}_by_advancedPPT_drought_event\{grade_by}\DroughtEventTimes({drought_or_wet_times})_SPEIstrength({spei_strength})_POSmean_difference(drought and normal)_way3_CorSig({SigCorPvalue}).png',
                dpi=300, bbox_inches='tight')
        elif analyze_by == 'delay PPT':
            plt.savefig(
                fr'D:\Fig\Fig 6 Driver of POS difference in drought event\{ML_model_in_spatio}_by_delayedPPT_drought_event\{grade_by}\DroughtEventTimes({drought_or_wet_times})_SPEIstrength({spei_strength})_POSmean_difference(drought and normal)_way3_CorSig({SigCorPvalue}).png',
                dpi=300, bbox_inches='tight')

    # plt.show()


### Fig 6
def plot_fig6(drought_data, wet_data,
              colorbarmin, colorbarmax,
              pos_data, sos_data, cor_data, sm_data, vpd_data, ta_data, pre_data, srad_data):

    # Uniformly set all font sizes and styles
    plt.rcParams.update({
        'font.family': 'Arial',

        'mathtext.fontset': 'custom',

        'mathtext.rm': 'Arial',  # Regular
        'mathtext.it': 'Arial:italic',  # Italic
        'mathtext.bf': 'Arial:bold',  # Bold

        # Optional (recommended)
        'mathtext.default': 'regular',  # Prevent automatic italicization

        'font.size': 10,
        'axes.titlesize': 10,
        'axes.labelsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        # 'text.usetex': False,  # Do not use external LaTeX
    })

    fig = plt.figure(figsize=(8.4, 7))
    gs = gridspec.GridSpec(2, 3,
                           width_ratios=[0.5, 4, 0.5],  # Width ratios for the columns
                           height_ratios=[1, 1],  # Reserved for colorbar
                           hspace=0.7, wspace=0.15)


    ### a Drought and normal diff
    combine_ax1 = plt.subplot(gs[0, :])
    plot_pos(drought_data, wet_data, colorbarmin, colorbarmax,'All', combine_ax1)

    ### d Driving factor analysis of All pixels
    combine_ax2 = plt.subplot(gs[1, 1])
    Partial_ML_calculate_and_plot(pos_data, sos_data, cor_data, sm_data, vpd_data, ta_data, pre_data, srad_data, 'All', 'All', combine_ax2)

    # ### d Driving factor analysis of only grassland
    # combine_ax3 = plt.subplot(gs[1, 1])
    # Partial_ML_calculate_and_plot(pos_data, sos_data, cor_data, sm_data, vpd_data, ta_data, pre_data, srad_data, 'Veg', 'Grass', combine_ax3)

    # plt.savefig(fr'D:\CAU\phenology_swc_vpd\Global_test4\Fig\6Partial\SM_VPD_Cor17\In spatio\SPEI{spei_length}\Climate{climate_test_number}\Based_on_detrendPheno\POSdiff_combined_SM_VPD_Cor17_8_{months_before_pos}_Global_All_XGBoost_POS drought normal diff_Outlier({Outlier})_CorSig({SigCorPvalue}).png',
    if analyze_by == 'All':
        plt.savefig(fr'{output_path}\DroughtEventTimes({drought_or_wet_times})_SPEIstrength({spei_strength})_POSdiff_combined_SM_VPD_Cor17_8_{months_before_pos}_Global_All_{ML_model_in_spatio}_TestSize{test_size}_POS drought normal diff_Outlier({Outlier})_CorSig({SigCorPvalue}).png',
                    dpi=300, bbox_inches='tight')
    elif analyze_by == 'advance PPT':
        plt.savefig(fr'{output_path}\DroughtTimes({drought_or_wet_times})_SPEI({spei_strength})_POSdiff_combined_SM_VPD_Cor17_8_{months_before_pos}_All_{ML_model_in_spatio}_TestSize{test_size}_POS drought normal diff_Outlier({Outlier})_CorSig({SigCorPvalue}).png',
                    dpi=300, bbox_inches='tight')
    elif analyze_by == 'delay PPT':
        plt.savefig(fr'{output_path}\DroughtTimes({drought_or_wet_times})_SPEI({spei_strength})_POSdiff_combined_SM_VPD_Cor17_8_{months_before_pos}_All_{ML_model_in_spatio}_TestSize{test_size}_POS drought normal diff_Outlier({Outlier})_CorSig({SigCorPvalue}).png',
                    dpi=300, bbox_inches='tight')

    print('Fig 6 plot done!')

    # plt.show()


### Control drought event frequency + drought intensity
pos_mean_difference_indrought_Nodrought = np.where((drought_event_count>=drought_or_wet_times), pos_diff_drought_normal_origin, np.nan)
pos_mean_difference_inwet_Nowet = np.where((wet_event_count>=drought_or_wet_times), pos_diff_wet_normal_origin, np.nan)

print(f'Valid pixel count for pos_mean_difference_indrought_Nodrought: {np.count_nonzero(np.isfinite(pos_mean_difference_indrought_Nodrought))}')
print(f'Valid pixel count for pos_mean_difference_inwet_Nowet: {np.count_nonzero(np.isfinite(pos_mean_difference_inwet_Nowet))}')

if analyze_by == 'All':
    mask = drought_event_count>=drought_or_wet_times
elif analyze_by == 'advance PPT':
    mask = (drought_event_count >= drought_or_wet_times) & (pos_mean_difference_indrought_Nodrought < 0)
elif analyze_by == 'delay PPT':
    mask = (drought_event_count >= drought_or_wet_times) & (pos_mean_difference_indrought_Nodrought > 0)


pos_diff_drought_normal = np.where(mask, pos_diff_drought_normal, np.nan)
sos_diff_drought_normal = np.where(mask, sos_diff_drought_normal, np.nan)
cor_diff_drought_normal = np.where(mask, cor_diff_drought_normal, np.nan)
sm_diff_drought_normal = np.where(mask, sm_diff_drought_normal, np.nan)
vpd_diff_drought_normal = np.where(mask, vpd_diff_drought_normal, np.nan)
ta_diff_drought_normal = np.where(mask, ta_diff_drought_normal, np.nan)
pre_diff_drought_normal = np.where(mask, pre_diff_drought_normal, np.nan)
srad_diff_drought_normal = np.where(mask, srad_diff_drought_normal, np.nan)

# sys.exit()
## Fig 6
plot_fig6(pos_mean_difference_indrought_Nodrought, pos_mean_difference_inwet_Nowet,
          -12, 12,
          pos_diff_drought_normal, sos_diff_drought_normal,
          cor_diff_drought_normal, sm_diff_drought_normal, vpd_diff_drought_normal,
          ta_diff_drought_normal, pre_diff_drought_normal, srad_diff_drought_normal)
print('Fig6 or S22 plot done!')

# # # Fig S 23 - 25
# ax = plt.figure(figsize=(17, 11))
# plot_pos(pos_mean_difference_indrought_Nodrought, pos_mean_difference_inwet_Nowet, -12, 12, 'Veg', ax)
# print('S23 plot done!')
# plot_pos(pos_mean_difference_indrought_Nodrought, pos_mean_difference_inwet_Nowet, -12, 12, 'AI', ax)
# print('S24 plot done!')
# plot_pos(pos_mean_difference_indrought_Nodrought, pos_mean_difference_inwet_Nowet, -12, 12, 'Cor mean', ax)
# print('S25 plot done!')

# # # Fig S 26 - 27
# ax = plt.figure(figsize=(17, 11))
# Partial_ML_calculate_and_plot(pos_diff_drought_normal, sos_diff_drought_normal,
#                               cor_diff_drought_normal, sm_diff_drought_normal, vpd_diff_drought_normal,
#                               ta_diff_drought_normal, pre_diff_drought_normal, srad_diff_drought_normal,
#                               'Veg', 'All', ax)
# print('S26 plot done!')
#
# Partial_ML_calculate_and_plot(pos_diff_drought_normal, sos_diff_drought_normal,
#                               cor_diff_drought_normal, sm_diff_drought_normal, vpd_diff_drought_normal,
#                               ta_diff_drought_normal, pre_diff_drought_normal, srad_diff_drought_normal,
#                               'AI', 'All', ax)
# print('S27 plot done!')
#
# Partial_ML_calculate_and_plot(pos_diff_drought_normal, sos_diff_drought_normal,
#                               cor_diff_drought_normal, sm_diff_drought_normal, vpd_diff_drought_normal,
#                               ta_diff_drought_normal, pre_diff_drought_normal, srad_diff_drought_normal,
#                               'Cor mean', 'All', ax)
# print('S28 plot done!')
