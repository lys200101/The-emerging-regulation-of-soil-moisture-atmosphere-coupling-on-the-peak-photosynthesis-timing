import glob
import os.path
import sys

import numpy as np
from osgeo import gdal
import datetime
from joblib import Parallel, delayed


############################## 1 Inputs and Outputs ##################################
### == Inputs == ###
startyear = 2001   ### Please modify carefully: 2001-2022
endyear = 2024
years_length = endyear - startyear + 1

drought_distinguish_way = 3   ### Way 1/2: Unified drought event definition as SPEI < -1 / -1.4 / -1.8
                                ### Way 3: SPEI2 or SPEI3 in the month before POS (POS-1) < 10% threshold (requires SPEI2/SPEI3)

spei_length = 2  ### 1, 2, or 3

spei_drought_value = -1

spei_drought_threshold_percent = 10   ### Please modify carefully (unit: %)
spei_wet_threshold_percent = 90       ### Please modify carefully (unit: %)

starPheno = 'SOS'
endPheno = 'POS'

sos_input_path = rf'D:\CAU\phenology_swc_vpd\paper\Data_and_code_online\FigShare_data\55km\{starPheno}_55km'
pos_input_path = rf'D:\CAU\phenology_swc_vpd\paper\Data_and_code_online\FigShare_data\55km\{endPheno}_55km'

sos_tiffiles = sorted(glob.glob(os.path.join(sos_input_path, '*.tif')))
pos_tiffiles = sorted(glob.glob(os.path.join(pos_input_path, '*.tif')))

### == Outputs == ###
if spei_length == 1:
    drought_event_outputpath = f'D:\CAU\phenology_swc_vpd\Global_test4\Data\SPEI\Drought_Event\drought_event({starPheno}-{endPheno}_SPEI{spei_length}_lt{spei_drought_value})'
if (spei_length == 2) or (spei_length == 3):
    if (drought_distinguish_way == 1) & (drought_distinguish_way == 2):
        drought_event_outputpath = f'D:\CAU\phenology_swc_vpd\Global_test4\Data\SPEI\Drought_Event\drought_event({endPheno}_SPEI{spei_length}_lt{spei_drought_value})'
    if drought_distinguish_way == 3:
        drought_event_outputpath = f'D:\CAU\phenology_swc_vpd\Global_test4\Data\SPEI\Drought_Event\drought_event({endPheno}_SPEI{spei_length}_threshold10%_way{drought_distinguish_way})'
    if drought_distinguish_way == 4:
        drought_event_outputpath = f'D:\CAU\phenology_swc_vpd\Global_test4\Data\SPEI\Drought_Event\drought_event({starPheno}-{endPheno}_SPEI{spei_length}_lt{spei_drought_value}for2months)'

spei_strength_outputpath = rf'D:\SPEI\NH_SPEI{spei_length}_{spei_length}monthBeforePOS'


############################### 2 Metadata Extraction ##################################
sample = pos_tiffiles[0]
sample_tif = gdal.Open(sample)
sample_array = sample_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)

rows = sample_array.shape[0]
cols = sample_array.shape[1]
# print('rows=', rows, 'cols=', cols)

crs = sample_tif.GetProjectionRef()
gt = sample_tif.GetGeoTransform()

rows_indices = np.repeat(np.arange(rows), cols)
cols_indices = np.tile(np.arange(cols), rows)

##################################### 3 SOS and POS Mean Calculation ####################################
def get_band(tif, stack):
    tif_data = gdal.Open(tif)
    tif_array = tif_data.GetRasterBand(1).ReadAsArray().astype(np.float32)
    stack.append(tif_array)

sos_stack = []
pos_stack = []

for tif in sos_tiffiles:
    get_band(tif, sos_stack)
for tif in pos_tiffiles:
    get_band(tif, pos_stack)

sos_stack = np.stack(sos_stack, axis=0)
pos_stack = np.stack(pos_stack, axis=0)

sos_mean = np.nanmean(sos_stack, axis=0)
print('sos_mean:\n', sos_mean[50, 501])
pos_mean = np.nanmean(pos_stack, axis=0)
print('pos_mean:\n', pos_mean[50, 501])

### Calculate corresponding months
sos_month = np.full((years_length, rows, cols), np.nan)
pos_month = np.full((years_length, rows, cols), np.nan)

for year in range(startyear, endyear + 1):
    y = year - startyear
    for i, j in zip(rows_indices, cols_indices):
        if np.isfinite(sos_stack[y, i, j]) & np.isfinite(pos_stack[y, i, j]):
            sos_date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=int(sos_stack[y, i, j]) - 1)
            pos_date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=int(pos_stack[y, i, j]) - 1)
            sos_month[y, i, j] = sos_date.month
            pos_month[y, i, j] = pos_date.month

##################################### Way 3: Calculate Drought and Wet Thresholds ######################################
### Stack all years and months
spei_all = []
for year in range(startyear, endyear + 1):

    spei_input_path = rf'D:\NH_SPEI01_03\{year}'

    spei_tiffiles = sorted(glob.glob(os.path.join(spei_input_path, '*.tif')))

    def get_SPEIband(tif, stack, spei_length):
        tif_data = gdal.Open(tif)
        tif_array = tif_data.GetRasterBand(spei_length).ReadAsArray().astype(np.float32)
        stack.append(tif_array)

    for tif in spei_tiffiles:
        get_SPEIband(tif, spei_all, spei_length)

spei_all = np.stack(spei_all, axis=0)


### Calculate drought and wet thresholds
spei_sort = np.sort(spei_all, axis=0)
spei_drought_threshold = np.nanpercentile(spei_sort, spei_drought_threshold_percent, axis=0)
spei_wet_threshold = np.nanpercentile(spei_sort, spei_wet_threshold_percent, axis=0)


##################################### 4 Annual Drought Identification ######################################
for year in range(startyear, endyear + 1):

    y = year - startyear

    ############################### 4.1 Extract Preseason SPEI and Identify Drought Events ####################

    drought_event = np.full((rows, cols), np.nan)
    spei_strength = np.full((rows, cols), np.nan)

    for i, j in zip(rows_indices, cols_indices):
        if np.isfinite(sos_stack[y, i, j]) & np.isfinite(sos_stack[y, i, j]):
            sos_month_pixel = int(sos_month[y, i, j])
            pos_month_pixel = int(pos_month[y, i, j])

            if spei_length == 1:

                spei_preseason = spei_all[(sos_month_pixel - 1):pos_month_pixel, i, j]

                if np.any(spei_preseason <= spei_drought_value):
                    drought_event[i, j] = 1
                else:
                    drought_event[i, j] = 0

            elif (spei_length == 2) or (spei_length == 3):

                if (drought_distinguish_way == 1) or (drought_distinguish_way == 2):
                    spei_coll = spei_all[(pos_month_pixel - 1), i, j]

                    if spei_coll <= spei_drought_value:
                        drought_event[i, j] = 1
                    else:
                        drought_event[i, j] = 0

                if drought_distinguish_way == 3:
                    spei_coll = spei_all[(pos_month_pixel - 1), i, j]

                    pixel_drought_threshold = spei_drought_threshold[i, j]
                    pixel_wet_threshold = spei_wet_threshold[i, j]

                    if spei_coll <= pixel_drought_threshold:
                        drought_event[i, j] = 1
                    elif spei_coll >= pixel_wet_threshold:
                        drought_event[i, j] = 2
                    else:
                        drought_event[i, j] = 0

                if drought_distinguish_way == 4:
                    spei_preseason = spei_all[(sos_month_pixel - 1):pos_month_pixel, i, j]

                    if np.sum(np.isfinite(spei_preseason)) >= 2:
                        # print('Preseason exceeds 2 months!')
                        current_streak = 0

                        for spei_value in spei_preseason:
                            if (np.isfinite(spei_value)) and (spei_value < spei_drought_value):
                                current_streak += 1
                                if current_streak == 2:
                                    drought_event[i, j] = 1
                                    break
                                else:
                                    continue
                            else:
                                current_streak = 0

                        if current_streak <= 1:
                            drought_event[i, j] = 0
                    elif np.sum(np.isfinite(spei_preseason)) == 1:
                        print('Preseason is only 1 month!')
                    else:
                        drought_event[i, j] = 0

            spei_strength[i, j] = spei_coll
        else:
            drought_event[i, j] = np.nan
            spei_strength[i, j] = np.nan

    print('drought_event[50, 501]:', drought_event[50, 501])

    ############################### 5 Export TIF Files #####################################
    if (drought_distinguish_way == 1) or (drought_distinguish_way == 2):
        output_path1 = os.path.join(drought_event_outputpath, f'SPEI{spei_length}_lt{spei_drought_value}_droughtEvent_{year}_way{drought_distinguish_way}.tif')

    if drought_distinguish_way == 3:
        output_path1 = os.path.join(drought_event_outputpath, f'SPEI{spei_length}_droughtAndwetEvent_{year}_way{drought_distinguish_way}.tif')

    if drought_distinguish_way == 4:
        output_path1 = os.path.join(drought_event_outputpath, f'SPEI{spei_length}_lt{spei_drought_value}for2months_droughtEvent_{year}_way{drought_distinguish_way}.tif')

    output_path2 = os.path.join(spei_strength_outputpath, f'SPEI{spei_length}_strength_{year}.tif')

    driver = gdal.GetDriverByName('GTiff')

    out_ds1 = driver.Create(
        output_path1,
        cols,
        rows,
        1,
        gdal.GDT_Float32
    )
    out_ds1.SetGeoTransform(gt)
    out_ds1.SetProjection(crs)
    out_ds1.GetRasterBand(1).WriteArray(drought_event)
    out_ds1 = None

    out_ds2 = driver.Create(
        output_path2,
        cols,
        rows,
        1,
        gdal.GDT_Float32
    )
    out_ds2.SetGeoTransform(gt)
    out_ds2.SetProjection(crs)
    out_ds2.GetRasterBand(1).WriteArray(spei_strength)
    out_ds2 = None

    print(f'File successfully exported: {output_path1}')
    print(f'{year} done!')