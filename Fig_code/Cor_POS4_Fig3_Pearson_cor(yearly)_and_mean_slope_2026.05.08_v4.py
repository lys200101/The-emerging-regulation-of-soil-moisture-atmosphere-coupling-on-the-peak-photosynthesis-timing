
import os
import glob
import datetime
import sys

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import pandas as pd
from fontTools.ttLib.tables.otTables import DeltaSetIndexMap
from mpl_toolkits.basemap import Basemap
from osgeo import gdal
import numpy as np
from joblib import parallel_backend
from pandas.core.methods.selectn import SelectNSeries
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import pearsonr
from joblib import Parallel, delayed
import matplotlib as mpl
from matplotlib import colormaps
from sklearn.linear_model import TheilSenRegressor
from scipy.stats import linregress
from scipy import stats
from scipy.stats import mode
import pingouin as pg
from scipy.stats import theilslopes
import pymannkendall as mk
from brokenaxes import brokenaxes
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches


######### Function ##########
### Read time information and extract the date (last 8 digits) from each TIF file
def extract_date_from_filename(filename):
    # Extract filename only (without path and extension)
    basename = os.path.basename(filename)
    filename_without_ext = os.path.splitext(basename)[0]

    # Extract the last 8 digits
    date_str = filename_without_ext[-8:]

    # Validate date format
    if not date_str.isdigit() or len(date_str) != 8:
        raise ValueError(
            f"The last 8 digits of file {filename} are not a valid date (YYYYMMDD required)!"
        )

    # Convert to datetime object
    return datetime.datetime.strptime(date_str, "%Y%m%d")


def clip_by_latitude(gt, rows, cols, lat_min, lat_max):
    """
    Crop data based on latitude range
    Returns: (row_start, row_end, new_gt)
    """
    # Calculate center latitude for each row
    row_centers = np.arange(rows) * gt[5] + gt[3] + gt[5] / 2

    # Find rows within the latitude range (e.g., 30-90 degrees)
    valid_rows = (row_centers >= lat_min) & (row_centers <= lat_max)

    if not np.any(valid_rows):
        raise ValueError(
            f"No valid data found within latitude range {lat_min}-{lat_max}"
        )

    # Find the first and last valid row indices
    valid_row_indices = np.where(valid_rows)[0]
    row_start = valid_row_indices[0]
    row_end = valid_row_indices[-1] + 1  # Slice is left-closed and right-open

    # Calculate new top-left coordinates
    new_top_left_x = gt[0] + row_start * gt[2]  # Usually gt[2] = 0
    new_top_left_y = gt[3] + row_start * gt[5]  # gt[5] is pixel height

    # Create new geotransform parameters
    new_gt = (
        new_top_left_x,
        gt[1],  # Pixel width remains unchanged
        gt[2],  # Row rotation remains unchanged
        new_top_left_y,
        gt[4],  # Column rotation remains unchanged
        gt[5],  # Pixel height remains unchanged
    )

    # print(f"Latitude range: {lat_min}-{lat_max}°N")
    # print(f"Corresponding row range: {row_start} - {row_end - 1}")
    # print(f"Original row count: {rows}, Cropped row count: {row_end - row_start}")
    # print(f"Original top-left latitude: {gt[3]:.2f}°N, New top-left latitude: {new_top_left_y:.2f}°N")

    return row_start, row_end, new_gt


### Read SM and VPD bands
def get_band(tif_file, stack):
    tif = gdal.Open(tif_file)
    data = tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
    stack.append(data)


def get_band_clip(tif_file, stack, row_start, row_end):
    tif = gdal.Open(tif_file)
    climate_data = tif.GetRasterBand(1).ReadAsArray().astype(np.float32)

    climate_data_cropped = climate_data[row_start:row_end, :]
    stack.append(climate_data_cropped)

    tif = None  # Release resources promptly


def cal_pixel_timelength_mean(i, j, data):

    # if len(time_series_clean) > (years_length - 3):
    if len(np.isfinite(data)) > 1:
        result = np.nanmean(data)
        # print(f'pheno mean={result}')
        return (i, j, result)

    else:
        return (i, j, np.nan)


def calculate_senSlope(data, i, j):

    mask = np.isfinite(data)

    data_clean = data[mask]

    years_valid = yearsList[mask]

    if (len(data_clean) >= (years_length / 2)) and (
        len(data_clean) <= years_length
    ):  ### >10 / >17/ >19

        # Sen slope
        result = theilslopes(data_clean, years_valid)

        slope = result.slope

        # Mann-Kendall
        mk_result = mk.original_test(data_clean)

        pvalue = mk_result.p

        # result = mk.original_test(data_clean, alpha=0.05)
        # slope = round(result.slope, 4)
        # pvalue = round(result.p, 2)

    else:
        slope = np.nan
        pvalue = np.nan

        # print(f'slope:{slope}, p:{pvalue}')
    return (i, j, slope, pvalue)


### Extract data within the preseason growing season
# def extract_time_window(year, pos, dates):
def extract_time_window(year, sos, pos, dates):
    # """Extract time window indices for a pixel in the given year based on its sos and pos"""
    # Calculate the start and end dates of the growing season for this pixel in 'year'

    # print('pos:', pos, flush=True)

    ### Preseason considers SOS-POS
    start_date = pd.to_datetime(f"{year}-01-01") + pd.Timedelta(
        days=int(pos - 90)
    )
    end_date = pd.to_datetime(f"{year}-01-01") + pd.Timedelta(days=int(pos))

    # ### Preseason considers only POS
    # start_date = pd.to_datetime(f"{year}-01-01") + pd.Timedelta(days=int(pos) - 30)
    # end_date = pd.to_datetime(f"{year}-01-01") + pd.Timedelta(days=int(pos))

    # print(f'start_date:{start_date}\n'
    #       f'end_date:{end_date}')

    # Find indices in the time series that fall within [start_date, end_date]
    # print('start_date:', start_date, 'end_date:', end_date)
    valid_mask = (dates >= start_date) & (dates < end_date)
    valid_indices = np.where(valid_mask)[0]
    # print('valid_indices:', valid_indices)
    return valid_indices


def compute_pearson_for_pixel(sm_series, vpd_series, pre_origin_series):
    """Calculate Pearson r and p for a pixel time series"""

    pre_origin_series = pd.Series(
        pre_origin_series
    )  # Convert to Series first, otherwise shift will not work

    # === 1. Define precipitation events ===
    pre_event1 = (
        (pre_origin_series.notna())
        & (pre_origin_series > 0.001)
        & (pre_origin_series <= 0.01)
    )  # Light rain
    pre_event2 = (pre_origin_series.notna()) & (
        pre_origin_series > 0.01
    )  # Moderate to heavy rain

    # === 2. Light rain: Exclude current day only ===
    pre_affected1 = pre_event1.copy()

    # === 3. Moderate to heavy rain: Exclude current day + next n days ===
    pre_affected2 = pre_event2.copy()
    # print(f'pre_affected2 without detecting current day and {n_days} days after: {pre_affected2}')

    for i in range(1, n_days + 1):
        pre_affected2 |= pre_event2.shift(i, fill_value=False)
    # print(f'pre_affected2 detecting current day and {n_days} days after: {pre_affected2}')

    # === 4. Combine all precipitation impacts ===
    pre_affected = (
        pre_affected1 | pre_affected2
    ).to_numpy()  # Combine and convert to numpy array

    # === 5. Final valid data mask ===
    valid_precip_mask = ~pre_affected

    mask = (
        np.isfinite(sm_series) & np.isfinite(vpd_series) & valid_precip_mask
    )
    # print(f'Uninfluenced by precipitation: {mask}')

    if (
        np.count_nonzero(mask) > len(sm_series) / 5
        and np.count_nonzero(mask) > 2
    ):
        # print(f'sm_series[mask]:{sm_series[mask]}\nvpd_series[mask]:{vpd_series[mask]}')
        r, p = pearsonr(sm_series[mask], vpd_series[mask])
        # print(f'sos-pos days: {len(sm_series)}, valid days: {np.count_nonzero(mask)}, r: {r}')
        # print(f'Precipitation amount: {pre_origin_series}\nSoil moisture: {sm_series[mask]}')
        return r, p
    else:
        # print(f'Mask count is less than 1/5 of sos-pos days, sos-pos days: {len(sm_series)}, valid days: {np.count_nonzero(mask)}')
        return np.nan, np.nan


def compute_partial_correlation_for_pixel_SM_VPDlag(
    sm_series,
    vpd_series,
    sm_origin_series,
    ta_origin_series,
    srad_origin_series,
    vpd_origin_series,
    pre_origin_series,
):
    """Calculate partial correlation coefficient r and p for a pixel time series"""

    ### Lag pairing
    ### 1. Original lag pairing (existing code)
    sm = sm_series[:-lag_day].flatten()  # SMi
    vpd = vpd_series[:-lag_day].flatten()  # VPDi
    vpd_lag = vpd_series[lag_day:].flatten()  # VPDi+1

    sm_origin = sm_origin_series[:-lag_day].flatten()
    ta_origin = ta_origin_series[:-lag_day].flatten()
    srad_origin = srad_origin_series[:-lag_day].flatten()
    vpd_origin = vpd_origin_series[:-lag_day].flatten()
    pre_origin = pre_origin_series[:-lag_day].flatten()

    n = len(sm)  # Length after truncation

    ### 2. Create new consecutive indices (starting from 0)

    # ### 3. Basic validity check
    # base_mask = (
    #         np.isfinite(sm) &
    #         np.isfinite(vpd) &
    #         np.isfinite(vpd_lag)
    # )
    #
    # # Get valid indices
    # valid_indices = np.where(base_mask)[0]
    #
    # if len(valid_indices) == 0:
    #     return np.nan, np.nan
    #
    # ### 4. Extract valid data (using new indices)
    # sm_valid = sm[valid_indices]
    # vpd_valid = vpd[valid_indices]
    # vpd_lag_valid = vpd_lag[valid_indices]
    #
    # # Control variables also use the same indices
    # sm_origin_valid = sm_origin[valid_indices]
    # ta_valid = ta_origin[valid_indices]
    # srad_valid = srad_origin[valid_indices]
    # vpd_origin_valid = vpd_origin[valid_indices]
    # pre_valid = pre_origin[valid_indices]
    #
    # n_valid = len(sm_valid)

    ### 5. Apply filtering conditions
    # if filter_condition == '1':
    #     # SM decreasing condition
    #     sm_decreasing = np.full(n_valid, False)
    #     if n_valid > 0:
    #         sm_decreasing[0] = True
    #         for t in range(1, n_valid):
    #             sm_decreasing[t] = sm_valid[t] < sm_valid[t - 1]
    #
    #     mask = (
    #             (ta_valid > 5) & (srad_valid > 110) &
    #             (vpd_origin_valid > 0.5) & sm_decreasing
    #     )
    #
    # elif filter_condition == '2':
    #     # SM raw values decreasing
    #     sm_origin_decreasing = np.full(n_valid, False)
    #     if n_valid > 0:
    #         sm_origin_decreasing[0] = True
    #         for t in range(1, n_valid):
    #             sm_origin_decreasing[t] = sm_origin_valid[t] < sm_origin_valid[t - 1]
    #
    #     mask = (
    #             (ta_valid > 5) & (srad_valid > 110) &
    #             (vpd_origin_valid > 0.5) & sm_origin_decreasing
    #     )
    #
    # elif filter_condition == '3':
    #     mask = (
    #             (ta_valid > 5) & (srad_valid > 110) &
    #             (vpd_origin_valid > 0.5) & (sm_valid < 0)
    #     )
    #
    # elif filter_condition == '4':
    #     mask = (
    #             (ta_valid > 5) & (srad_valid > 110) &
    #             (vpd_origin_valid > 0.5)
    #     )

    if filter_condition == "5":

        pre_origin_series = pd.Series(
            pre_origin
        )  # Convert to Series first, otherwise shift will not work

        # === 1. Define precipitation events ===
        pre_event1 = (pre_origin_series > 0.001) & (
            pre_origin_series <= 0.01
        )  # Light rain
        pre_event2 = pre_origin_series > 0.01  # Moderate to heavy rain

        # === 2. Light rain: Exclude current day only ===
        pre_affected1 = pre_event1.copy()

        # === 3. Moderate to heavy rain: Exclude current day + next n days ===
        pre_affected2 = pre_event2.copy()

        for i in range(1, n_days + 1):
            pre_affected2 |= pre_event2.shift(i, fill_value=False)

        # === 4. Combine all precipitation impacts ===
        pre_affected = (
            pre_affected1 | pre_affected2
        ).to_numpy()  # Combine and convert to numpy array

        # === 5. Final valid data mask ===
        valid_precip_mask = ~pre_affected

        mask = (
            np.isfinite(sm)
            & np.isfinite(vpd)
            & np.isfinite(vpd_lag)
            & valid_precip_mask
        )

    # elif filter_condition == '6':
    #     mask = (sm_valid < 0)
    #
    # elif filter_condition == '7':
    #     mask = np.isfinite(sm_valid)

    ### 6. Get final valid points
    final_indices = np.where(mask)[0]
    n_final = len(final_indices)

    ### 7. Check and calculate
    if n_final > n / 5 and n_final > 3:
        sm_final = sm[final_indices]
        vpd_final = vpd[final_indices]
        vpd_lag_final = vpd_lag[final_indices]

        def safe_corr(a, b):
            valid = np.isfinite(a) & np.isfinite(b)
            if np.sum(valid) < 3:
                return np.nan
            a_valid = a[valid]
            b_valid = b[valid]

            return np.corrcoef(a_valid, b_valid)[0, 1]

        r_xy = safe_corr(sm_final, vpd_lag_final)
        r_xz = safe_corr(sm_final, vpd_final)
        r_yz = safe_corr(vpd_lag_final, vpd_final)

        if np.isnan(r_xy) or np.isnan(r_xz) or np.isnan(r_yz):
            return np.nan, np.nan

        # Calculate partial correlation
        denominator = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
        if denominator == 0:
            return np.nan, np.nan
        else:
            # print(f'r_xy={r_xy} r_xz={r_xz} r_yz={r_yz}')
            pcorr = (r_xy - r_xz * r_yz) / denominator

            if abs(pcorr) < 1.0:
                t_stat = pcorr * np.sqrt((n_final - 3) / (1 - pcorr**2))

                # Use t-distribution to calculate two-tailed p-value
                # Degrees of freedom df = n_final - 3
                from scipy import stats

                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_final - 3))
            else:
                # Set p-value to 0 when |r|=1
                p_value = 0.0

            # print(f'sm_valid:{sm_valid}\n'
            #       f'vpd_valid:{vpd_valid}\n'
            #       f'vpd_lag_valid:{vpd_lag_valid}\n'
            #       f'sm_final:{sm_final}\n'
            #       f'vpd_final:{vpd_final}\n'
            #       f'vpd_lag_final:{vpd_lag_final}')

        return pcorr, p_value
    else:
        # print(f'Mask count is less than 1/5 of sos-pos days, sos-pos days: {len(sm_series)}, valid days: {np.count_nonzero(mask)}')
        return np.nan, np.nan


def compute_partial_correlation_for_pixel_SM_VPD(
    sm_series,
    vpd_series,
    sm_origin_series,
    ta_origin_series,
    srad_origin_series,
    vpd_origin_series,
    pre_origin_series,
):
    """Calculate partial correlation coefficient r and p for a pixel time series"""

    ### Lag pairing
    ### 1. Original lag pairing (existing code)
    sm_lag = sm_series[1:]  # SMi
    vpd = vpd_series[:-1]  # VPDi
    vpd_lag = vpd_series[1:]  # VPDi+1

    sm_origin = sm_origin_series[1:]
    ta_origin = ta_origin_series[1:]
    srad_origin = srad_origin_series[1:]
    vpd_origin = vpd_origin_series[1:]
    pre_origin = pre_origin_series[1:]

    n = len(sm_lag)  # Length after truncation
    #
    # ### 2. Create new consecutive indices (starting from 0)
    #
    # ### 3. Basic validity check
    # base_mask = (
    #         np.isfinite(sm_lag) &
    #         np.isfinite(vpd) &
    #         np.isfinite(vpd_lag)
    # )
    #
    # # Get valid indices
    # valid_indices = np.where(base_mask)[0]
    #
    # if len(valid_indices) == 0:
    #     return np.nan, np.nan
    #
    # ### 4. Extract valid data (using new indices)
    # sm_lag_valid = sm_lag[valid_indices]
    # vpd_valid = vpd[valid_indices]
    # vpd_lag_valid = vpd_lag[valid_indices]
    #
    # # Control variables also use the same indices
    # sm_origin_valid = sm_origin[valid_indices]
    # ta_valid = ta_origin[valid_indices]
    # srad_valid = srad_origin[valid_indices]
    # vpd_origin_valid = vpd_origin[valid_indices]
    # pre_valid = pre_origin[valid_indices]
    #
    # n_valid = len(sm_lag_valid)
    #
    # ### 5. Apply filtering conditions
    # if filter_condition == '1':
    #     # SM decreasing condition
    #     sm_decreasing = np.full(n_valid, False)
    #     if n_valid > 0:
    #         sm_decreasing[0] = True
    #         for t in range(1, n_valid):
    #             sm_decreasing[t] = sm_lag_valid[t] < sm_lag_valid[t - 1]
    #
    #     mask = (
    #             (ta_valid > 5) & (srad_valid > 110) &
    #             (vpd_origin_valid > 0.5) & sm_decreasing
    #     )
    #
    # elif filter_condition == '2':
    #     # SM raw values decreasing
    #     sm_origin_decreasing = np.full(n_valid, False)
    #     if n_valid > 0:
    #         sm_origin_decreasing[0] = True
    #         for t in range(1, n_valid):
    #             sm_origin_decreasing[t] = sm_origin_valid[t] < sm_origin_valid[t - 1]
    #
    #     mask = (
    #             (ta_valid > 5) & (srad_valid > 110) &
    #             (vpd_origin_valid > 0.5) & sm_origin_decreasing
    #     )
    #
    # elif filter_condition == '3':
    #     mask = (
    #             (ta_valid > 5) & (srad_valid > 110) &
    #             (vpd_origin_valid > 0.5) & (sm_lag_valid < 0)
    #     )
    #
    # elif filter_condition == '4':
    #     mask = (
    #             (ta_valid > 5) & (srad_valid > 110) &
    #             (vpd_origin_valid > 0.5)
    #     )

    if filter_condition == "5":
        # Precipitation impact
        pre_origin_series = pd.Series(
            pre_origin
        )  # Convert to Series first, otherwise shift will not work

        # === 1. Define precipitation events ===
        pre_event1 = (pre_origin_series > 0.001) & (
            pre_origin_series <= 0.01
        )  # Light rain
        pre_event2 = pre_origin_series > 0.01  # Moderate to heavy rain

        # === 2. Light rain: Exclude current day only ===
        pre_affected1 = pre_event1.copy()

        # === 3. Moderate to heavy rain: Exclude current day + next n days ===
        pre_affected2 = pre_event2.copy()

        for i in range(1, n_days + 1):
            pre_affected2 |= pre_event2.shift(i, fill_value=False)

        # === 4. Combine all precipitation impacts ===
        pre_affected = (
            pre_affected1 | pre_affected2
        ).to_numpy()  # Combine and convert to numpy array

        # === 5. Final valid data mask ===
        valid_precip_mask = ~pre_affected

        mask = (
            np.isfinite(sm_lag)
            & np.isfinite(vpd)
            & np.isfinite(vpd_lag)
            & valid_precip_mask
        )

    # elif filter_condition == '6':
    #     mask = (sm_lag_valid < 0)
    #
    # elif filter_condition == '7':
    #     mask = np.isfinite(sm_lag_valid)

    ### 6. Get final valid points
    final_indices = np.where(mask)[0]
    n_final = len(final_indices)

    ### 7. Check and calculate
    if n_final > n / 5 and n_final > 3:
        sm_lag_final = sm_lag[final_indices]
        vpd_final = vpd[final_indices]
        vpd_lag_final = vpd_lag[final_indices]

        def safe_corr(a, b):
            valid = np.isfinite(a) & np.isfinite(b)
            if np.sum(valid) < 3:
                return np.nan
            a_valid = a[valid]
            b_valid = b[valid]

            return np.corrcoef(a_valid, b_valid)[0, 1]

        r_xy = safe_corr(sm_lag_final, vpd_lag_final)
        r_xz = safe_corr(vpd_final, sm_lag_final)
        r_yz = safe_corr(vpd_final, vpd_lag_final)

        if np.isnan(r_xy) or np.isnan(r_xz) or np.isnan(r_yz):
            return np.nan, np.nan

        # Calculate partial correlation
        denominator = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
        if denominator == 0:
            return np.nan, np.nan
        else:
            # print(f'r_xy={r_xy} r_xz={r_xz} r_yz={r_yz}')
            pcorr = (r_xy - r_xz * r_yz) / denominator

            if abs(pcorr) < 1.0:
                t_stat = pcorr * np.sqrt((n_final - 3) / (1 - pcorr**2))

                # Use t-distribution to calculate two-tailed p-value
                # Degrees of freedom df = n_final - 3
                from scipy import stats

                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_final - 3))
            else:
                # Set p-value to 0 when |r|=1
                p_value = 0.0

            # print(f'sm_valid:{sm_valid}\n'
            #       f'vpd_valid:{vpd_valid}\n'
            #       f'vpd_lag_valid:{vpd_lag_valid}\n'
            #       f'sm_final:{sm_final}\n'
            #       f'vpd_final:{vpd_final}\n'
            #       f'vpd_lag_final:{vpd_lag_final}')

        return pcorr, p_value
    else:
        # print(f'Mask count is less than 1/5 of sos-pos days, sos-pos days: {len(sm_series)}, valid days: {np.count_nonzero(mask)}')
        return np.nan, np.nan


def process_pixel_sm_vpd_coupling(
    i,
    j,
    year,
    pos,
    sos,
    year_dates,
    sm_data,
    vpd_data,
    pre_origin_data,
):
    # def process_pixel1(i, j, year, pos, dates, sm_data, vpd_data):
    """Function to process a single pixel (for parallel execution)"""
    sos_pixel = sos[i, j]
    pos_pixel = pos[i, j]
    # print(f'sos_pixel:{sos_pixel}, pos_pixel:{pos_pixel}')

    if (
        not pd.isna(pos_pixel)
        and not pd.isna(sos_pixel)
        and not pd.isna(sm_data[0, i, j])
    ):

        valid_indices = extract_time_window(
            year, sos_pixel, pos_pixel, year_dates
        )
        sm_series = sm_data[valid_indices, i, j].flatten()
        vpd_series = vpd_data[valid_indices, i, j].flatten()

        pre_origin_series = pre_origin_data[valid_indices, i, j].flatten()

        # Calculate Pearson correlation coefficient
        if lag_day == 0:
            r, p = compute_pearson_for_pixel(
                sm_series, vpd_series, pre_origin_series
            )
        elif lag_day != 0:
            r, p = compute_pearson_for_pixel(
                sm_series[:-lag_day],
                vpd_series[lag_day:],
                pre_origin_series[:-lag_day],
            )

        # print('Cor:', r)
        # print('SM:', mean1)
        # print('VPD:', mean2)

        return (i, j, r, p)

    else:
        return (i, j, np.nan, np.nan)


def process_pixel2(i, j, year, pos, sos, dates, ta_data, pre_data, srad_data):
    # def process_pixel2(i, j, year, pos, dates, ta_data, pre_data, srad_data):
    """Function to process a single pixel (for parallel execution)"""

    pos_pixel = pos[i, j]
    sos_pixel = sos[i, j]

    if (
        not pd.isna(pos_pixel)
        and not pd.isna(sos_pixel)
        and not pd.isna(ta_data[0, i, j])
    ):
        valid_indices = extract_time_window(year, sos_pixel, pos_pixel, dates)
        # if not pd.isna(pos_pixel) and not pd.isna(ta_data[0, i, j]):
        #     valid_indices = extract_time_window(year, pos_pixel, dates)
        # print('valid_indices:\n', valid_indices)

        # Extract time series data for SM and VPD (shape: [time, 1, 1] → flatten to [time])
        # print('ta_data:\n', np.where(ta_data)[0])
        # print('pre_data:\n', np.where(pre_data)[0])
        # print('srad_data:\n', np.where(srad_data)[0])
        ta_series = ta_data[valid_indices, i, j].flatten()
        pre_series = pre_data[valid_indices, i, j].flatten()
        srad_series = srad_data[valid_indices, i, j].flatten()
        # srad_series = srad_series / (60 * 60 * 24)   # Convert Srad units to daily scale

        mean1 = np.nanmean(ta_series)
        sum1 = np.nansum(pre_series)
        sum2 = np.nansum(srad_series)

        return (i, j, mean1, sum1, sum2)

    else:
        return (i, j, np.nan, np.nan, np.nan)


def save_tif_gdal(output_path, data, crs, transform):
    """Save TIFF file, automatically retrieve data dimensions, and apply geotransform"""
    rows, cols = data.shape
    driver = gdal.GetDriverByName("GTiff")

    output_ds = driver.Create(output_path, cols, rows, 1, gdal.GDT_Float32)
    if not output_ds:
        raise RuntimeError(f"Failed to create output file: {output_path}")

    output_band = output_ds.GetRasterBand(1)
    output_band.WriteArray(data, 0, 0)
    output_band.SetNoDataValue(np.nan)  # Set NaN value

    output_ds.SetProjection(crs)
    output_ds.SetGeoTransform(transform)  # Use adjusted geotransform parameters
    output_ds = None
    return True




##################################### 6 Plot #########################################
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
import matplotlib.gridspec as gridspec

scale = 55

same_input_path = rf'D:\FigShare_data\{scale}km'

cor_mean_path = rf'{same_input_path}\3Cor_mean_slope\mean\SM_VPD_Cor17_8_2\Cor_mean_{scale}km_All.tif'
cor_slope_path = rf'{same_input_path}\3Cor_mean_slope\slope\SM_VPD_Cor17_8_2\Cor_slope_{scale}km_All.tif'
cor_slope_pvalue_path = rf'{same_input_path}\3Cor_mean_slope\pvalue\SM_VPD_Cor17_8_2\Cor_pvalue_{scale}km_All.tif'

veg_type_tif = rf'{same_input_path}\Veg_type\NH_veg_type_{scale}km(Python).tif'

ai_tif = rf'{same_input_path}\AI\NH30_84_AI(graident)_{scale}km.tif'

cor_mean_tif = gdal.Open(cor_mean_path)
cor_mean = cor_mean_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)

cor_slope_tif = gdal.Open(cor_slope_path)
cor_slope = cor_slope_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)

cor_slope_pvalue_tif = gdal.Open(cor_slope_pvalue_path)
cor_slope_pvalue = cor_slope_pvalue_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)

############# 4.2 Veg type ################
veg_type_data = gdal.Open(veg_type_tif)
veg_type_data = veg_type_data.GetRasterBand(1).ReadAsArray().astype(np.float32)

# ############# 4.4 AI ################
ai_type_data = gdal.Open(ai_tif)
ai_type_data = ai_type_data.GetRasterBand(1).ReadAsArray().astype(np.float32)


sample_tif = gdal.Open(cor_mean_path)

if sample_tif is None:
    raise RuntimeError(f"Can not open TIF files：{sample_tif}")

# Get geotransform parameters: projection, pixel size
# Coordinates and projection    Coordinate Reference System (CRS): spatial reference framework for the data
crs = sample_tif.GetProjectionRef()          # Automatically retrieve input CRS
# print('crs:', crs)
gt = sample_tif.GetGeoTransform()  # Geographic coordinates: lat/lon. Mathematical transformation parameters mapping pixel coordinates to real-world geographic coordinates.
proj = sample_tif.GetProjection()  # Projected coordinates: xy (in meters)

# Pixel dimensions
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

# Calculate longitude and latitude bounds (accounts for negative pixel_height)
lon_min = top_left_x
lon_max = top_left_x + cols * pixel_width  # Right boundary longitude
lat_min = top_left_y + rows * pixel_height  # Bottom boundary latitude (southernmost point, potentially smaller value)
lat_max = top_left_y  # Top boundary latitude (northernmost point, potentially larger value)
print(f"Longitude range: {lon_min:.6f} -> {lon_max:.6f}")
print(f"Latitude range: {lat_min:.6f} -> {lat_max:.6f}")




## a left
def plot_cor_mean_or_slope_and_pvalue_forAllvegType(plot_data, plot_data_pvalue, colorbarmin, colorbarmax, data_type, name, ax):
    # # Create 5 subplots: 4 maps + 1 colorbar
    # fig = plt.figure(figsize=(6, 4))
    # gs = gridspec.GridSpec(1, 3,
    #                        width_ratios=[4, 1, 1],  # Width ratios of the three columns
    #                        hspace=0.5, wspace=0.2)

    fig = ax.get_figure()
    if name == 'All':
        ax2_width = 0.8
    else:
        ax2_width = 1
    gs_inner = ax.get_subplotspec().subgridspec(2, 2,
                                                width_ratios=[5, ax2_width],
                                                height_ratios=[5, 0.3],
                                                hspace=0.23, wspace=0.01)

    # Hide parent ax as it serves only as a placeholder container
    ax.axis('off')

    plots = []  # Store plot objects for each subplot

    # ax1 = plt.subplot(gs[0, 0])
    # ax2 = plt.subplot(gs[0, 1])
    # ax3 = plt.subplot(gs[0, 2])
    # ax3 = plt.subplot(gs[0, 2])
    # Create the actual three inner sub-axes
    ax1 = fig.add_subplot(gs_inner[0, 0])  # Map plot
    ax2 = fig.add_subplot(gs_inner[0, 1])  # Latitudinal profile curve
    ax3 = fig.add_subplot(gs_inner[1, :])  # Colorbar spanning across both columns

    # word = 'a'

    ########### Subplot 1: Spatial Distribution #################
    ax1.set_box_aspect(1)  # Force map axis aspect ratio to square so its diameter fills grid height
    ax1.axis('off')
    ### Initialize map
    m = Basemap(ax=ax1,
                projection='npstere',   # North Polar Stereographic projection
                boundinglat=30,         # Lowest displayed latitude (currently 30N)
                lon_0=0,                # Central meridian (adjustable). 180: Pacific centered; 90: Asia centered
                resolution='l')

    # Generate grid coordinates
    lons = np.linspace(-180, 180, cols, endpoint=False)
    lats = np.linspace(lat_max, lat_min, rows)
    lons, lats = np.meshgrid(lons, lats)

    # Configure latitude and longitude grid lines
    m.drawparallels(np.arange(30, 90, 30),
                    labels=[0, 0, 0, 0],
                    linewidth=0.5,
                    # fontsize=8,
                    color="black")

    m.drawmeridians(np.arange(0, 360, 60),
                    latmax=90,  # Convergence of meridians at the North Pole
                    labels=[0, 0, 0, 0],  # labels=[left, right, top, bottom] toggles meridian label visibility
                    linewidth=0.5,
                    # fontsize=8,
                    color="gray")

    # # Fill continent interiors
    # m.fillcontinents(color='white', lake_color='white', zorder=1)

    # map_boundary = m.drawmapboundary(linewidth=0)  # Hide outer boundary line


    ### Render raster data
    # Colormap selection
    if data_type == 'cor slope':
        color_list = ['#67001f', '#b2182b', '#d6604d', '#f4a582', '#fddbc7',
                      '#d1e5f0', '#92c5de', '#4393c3', '#2166ac', '#053061']
        cmap = mpl.colors.ListedColormap(color_list)
        bins = np.linspace(colorbarmin, colorbarmax, 11)
    elif data_type == 'cor mean':
        color_list = ['#a50f15', '#de2d26', '#fb6a4a', '#fc9272',
                      '#fcbba1', '#fee5d9', '#9ecae1']
        cmap = mpl.colors.ListedColormap(color_list)
        bins = np.linspace(colorbarmin, colorbarmax, 8)

    norm = mpl.colors.BoundaryNorm(bins, cmap.N)


    plot = m.pcolormesh(lons, lats, plot_data, cmap=cmap, norm=norm, shading='nearest', latlon=True,
                        zorder=1)  # Prevents tearing artifacts near polar region

    plots.append(plot)  # Save plot reference object

    if data_type == 'cor slope':
        # Overlay statistical significance markers
        significant_mask = (plot_data_pvalue < 0.05) & np.isfinite(plot_data_pvalue) & np.isfinite(plot_data)

        if np.any(significant_mask):
            sig_rows, sig_cols = np.where(significant_mask)
            sig_lons = lons[sig_rows, sig_cols]
            sig_lats = lats[sig_rows, sig_cols]

            # Convert geographic coordinates to map projection coordinates
            sig_x, sig_y = m(sig_lons, sig_lats)

            ax1.scatter(sig_x, sig_y, marker='.', color='black', s=0.5,
                       linewidth=0.1, zorder=2)
    ax1.set_frame_on(False)

    ### Render shapefile boundaries
    terrence = r'D:\CAU\Border\NH_terrence\NH_Terrence_30_90'
    # m.readshapefile(terrence, 'NH_Terrence', drawbounds=True, linewidth=0.2, color='black', zorder=3)
    m.readshapefile(terrence, 'NH_Terrence', drawbounds=False)
    for shape in m.NH_Terrence:
        # Convert coordinate list to numpy array for vector operations
        points = np.array(shape)
        x, y = points[:, 0], points[:, 1]

        # Core logic: calculate projection distance between adjacent points
        # If the distance between two adjacent points spikes on the projection plane, it indicates a circular wraparound loop artifact across the center
        dist = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)

        # Define a distance threshold (projection coordinates are large, typically around 100,000 range)
        # Any step greater than 1/10th of the map diameter is flagged as a projection discontinuity anomaly
        threshold = (ax1.get_xlim()[1] - ax1.get_xlim()[0]) * 0.1

        # Retrieve indices where step distance exceeds threshold
        break_indices = np.where(dist > threshold)[0]

        if len(break_indices) == 0:
            # No jump anomalies found; render complete continuous polyline
            ax1.plot(x, y, color='black', linewidth=0.3, zorder=3)
        else:
            # Discontinuities found; split into distinct line segments to draw separately
            # Removes center-crossing artifact lines while preserving normal border paths
            start_idx = 0
            for break_idx in break_indices:
                ax1.plot(x[start_idx:break_idx + 1], y[start_idx:break_idx + 1],
                         color='black', linewidth=0.3, zorder=3)
                start_idx = break_idx + 1
            # Plot the final segment
            ax1.plot(x[start_idx:], y[start_idx:], color='black', linewidth=0.3, zorder=3)

    # x_pole, y_pole = m(0, 90)
    # ax1.scatter(x_pole, y_pole, color='black', s=5, zorder=3)

    ### Outer boundary circular clipping mask
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
        edgecolor='black',  # Border line color
        linewidth=0.8,
        clip_on=False,
        zorder=4  # Position on top layer
    )

    ax1.add_patch(boundary_circle)

    # if name == 'All':
    #     if data_type == 'cor mean':
    #         ax1.set_title(f'(a)', pad=10, fontweight='bold')
    #     elif data_type == 'cor slope':
    #         ax1.set_title(f'(c)', pad=10, fontweight='bold')
    # else:
    #     if name == 'Forest' or name == 'Arid':
    #         word = 'a'
    #     elif name == 'Shrub' or name == 'Semi-arid':
    #         word = 'b'
    #     elif name == 'Savanna' or name == 'Dry sub-humid':
    #         word = 'c'
    #     elif name == 'Grass' or name == 'Humid':
    #         word = 'd'
    #
    #     ax1.set_title(f'({word}) {name}', pad=10, fontweight='bold')


    ## Descriptive statistics ##
    data_gte0 = plot_data[(plot_data >= 0) & np.isfinite(plot_data)]
    data_lt0 = plot_data[(plot_data < 0) & np.isfinite(plot_data)]
    sum_count = np.sum(np.isfinite(plot_data))

    data_gte0_count = np.sum(np.isfinite(data_gte0))
    data_lt0_count = np.sum(np.isfinite(data_lt0))

    data_gte0_ratio = data_gte0_count / sum_count * 100
    data_lt0_ratio = data_lt0_count / sum_count * 100


    if data_type == 'cor mean':
        h = 0.25
        v = 0.82
        ax1.text(h, v,
                 f'Mean = {np.nanmean(plot_data):.2f}\n'
                 f'Pos = {np.nanmean(data_gte0):.2f} ({data_gte0_ratio:.1f}%)\n'
                 f'Neg = {np.nanmean(data_lt0):.2f} ({data_lt0_ratio:.1f}%)',
                 transform=ax1.transAxes,  # Use normalized relative axes coordinates for positioning
                 multialignment='center',  # Vertical alignment
                 fontsize=6)
    elif data_type == 'cor slope':
        h = 0.21
        v = 0.82
        ax1.text(h, v,
                 f'Mean = {np.nanmean(plot_data):.3f}\n'
                 f'Pos trend = {np.nanmean(data_lt0):.3f} ({data_lt0_ratio:.1f}%)\n'
                 f'Neg trend = {np.nanmean(data_gte0):.3f} ({data_gte0_ratio:.1f}%)',
                 transform=ax1.transAxes,  # Use normalized relative axes coordinates for positioning
                 multialignment='center',  # Vertical alignment
                 fontsize=6)

    ########### Subplot 2: Zonal profile trend across latitudes ###########

    # Use physical latitude values on y-axis
    lat_centers = lats[:, 0]

    plot_data_lat = np.nanmean(plot_data, axis=1)

    if data_type == 'cor slope':
        ax2.axvline(x=0, color='gray', linestyle='--', linewidth=1)

    elif data_type == 'cor mean':
        ax2.axvline(x=-0.3, color='gray', linestyle='--', linewidth=1)

    ax2.plot(plot_data_lat, lat_centers, color='red', linewidth=1, alpha=0.8)

    if data_type == 'cor slope':
        ax2.set_xlim(-0.015, 0.015)
        ax2.set_xticks(np.arange(-0.01, 0.011, 0.01))
        ax2.set_xticklabels(['-1', '0', '1'])  # Manual tick labels

        tick_size = plt.rcParams['xtick.labelsize']
        ax2.text(
            0.98,  # x = tick position (data coordinates)
            -0.02,  # y = offset downward (axes coordinates)
            r'$×10^{-2}$',  # Scientific notation label text
            transform=ax2.transAxes,
            ha='left',  # Expand rightward to prevent clipping
            va='top',
            fontsize=8,
            clip_on=False
        )
    elif data_type == 'cor mean':
        ax2.set_xlim(-0.5, 0)
        ax2.set_xticks(np.arange(-0.3, 0.01, 0.3))
        ax2.set_xticklabels(['-0.3', '0'])  # Manual tick labels

    ax2.set_ylim(30, 90)
    ticks = np.arange(30, 91, 10)
    ax2.set_yticks(ticks)
    ax2.set_yticklabels(f'{x}°' for x in ticks)


    ax2.tick_params(axis='both', which='major', length=2, pad=3)

    ########### Subplot 3: Colorbar ###########
    ### Create Colorbar (occupies bottom position)
    cbar = fig.colorbar(plots[0], cax=ax3, orientation='horizontal')

    cbar.set_ticks(bins)

    if data_type == 'cor slope':
        cbar.set_label('SM-VPD coupling trend (per year)', labelpad=13)
        cbar.set_ticklabels(['0' if x == 0 else
            f'{int(x*100)}' if x*100 == int(x*100) else
            f'{x*100:.1f}'
            for x in bins])
        tick_size = plt.rcParams['xtick.labelsize']
        ax3.text(
            0.9,  # x = tick position (data coordinates)
            -2.05,  # y = offset downward (axes coordinates)
            r'$×10^{-2}$',  # Scientific notation multiplier
            transform=ax3.transAxes,
            ha='left',  # Expand rightward to prevent clipping
            va='top',
            fontsize=8,
            clip_on=False
        )
    if data_type == 'cor mean':
        cbar.set_label('SM-VPD coupling', labelpad=13)
        cbar.set_ticklabels(['0' if x == 0 else
            f'{x:.1f}' for x in bins])

    ax3.tick_params(axis='both', length=2, pad=3)

    plt.tight_layout()

    # Get bounding box position of ax1
    pos1 = ax1.get_position()

    pos2 = ax2.get_position()

    pos3 = ax3.get_position()

    if name == 'All':
        # Adjust ax1 layout
        ax1.set_position([
            pos1.x0 - 0.04,  # Left margin unchanged
            pos2.y0,  # Align bottom with ax2
            pos2.height,
            pos2.height
        ])  # [left, bottom, width, height]
    else:
        if data_type == 'cor mean':
            if name == 'All':
                xpos = 0.105
            else:
                xpos = 0.145
            ax1.set_position([
                pos1.x0 - xpos,  # Left margin unchanged
                pos2.y0,  # Align bottom with ax2
                pos2.height,
                pos2.height
            ])  # [left, bottom, width, height]
        if data_type == 'cor slope':
            ax1.set_position([
                pos1.x0 - 0.145,  # Left margin unchanged
                pos2.y0,  # Align bottom with ax2
                pos2.height,
                pos2.height
            ])  # [left, bottom, width, height]

    # ax2.set_position([
    #     pos2.x0 + 0.055,
    #     pos2.y0,
    #     pos2.width,
    #     pos2.height
    # ])

    pos1_new = ax1.get_position()
    ax3.set_position([
        pos1_new.x0,
        pos3.y0,
        pos2.x1 - pos1_new.x0,
        pos3.height])

    # plt.tight_layout()

    # plt.show()

## Subplot right side panel
def plot_cor_mean_or_slope_lt0_sig_and_lt0_nosig_and_gt0_forAllvegType(plot_data, data_pvalue, data_type, colorbarmin, colorbarmax, name, ax):
    if data_type == 'cor slope':
        ### Data binning and classification
        # Three categorical masks
        plot_data_lt0_mask = plot_data < 0
        plot_data_gte0_mask = plot_data >= 0

        pvalue_sig_mask = data_pvalue <= 0.05
        #
        # plot_data_lt0_sig = np.where(plot_data_lt0_mask & pvalue_sig_mask, plot_data, np.nan)
        # plot_data_lt0_all = np.where(plot_data_lt0_mask, plot_data, np.nan)
        # plot_data_gte0_sig = np.where(plot_data_gte0_mask & pvalue_sig_mask, plot_data, np.nan)
        # plot_data_gte0_all = np.where(plot_data_gte0_mask, plot_data, np.nan)

        plot_data_sig = np.where(pvalue_sig_mask, plot_data, np.nan)

        bins = np.arange(colorbarmin, colorbarmax + 0.006, 0.006)
        # count_lt0_sig, _ = np.histogram(plot_data_lt0_sig, bins=bins)
        # count_lt0_all, _ = np.histogram(plot_data_lt0_all, bins=bins)
        # count_gte0_sig, _ = np.histogram(plot_data_gte0_sig, bins=bins)
        # count_gte0_all, _ = np.histogram(plot_data_gte0_all, bins=bins)

        count_sig, _ = np.histogram(plot_data_sig, bins=bins)
        count_all, _ = np.histogram(plot_data, bins=bins)



    elif data_type == 'cor mean':
        bins = np.arange(colorbarmin, colorbarmax + 0.1, 0.1)
        count_mean, _ = np.histogram(plot_data, bins=bins)

    ### Render plot
    fig = ax.get_figure()
    gs_inner = ax.get_subplotspec().subgridspec(2, 1,
                                                height_ratios=[5, 0.3],
                                                hspace=0.15)

    # Hide parent ax as it serves only as a placeholder container
    ax.axis('off')

    # Configure broken/truncated y-axis
    # if data_type == 'cor slope':
    #     ax.axis('off')
    #     bax = brokenaxes(
    #         ylims=((0, 5000), (10000, 11000)),
    #         hspace=0.1,
    #         height_ratios=[1, 5],  # Upper subplot gets 1 unit, lower gets 2 units. Larger values occupy more layout space
    #         subplot_spec=sub_gs
    #     )
    #
    #     # Configure bar positioning
    #     bin_centers = (bins[:-1] + bins[1:]) / 2
    #     print(f'bin_centers:{bin_centers}')
    #
    #     total_width = 0.007  # Total width allocated for bars per tick interval
    #     n = 2  # Number of categories
    #     width = total_width / n  # Width of an individual bar
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
    #     bax.set_xlabel('VPD-SM coupling trend (per decade)', labelpad=20) # Controls offset spacing between axis label and ticks
    #
    #     bax.set_ylabel('Frequency', labelpad=31)  # Controls offset spacing between axis label and ticks

    #     # Colorbar
    #     bax.legend(
    #         loc='upper right',
    #         bbox_to_anchor=(1.2, 1),
    #         ncol=1,
    #         frameon=False,  # Controls whether to render legend bounding box frame
    #         handlelength=1,
    #         handleheight=1
    #     )


    # Standard continuous (non-broken) y-axis rendering
    if data_type == 'cor slope':

        ax1 = fig.add_subplot(gs_inner[0])

        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        # Configure bar positioning
        bin_centers = (bins[:-1] + bins[1:]) / 2
        print(f'bin_centers:{bin_centers}')

        color_list = ['#67001f', '#b2182b', '#d6604d', '#f4a582', '#fddbc7',
                      '#d1e5f0', '#92c5de', '#4393c3', '#2166ac', '#053061']

        cmap = mpl.colors.ListedColormap(color_list)

        # bins = np.linspace(colorbarmin, colorbarmax, 11)

        norm = mpl.colors.BoundaryNorm(bins, cmap.N)

        bin_colors = cmap(norm(bin_centers))

        total_width = 0.006  # Total width allocated for bars per tick interval
        n = 2  # Number of categories
        width = total_width / n  # Width of an individual bar

        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

        # ax1.bar(bin_centers, count_lt0_sig, width=0.004, linewidth=0.4, hatch='///', facecolor='none', edgecolor='#f46d43', label='Sig-strong', zorder=2)
        # ax1.bar(bin_centers, count_lt0_all, width=0.004, linewidth=0.5, color='#fee090', label='Nonsig-strong', zorder=1)
        # ax1.bar(bin_centers, count_gte0_all, width=0.004, linewidth=0.5, color='#e0f3f8', label='Nonsig-weaken', zorder=1)
        # ax1.bar(bin_centers, count_gte0_sig, width=0.004, linewidth=0.4, hatch='///', facecolor='none', edgecolor='#74add1', label='Sig-weaken', zorder=2)

        # Draw individual bars sequentially to enforce strict color mapping
        for j in range(len(count_all)):
            ax1.bar(
                bin_centers[j],
                count_all[j],
                width=0.004,
                color=bin_colors[j],
                linewidth=0.5,
                zorder=1,
                edgecolor='none'
            )
        ax1.bar(
            bin_centers,
            count_sig,
            width=0.004,
            hatch='/////',
            facecolor='none',
            edgecolor='black',
            linewidth=0.8,
            zorder=2
        )

        ### X-axis settings
        ticks = np.arange(colorbarmin, colorbarmax + 0.006, 0.006)
        ax1.set_xlim(colorbarmin, colorbarmax)
        ax1.set_xticks(ticks)
        ax1.set_xticklabels([
            '0' if np.isclose(x * 100, 0) else
            '3' if np.isclose(x * 100, 3) else
            '-3' if np.isclose(x * 100, -3) else
            f'{x * 100:.1f}'
            for x in ticks
        ], fontsize=8)
        ax1.tick_params(axis='both', length=2, pad=3)
        if name == 'All':
            ax_labelrotation = 45
        else:
            ax_labelrotation = 90
        ax1.tick_params(axis='x', labelrotation=ax_labelrotation)

        tick_size = plt.rcParams['xtick.labelsize']
        if name == 'All':
            x_pos = 0.88
            y_pos = -0.15
        else:
            x_pos = 0.78
            y_pos = -0.15
        ax1.text(
            x_pos,  # x = tick position (data coordinates)
            y_pos,  # y = offset downward (axes coordinates)
            r'$×10^{-2}$',  # Scientific multiplier text
            transform=ax1.transAxes,
            ha='left',  # Expand rightward to prevent clipping
            va='top',
            rotation=0,
            fontsize=8,
            clip_on=False
        )

        # ax1.set_xticklabels(labels, rotation=45)  # Key: Map physical location 0.007 to tick label "0.07"

        if name == 'All':
            ax1.set_ylim(0, 5000)
            ticks = np.arange(0, 5000.1, 1000)

            ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(abs(x))}"))

            ### Y-axis settings
            ax1.set_yticklabels(f'{int(x * 0.001)}' for x in ticks)



        elif name in ['Forests', 'Shrublands', 'Savannas', 'Grasslands']:
            ax1.set_ylim(0, 1500)
            ticks = np.arange(0, 1500.1, 300)
            ax1.set_yticks(ticks)

            ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(abs(x))}"))

            ### Y-axis settings
            ax1.set_yticklabels('0' if x == 0 else
                                f'{x * 0.001:.1f}' for x in ticks)

        elif name in ['Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']:
            ax1.set_ylim(0, 2500)
            ticks = np.arange(0, 2500.1, 500)
            ax1.set_yticks(ticks)

            ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(abs(x))}"))

            ### Y-axis settings
            ax1.set_yticklabels('0' if x == 0 else
                f'{x * 0.001:.1f}' for x in ticks)

        ax1.set_ylabel('Frequency', labelpad=5)  # Controls spacing between label and tick marks

        ax1.text(
            -0.1,  # x = tick position (data coordinates)
            1.12,  # y = offset downward (axes coordinates)
            r'$×10^{3}$',  # Multiplier label
            transform=ax1.transAxes,
            ha='left',  # Expand rightward to prevent clipping
            va='top',
            fontsize=8,
            clip_on=False
        )

        if name == 'All':
            x_position = 0.52
        else:
            x_position = 0.4

        # Legend configuration
        sig_patch = mpatches.Patch(
            facecolor='white',
            edgecolor='black',
            hatch='/////',
            label='Significant'
        )
        ax1.legend(
            handles=[sig_patch],
            loc='lower center',
            bbox_to_anchor=(x_position, -0.4),
            ncol=1,
            frameon=False,  # Toggles visibility of legend border frame
            handlelength=1,
            handleheight=1,
            columnspacing=0.5
        )



    elif data_type == 'cor mean':
        if name == 'All':
            bax = brokenaxes(
                ylims=((0, 4500), (6000, 7000)),
                hspace=0.1,
                height_ratios=[1, 5],  # Upper subplot gets 1 unit, lower gets 5 units. Larger ratio allocates more space
                subplot_spec=gs_inner[0],
                d=0.005
            )

            total_width = 0.2  # Total width allocated for bars per tick interval
            n = 2  # Number of categories
            width = total_width / n  # Width of an individual bar

            bin_centers = (bins[:-1] + bins[1:]) / 2


            color_list = ['#a50f15', '#de2d26', '#fb6a4a', '#fc9272',
                          '#fcbba1', '#fee5d9', '#9ecae1']

            # Render individual bars sequentially to enforce strict color mapping
            for j in range(len(count_mean)):
                bax.bar(bin_centers[j], count_mean[j], width=0.08,
                        color=color_list[j], edgecolor='none')

            # bax.set_xlim(colorbarmin, colorbarmax)
            # bax.set_xticks(np.arange(colorbarmin, colorbarmax, 0.1))
            # bax.axs[1].set_xticklabels(bax.axs[1].get_xticklabels())

            bax.tick_params(axis='both', length=2, pad=3)

            bax.set_xlim(colorbarmin, colorbarmax)
            xticks = np.arange(colorbarmin, colorbarmax + 0.1, 0.1)

            bax.axs[1].set_xticks(xticks)
            bax.axs[1].set_xticklabels(
                ['0' if np.isclose(x, 0) else f'{x:.1f}'
                    for x in xticks],
                rotation=45
            )


            # bax.set_xlabel('VPD-SM coupling', labelpad=20) # Controls spacing between label and tick marks

            bax.axs[0].set_yticks([6000, 7000])
            bax.axs[1].set_yticks([0, 1000, 2000, 3000, 4000])

            for ax in bax.axs:
                # Retrieve current y-axis tick values
                y_ticks = ax.get_yticks()
                # Dynamically generate labels based on y-axis tick values
                ax.set_yticklabels([
                    '0' if y == 0 else
                    f'{int(y * 0.001)}'
                    for y in y_ticks
                ])

            ## Format Y-axis order of magnitude text (10^n multiplier)
            bax.axs[0].text(
                -0.1,  # x = tick position (data coordinates)
                1.65,  # y = offset downward (axes coordinates)
                r'$×10^{3}$',  # Multiplier label
                transform=bax.axs[0].transAxes,
                ha='left',  # Expand rightward to prevent clipping
                va='top',
                rotation=0,
                fontsize=8,
                clip_on=False
            )


            bax.set_ylabel('Frequency', labelpad=15)  # Controls spacing between label and tick marks




        else:
            ax1 = fig.add_subplot(gs_inner[0])

            ax1.spines['top'].set_visible(False)

            ax1.spines['right'].set_visible(False)

            total_width = 0.2  # Total width allocated for bars per tick interval
            n = 2  # Number of categories
            width = total_width / n  # Width of an individual bar

            bin_centers = (bins[:-1] + bins[1:]) / 2

            color_list = ['#a50f15', '#de2d26', '#fb6a4a', '#fc9272',
                          '#fcbba1', '#fee5d9', '#9ecae1']

            # Render individual bars sequentially to enforce strict color mapping
            for j in range(len(count_mean)):
                ax1.bar(bin_centers[j], count_mean[j], width=0.08,
                       color=color_list[j], edgecolor='none')

            ticks = np.arange(colorbarmin, colorbarmax + 0.1, 0.1)
            ax1.set_xlim(colorbarmin, colorbarmax)
            ax1.set_xticks(ticks)
            ax1.set_xticklabels(
                ['0' if np.isclose(x, 0) else f'{x:.1f}'
                 for x in ticks],
                rotation=45
            )

            # ax1.set_xlabel('VPD-SM coupling', labelpad=3)  # Controls spacing between label and tick marks
            if name in ['Forests', 'Shrublands', 'Savannas', 'Grasslands']:
                ax1.set_ylim(0, 2500)
                ticks = np.arange(0, 2500.1, 500)
                ax1.set_yticks(ticks)
            elif name in ['Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']:
                ax1.set_ylim(0, 4000)
                ticks = np.arange(0, 4000.1, 1000)
                ax1.set_yticks(ticks)

            ### Y-axis settings
            ax1.set_yticklabels(f'{x * 0.001:.1f}' for x in ticks)
            tick_size = plt.rcParams['xtick.labelsize']
            if data_type == 'cor mean':
                if name != 'All':
                    ypos = 1.1
            else:
                ypos = 1
            ax1.text(
                -0.1,  # x = tick position (data coordinates)
                ypos,  # y = offset downward (axes coordinates)
                r'$×10^{3}$',  # Multiplier label
                transform=ax1.transAxes,
                ha='left',  # Expand rightward to prevent clipping
                va='top',
                fontsize=8,
                clip_on=False
            )

            ax1.set_ylabel('Frequency', labelpad=3)  # Controls spacing between label and tick marks

            ax1.tick_params(axis='both', length=2, pad=3)


    # if name == 'All':
    #     if data_type == 'cor mean':
    #         bax.set_title(f'(b)', pad=10, fontweight='bold')
    #     elif data_type == 'cor slope':
    #         ax.set_title(f'(d)', pad=10, fontweight='bold')
    # plt.show()

## b
def plot_cor_mean_or_slope_forDiffvegType_and_AI(plot_data, veg_data, ai_data, data_type, ax):

    # 1. Data Preparation
    veg_list = [plot_data[(veg_data == i) & np.isfinite(plot_data)] for i in [1, 2, 3, 4]]
    veg_labels = ['Forests', 'Shrublands', 'Savannas', 'Grasslands']

    ai_list = [
        plot_data[(ai_data == 2) & np.isfinite(plot_data)],  # Arid
        plot_data[((ai_data == 3) | (ai_data == 4)) & np.isfinite(plot_data)],  # Semi-Arid (Merge categories 3 and 4)
        plot_data[(ai_data == 5) & np.isfinite(plot_data)],  # Dry sub-humid
        plot_data[(ai_data == 6) & np.isfinite(plot_data)]  # Humid
    ]
    ai_labels = ['Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']

    full_data = veg_list + ai_list
    x_positions = [1, 2, 3, 4, 5, 6, 7, 8]  # Physical positions on X-axis

    # Fix: Safely calculate mean values
    full_data_mean = [np.mean(d) if len(d) > 0 else 0 for d in full_data]

    ### Plot configuration
    # fig, ax = plt.subplots(figsize=(4, 4))  # Adjust width slightly to avoid cramped text
    # plt.subplots_adjust(bottom=0.2)  # Leave bottom margin for rotated tick labels

    fig = ax.get_figure()

    gs_inner = ax.get_subplotspec().subgridspec(
        2, 1,
        height_ratios=[5, 0.3],
        hspace=0.15
    )

    ax1 = fig.add_subplot(gs_inner[0])
    ax.axis('off')

    # Add showmeans=True for mean indicators
    vio = ax1.violinplot(full_data, positions=x_positions,
                        showmeans=True, showextrema=False)

    # Color Settings
    # v_colors = ['#0ebeff', '#ae63e4', '#ffd200', '#ff3c41',  # Vegetation colors
    #             '#0ebeff', '#ae63e4', '#ffd200', '#ff3c41']  # AI colors (recommended to keep distinct)
    # Retrieve Paired colormap
    paired_colors = plt.cm.Paired(np.linspace(0, 1, 12))  # Extract 8 base colors
    indices = [1, 3, 7, 9]
    colors = [paired_colors[i] for i in indices]
    v_colors = list(colors) + list(colors)

    for i, pc in enumerate(vio['bodies']):
        pc.set_facecolor(v_colors[i])
        pc.set_edgecolor('none')
        # pc.set_linewidth(0.5)
        # pc.set_alpha(0.7)

    # Set distinct colors for mean and median lines
    # Mean Lines (Means) - Red
    vio['cmeans'].set_edgecolor('red')
    vio['cmeans'].set_linestyle('-')
    vio['cmeans'].set_linewidth(1.5)


    # Axis Styling
    ax1.set_xticks(x_positions)
    # if data_type == 'cor mean':
    #     ax1.set_xticklabels([])
    # elif data_type == 'cor slope':
    ax1.set_xticklabels(veg_labels + ai_labels, rotation=90)

    # if data_type == 'cor slope':
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)

    ax1.axvline(4.5, color='black', linestyle='-', linewidth=1)  # Vertical boundary line

    if data_type == 'cor slope':
        ticks = np.arange(-0.03, 0.031, 0.01)
        ax1.set_ylim(-0.03, 0.03)
        ax1.set_yticks(ticks)
        ax1.set_yticklabels('0' if np.isclose(x, 0) else
                            f'{int(round(x*100))}' for x in ticks)
        ax1.set_ylabel('SM-VPD coupling trend (per year)')

        tick_size = plt.rcParams['xtick.labelsize']
        ax1.text(-0.01, 1.06,
                r'$×10^{-2}$',
                transform=ax.transAxes,
                ha='center',
                va='center',
                fontsize=8,
                clip_on=False
                )
    elif data_type == 'cor mean':
        ticks = np.arange(-0.7, 0.3, 0.1)
        ax1.set_ylim(-0.7, 0.2)
        ax1.set_yticks(ticks)
        ax1.set_yticklabels(['0' if np.isclose(x, 0) else
                             f'{x:.1f}' for x in ticks])
        ax1.set_ylabel('SM-VPD coupling')

    for i, m in enumerate(full_data_mean):

        color = v_colors[i]

        if data_type == 'cor mean':

            if i == 0:  # Forests
                yheight = 0.08
            elif i == 1:  # Shrublands
                yheight = 0.15
            elif i == 2:  # Savannas
                yheight = 0.10
            elif i == 3:  # Grasslands
                yheight = 0.05
            elif i == 4:  # Arid
                yheight = 0.03
            elif i == 5:  # Semi-arid
                yheight = 0.05
            elif i == 6:  # Dry sub-humid
                yheight = 0.15
            elif i == 7:  # Humid
                yheight = 0.10

            ax1.text(
                x_positions[i],
                yheight,
                f'{m:.2f}',
                color=color,
                ha='center',
                va='bottom',
                fontsize=6
            )

        elif data_type == 'cor slope':

            if i in [0, 2, 6, 7]:
                yheight = 0.022
            else:
                yheight = 0.027

            if i == 3:
                xheight = -0.2
            elif i == 6:
                xheight = -0.3
            elif i == 4:
                xheight = +0.2
            elif i == 5:
                xheight = +0.4
            else:
                xheight = 0

            ax1.text(
                x_positions[i] + xheight,
                yheight,
                f'{m:.3f}',
                color=color,
                ha='center',
                va='bottom',
                fontsize=6
            )


    ax1.tick_params(axis='both', length=2, pad=3)


    # if data_type == 'cor mean':
    #     ax1.set_title(f'(c)', pad=10, fontweight='bold')
    # elif data_type == 'cor slope':
    #     ax1.set_title(f'(f)', pad=10, fontweight='bold')

    # plt.show()

### All, Fig 4
def plot_fig3(data_mean, data_slope, data_pvalue):

    # Standardize global font settings across plots
    plt.rcParams.update({
        'font.family': 'Arial',

        'mathtext.fontset': 'custom',

        'mathtext.rm': 'Arial',  # Regular
        'mathtext.it': 'Arial:italic',  # Italic
        'mathtext.bf': 'Arial:bold',  # Bold

        # Optional (recommended addition)
        'mathtext.default': 'regular',  # Prevent automatic italicization

        'font.size': 10,
        'axes.titlesize': 10,
        'axes.labelsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        # 'text.usetex': False,  # Disable external LaTeX renderer
    })

    # Create 5 subplots: 4 maps + 1 colorbar
    fig = plt.figure(figsize=(8.2, 6.5))
    gs = gridspec.GridSpec(2, 3,
                           width_ratios=[6, 3.5, 3.5],  # Column width proportions
                           height_ratios=[1, 1],  # Bottom row allocated to colorbar
                           hspace=0.6, wspace=0.33)

    ax1 = plt.subplot(gs[0, 0])  ## Fig a left
    ax2 = plt.subplot(gs[0, 1])  ## Fig a right
    ax3 = plt.subplot(gs[0, 2])  ## Fig b

    ax4 = plt.subplot(gs[1, 0])  ## Fig a left
    ax5 = plt.subplot(gs[1, 1])  ## Fig a right
    ax6 = plt.subplot(gs[1, 2])  ## Fig b

    ### Fig 1a left
    plot_cor_mean_or_slope_and_pvalue_forAllvegType(data_mean, data_pvalue,  -0.6, 0.1,  'cor mean', 'All',ax = ax1)

    ### Fig 1a right
    plot_cor_mean_or_slope_lt0_sig_and_lt0_nosig_and_gt0_forAllvegType(data_mean,  data_pvalue, 'cor mean', -0.6, 0.1, 'All',ax = ax2)

    ### Fig 1b
    plot_cor_mean_or_slope_forDiffvegType_and_AI(data_mean, veg_type_data, ai_type_data, 'cor mean', ax = ax3)

    ### Fig 1c left
    plot_cor_mean_or_slope_and_pvalue_forAllvegType(data_slope, data_pvalue, -0.03, 0.03,'cor slope', 'All',ax = ax4)

    ### Fig 1c right
    plot_cor_mean_or_slope_lt0_sig_and_lt0_nosig_and_gt0_forAllvegType(data_slope, data_pvalue, 'cor slope', -0.03, 0.03, 'All',ax = ax5)

    ### Fig 1d
    plot_cor_mean_or_slope_forDiffvegType_and_AI(data_slope, veg_type_data, ai_type_data, 'cor slope', ax = ax6)

    # plt.savefig(fr'D:\CAU\phenology_swc_vpd\Global_test4\Fig\3Cor_mean_slope\Cor17_8_2_mean_Slope_pvalue.png', dpi=600, bbox_inches='tight')
    plt.savefig(fr'D:\Fig\Fig 3 SM-VPD coupling mean and trend\All\Cor17_8_2_mean_Slope_pvalue.png', dpi=300, bbox_inches='tight')

    # plt.show()

### S9-12
def plot_S9_12_forMean_or_Slope(data_mean, data_slope, data_pvalue, data_type):

    if data_type == 'cor mean':
        data = data_mean
        colorbarmax = 0.1
        colorbarmin = -0.6
    elif data_type == 'cor slope':
        data = data_slope
        colorbarmax = 0.03
        colorbarmin = -0.03

    # 1. Data Preparation
    veg_data_list = [np.where(veg_type_data == i, data, np.nan) for i in [1, 2, 3, 4]]
    veg_pvalue_list = [np.where(veg_type_data == i, data_pvalue, np.nan) for i in [1, 2, 3, 4]]
    veg_labels = ['Forests', 'Shrublands', 'Savannas', 'Grasslands']

    ai_data_list = [
        np.where(ai_type_data == 2, data, np.nan),  # Arid
        np.where((ai_type_data == 3) | (ai_type_data == 4), data, np.nan),  # Semi-Arid (Merge categories 3 and 4)
        np.where(ai_type_data == 5, data, np.nan),  # Dry sub-humid
        np.where(ai_type_data == 6, data, np.nan)  # Humid
    ]

    ai_pvalue_list = [
        np.where(ai_type_data == 2, data_pvalue, np.nan),  # Arid
        np.where((ai_type_data == 3) | (ai_type_data == 4), data_pvalue, np.nan),  # Semi-Arid
        np.where(ai_type_data == 5, data_pvalue, np.nan),  # Dry sub-humid
        np.where(ai_type_data == 6, data_pvalue, np.nan)  # Humid
    ]
    ai_labels = ['Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']

    # Standardize global font settings across plots
    plt.rcParams.update({
        'font.family': 'Arial',

        'mathtext.fontset': 'custom',

        'mathtext.rm': 'Arial',  # Regular
        'mathtext.it': 'Arial:italic',  # Italic
        'mathtext.bf': 'Arial:bold',  # Bold

        # Optional (recommended addition)
        'mathtext.default': 'regular',  # Prevent automatic italicization

        'font.size': 9,
        'axes.titlesize': 9,
        'axes.labelsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        # 'text.usetex': False,  # Disable external LaTeX renderer
    })

    ############################################## Vegetation Types ##################################################
    fig = plt.figure(figsize=(8.4, 6.5))
    gs = gridspec.GridSpec(2, 5,
                           width_ratios=[6, 4, 0.35, 6, 4],  # Column width proportions
                           height_ratios=[1, 1],  # Bottom row allocated to colorbar
                           hspace=0.45, wspace=0.6)

    ax1 = plt.subplot(gs[0, 0])  ## Forests left
    ax2 = plt.subplot(gs[0, 1])  ## Forests right
    ax3 = plt.subplot(gs[0, 3])  ## Shrublands left
    ax4 = plt.subplot(gs[0, 4])  ## Shrublands right

    ax5 = plt.subplot(gs[1, 0])  ## Savannas left
    ax6 = plt.subplot(gs[1, 1])  ## Savannas right
    ax7 = plt.subplot(gs[1, 3])  ## Grasslands left
    ax8 = plt.subplot(gs[1, 4])  ## Grasslands right

    ax_pairs = [(ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)]

    for veg_data, veg_pvalue, veg_name, (ax_l, ax_r) in zip(veg_data_list, veg_pvalue_list, veg_labels, ax_pairs):
        # Render left-side map group (ax_l)
        plot_cor_mean_or_slope_and_pvalue_forAllvegType(veg_data, veg_pvalue, colorbarmin, colorbarmax, data_type, veg_name, ax=ax_l)

        # Render right-side histogram group (ax_r)
        plot_cor_mean_or_slope_lt0_sig_and_lt0_nosig_and_gt0_forAllvegType(veg_data, veg_pvalue, data_type, colorbarmin, colorbarmax, veg_name, ax=ax_r)

    # plt.savefig(fr'D:\CAU\phenology_swc_vpd\Global_test4\Fig\3Cor_mean_slope\Cor17_8_2_{data_type}_Vegtype.png', dpi=600, bbox_inches='tight')
    plt.savefig(fr'D:\Fig\Fig 3 SM-VPD coupling mean and trend\Veg\Cor17_8_2_{data_type}_Vegtype.png', dpi=300, bbox_inches='tight')

    # plt.show()

    ############################################## AI Types ##################################################
    fig = plt.figure(figsize=(8.4, 6.5))
    gs = gridspec.GridSpec(2, 5,
                           width_ratios=[6, 4, 0.8, 6, 4],  # Column width proportions
                           height_ratios=[1, 1],  # Bottom row allocated to colorbar
                           hspace=0.45, wspace=0.5)

    ax1 = plt.subplot(gs[0, 0])  ## Forests left
    ax2 = plt.subplot(gs[0, 1])  ## Forests right
    ax3 = plt.subplot(gs[0, 3])  ## Shrublands left
    ax4 = plt.subplot(gs[0, 4])  ## Shrublands right

    ax5 = plt.subplot(gs[1, 0])  ## Savannas left
    ax6 = plt.subplot(gs[1, 1])  ## Savannas right
    ax7 = plt.subplot(gs[1, 3])  ## Grasslands left
    ax8 = plt.subplot(gs[1, 4])  ## Grasslands right



    ax_pairs = [(ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)]

    for ai_data, ai_pvalue, ai_name, (ax_l, ax_r) in zip(ai_data_list, ai_pvalue_list, ai_labels, ax_pairs):
        # Render left-side map group (ax_l)
        # Note: Assumes function signature was modified to include veg_name
        plot_cor_mean_or_slope_and_pvalue_forAllvegType(ai_data, ai_pvalue, colorbarmin, colorbarmax, data_type, ai_name, ax=ax_l)

        # Render right-side histogram group (ax_r)
        plot_cor_mean_or_slope_lt0_sig_and_lt0_nosig_and_gt0_forAllvegType(ai_data, ai_pvalue, data_type, colorbarmin, colorbarmax, ai_name, ax=ax_r)

    # plt.savefig(fr'D:\CAU\phenology_swc_vpd\Global_test4\Fig\3Cor_mean_slope\Cor17_8_2_{data_type}_AItype.png', dpi=600, bbox_inches='tight')
    plt.savefig(fr'D:\Fig\Fig 3 SM-VPD coupling mean and trend\AI\Cor17_8_2_{data_type}_AItype.png', dpi=300, bbox_inches='tight')

    # # plt.show()


plot_fig3(cor_mean, cor_slope, cor_slope_pvalue)
print('Fig3 plot done!')
plot_S9_12_forMean_or_Slope(cor_mean, cor_slope, cor_slope_pvalue, 'cor mean')
print('S9-10 plot done!')
plot_S9_12_forMean_or_Slope(cor_mean, cor_slope, cor_slope_pvalue, 'cor slope')
print('S11-12 plot done!')


