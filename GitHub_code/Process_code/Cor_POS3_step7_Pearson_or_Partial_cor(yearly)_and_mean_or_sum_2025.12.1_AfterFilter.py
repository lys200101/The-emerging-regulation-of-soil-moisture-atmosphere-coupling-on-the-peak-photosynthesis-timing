import os
import glob
import datetime
import sys

import matplotlib.colors as colors
import matplotlib.plt as plt
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
import pingouin as pg
from scipy.stats import theilslopes


import pymannkendall as mk





######### Function ##########

### Read time information and extract the date from each TIF filename (last 8 digits)
def extract_date_from_filename(filename):
    # Extract basename without path and extension
    basename = os.path.basename(filename)
    filename_without_ext = os.path.splitext(basename)[0]

    # Get the last 8 characters
    date_str = filename_without_ext[-8:]

    # Validate date format
    if not date_str.isdigit() or len(date_str) != 8:
        raise ValueError(f"The last 8 digits of file {filename} is not a valid date (expected YYYYMMDD)!")

    # Convert to datetime object
    return datetime.datetime.strptime(date_str, "%Y%m%d")



### Read SM and VPD bands
def get_band(tif_file, stack):

    tif = gdal.Open(tif_file)

    data = tif.GetRasterBand(1).ReadAsArray().astype(np.float32)

    stack.append(data)

    tif = None
    del data


def cal_pixel_timelength_mean(i, j, data):

    # if len(time_series_clean) > (years_length - 3):
    if len(np.isfinite(data)) > 1:
        result = np.nanmean(data)
        # print(f'pheno mean={result}')
        return (i, j, result)

    else:
        return (i, j, np.nan)



### Extract data within the preseason growing period
def extract_time_window(year, sos, pos, dates):
    # """Extract time window indices corresponding to the year based on pixel's SOS and POS"""
    # Calculate the start and end dates of the growing season for the pixel in the given year

    # print('pos:', pos, flush=True)

    ### Preseason considers SOS-POS
    start_date = pd.to_datetime(f"{year}-01-01") + pd.Timedelta(days=int(pos - interval))
    end_date = pd.to_datetime(f"{year}-01-01") + pd.Timedelta(days=int(pos))


    # ### Preseason considers POS only
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


def compute_pearson_for_pixel(sm_series, vpd_series, sm_origin_series, ta_origin_series, srad_origin_series, vpd_origin_series, pre_origin_series):
    """Calculate Pearson r and p for a pixel's time series"""

    if filter_condition == '1':
        sm_decreasing = np.full(len(sm_series), False)
        sm_decreasing[0] = True  # Cannot compare on the first day, set to True
        for t in range(1, len(sm_series)):
            sm_decreasing[t] = sm_series[t] < sm_series[t - 1]

        mask = (np.isfinite(sm_series) & np.isfinite(vpd_series) &
                (ta_origin_series > 5) & (srad_origin_series > 110) & (vpd_origin_series > 0.5)  & sm_decreasing)

    elif filter_condition == '2':
        sm_decreasing = np.full(len(sm_origin_series), False)
        sm_decreasing[0] = True  # Cannot compare on the first day, set to True
        for t in range(1, len(sm_origin_series)):
            sm_decreasing[t] = sm_origin_series[t] < sm_origin_series[t - 1]

        mask = (np.isfinite(sm_series) & np.isfinite(vpd_series) &
                (ta_origin_series > 5) & (srad_origin_series > 110) & (vpd_origin_series > 0.5) & sm_decreasing)

    elif filter_condition == '3':

        mask = (np.isfinite(sm_series) & np.isfinite(vpd_series) &
                (ta_origin_series > 5) & (srad_origin_series > 110) & (vpd_origin_series > 0.5) & (sm_series < 0))

    elif filter_condition == '4':

        mask = (np.isfinite(sm_series) & np.isfinite(vpd_series) &
                (ta_origin_series > 5) & (srad_origin_series > 110) & (vpd_origin_series > 0.5) )

    elif filter_condition == '5':
        # print(f'{type(sm_series)}\n{type(vpd_series)}\n{type(pre_origin_series)}')
        # print(f'{np.where(sm_series)[0]}\n{np.where(vpd_series)[0]}\n{np.where(pre_origin_series)[0]}')
        # sm_series = sm_series.reset_index(drop=True)
        # vpd_series = vpd_series.reset_index(drop=True)
        # pre_origin_series = pre_origin_series.reset_index(drop=True)

        pre_origin_series = pd.Series(pre_origin_series) # Convert to series first, otherwise shift cannot be executed

        # === 1. Define precipitation events ===
        pre_event1 = (pre_origin_series.notna()) & (pre_origin_series > 0.001) & (pre_origin_series <= 0.01)  # Light rain
        pre_event2 = (pre_origin_series.notna()) & (pre_origin_series > 0.01)  # Moderate-to-heavy rain

        # === 2. Light rain: exclude the current day only ===
        pre_affected1 = pre_event1.copy()

        # === 3. Moderate-to-heavy rain: exclude the current day + subsequent n days ===
        pre_affected2 = pre_event2.copy()
        # print(f'pre_affected2 current day and {n_days} days after not checked: {pre_affected2}')

        for i in range(1, n_days + 1):
            pre_affected2 |= pre_event2.shift(i, fill_value=False)
        # print(f'pre_affected2 current day and {n_days} days after checked: {pre_affected2}')

        # === 4. Combine all precipitation effects ===
        pre_affected = (pre_affected1 | pre_affected2).to_numpy()  # Combine and convert to numpy array

        # === 5. Final valid data ===
        valid_precip_mask = ~pre_affected

        mask = (
                np.isfinite(sm_series) &
                np.isfinite(vpd_series) &
                valid_precip_mask
        )
        # print(f'Unaffected by precipitation: {mask}')
        # print(f'Masked days count: {np.count_nonzero(mask)}')

    elif filter_condition == '6':

        mask = (np.isfinite(sm_series) & np.isfinite(vpd_series) & (sm_series < 0))

    elif filter_condition == '7':

        pre_origin_series = pd.Series(pre_origin_series) # Convert to series first, otherwise shift cannot be executed

        # === 1. Define precipitation events ===
        pre_event1 = (pre_origin_series.notna()) & (pre_origin_series > 0.001) & (pre_origin_series <= 0.01)  # Light rain
        pre_event2 = (pre_origin_series.notna()) & (pre_origin_series > 0.01)  # Moderate-to-heavy rain

        # === 2. Light rain: exclude the current day only ===
        pre_affected1 = pre_event1.copy()

        # === 3. Moderate-to-heavy rain: exclude the current day + subsequent n days ===
        pre_affected2 = pre_event2.copy()
        print(f'pre_affected2 current day and {n_days} days after not checked: {pre_affected2}')

        for i in range(1, n_days + 1):
            pre_affected2 |= pre_event2.shift(i, fill_value=False)
        print(f'pre_affected2 current day and {n_days} days after checked: {pre_affected2}')

        # === 4. Combine all precipitation effects ===
        pre_affected = (pre_affected1 | pre_affected2).to_numpy()  # Combine and convert to numpy array

        # === 5. SManomaly < 0 ===
        sm_mask = sm_series < 0

        # === 5. Final valid data ===
        valid_precip_mask = (~pre_affected)


        mask = (
                np.isfinite(sm_series) &
                np.isfinite(vpd_series) &
                sm_mask &
                valid_precip_mask
        )
        # print(f'Unaffected by precipitation: {mask}')
        # print(f'Masked days count: {np.count_nonzero(mask)}')

    elif filter_condition == '8':
        mask = np.isfinite(sm_series) & np.isfinite(vpd_series)


    if np.count_nonzero(mask) > len(sm_series)/5 and np.count_nonzero(mask) > 2:
        # print(f'sm_series[mask]:{sm_series[mask]}\nvpd_series[mask]:{vpd_series[mask]}')
        r, p = pearsonr(sm_series[mask], vpd_series[mask])
        # print(f'sos-pos days: {len(sm_series)}, valid days: {np.count_nonzero(mask)}, r: {r}')
        # print(f'Precipitation amount: {pre_origin_series}\nSoil moisture: {sm_series[mask]}')
        return r, p
    else:
        # print(f'Mask count is less than half of sos-pos days, sos-pos days: {len(sm_series)}, valid days: {np.count_nonzero(mask)}')
        return np.nan, np.nan



def compute_partial_correlation_for_pixel_SM_VPDlag(sm_series, vpd_series, sm_origin_series, ta_origin_series, srad_origin_series, vpd_origin_series, pre_origin_series):
    """Calculate Pearson r and p for a pixel's time series"""

    ### Lagged pairing
    ### 1. Original lagged pairing (existing code)
    sm = sm_series[:-lag_day].flatten()  # SMi
    vpd = vpd_series[:-lag_day].flatten()  # VPDi
    vpd_lag = vpd_series[lag_day:].flatten()  # VPDi+1

    sm_origin = sm_origin_series[:-lag_day].flatten()
    ta_origin = ta_origin_series[:-lag_day].flatten()
    srad_origin = srad_origin_series[:-lag_day].flatten()
    vpd_origin = vpd_origin_series[:-lag_day].flatten()
    pre_origin = pre_origin_series[:-lag_day].flatten()

    n = len(sm)  # Truncated length

    ### 2. Create new continuous indices (starting from 0)

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
    #     # SM original values decreasing
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

    if filter_condition == '5':

        pre_origin_series = pd.Series(pre_origin)  # Convert to series first, otherwise shift cannot be executed

        # === 1. Define precipitation events ===
        pre_event1 = (pre_origin_series > 0.001) & (pre_origin_series <= 0.01)  # Light rain
        pre_event2 = pre_origin_series > 0.01  # Moderate-to-heavy rain

        # === 2. Light rain: exclude the current day only ===
        pre_affected1 = pre_event1.copy()

        # === 3. Moderate-to-heavy rain: exclude the current day + subsequent n days ===
        pre_affected2 = pre_event2.copy()

        for i in range(1, n_days + 1):
            pre_affected2 |= pre_event2.shift(i, fill_value=False)

        # === 4. Combine all precipitation effects ===
        pre_affected = (pre_affected1 | pre_affected2).to_numpy()  # Combine and convert to numpy array

        # === 5. Final valid data ===
        valid_precip_mask = ~pre_affected

        mask = (
                np.isfinite(sm) &
                np.isfinite(vpd) &
                np.isfinite(vpd_lag) &
                valid_precip_mask
        )


    # elif filter_condition == '6':
    #     mask = (sm_valid < 0)
    #
    # elif filter_condition == '7':
    #     mask = np.isfinite(sm_valid)


    ### 6. Get final valid points
    final_indices = np.where(mask)[0]
    n_final = len(final_indices)

    ### 7. Check and compute
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
        denominator = np.sqrt((1 - r_xz ** 2) * (1 - r_yz ** 2))
        if denominator == 0:
            return np.nan, np.nan
        else:
            # print(f'r_xy={r_xy} r_xz={r_xz} r_yz={r_yz}')
            pcorr = (r_xy - r_xz * r_yz) / denominator

            if abs(pcorr) < 1.0:
                t_stat = pcorr * np.sqrt((n_final - 3) / (1 - pcorr ** 2))

                # Compute two-tailed p-value using t-distribution
                # Degrees of freedom df = n_final - 3
                from scipy import stats
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_final - 3))
            else:
                # When |r|=1, set p-value to 0
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


def compute_partial_correlation_for_pixel_SM_VPD(sm_series, vpd_series, sm_origin_series, ta_origin_series, srad_origin_series, vpd_origin_series, pre_origin_series):
    """Calculate Pearson r and p for a pixel's time series"""

    ### Lagged pairing
    ### 1. Original lagged pairing (existing code)
    sm_lag = sm_series[1:]  # SMi
    vpd = vpd_series[:-1]  # VPDi
    vpd_lag = vpd_series[1:]  # VPDi+1

    sm_origin = sm_origin_series[1:]
    ta_origin = ta_origin_series[1:]
    srad_origin = srad_origin_series[1:]
    vpd_origin = vpd_origin_series[1:]
    pre_origin = pre_origin_series[1:]

    n = len(sm_lag)  # Truncated length
    #
    # ### 2. Create new continuous indices (starting from 0)
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
    #     # SM original values decreasing
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

    if filter_condition == '5':
        # Precipitation effect
        pre_origin_series = pd.Series(pre_origin)  # Convert to series first, otherwise shift cannot be executed

        # === 1. Define precipitation events ===
        pre_event1 = (pre_origin_series > 0.001) & (pre_origin_series <= 0.01)  # Light rain
        pre_event2 = pre_origin_series > 0.01  # Moderate-to-heavy rain

        # === 2. Light rain: exclude the current day only ===
        pre_affected1 = pre_event1.copy()

        # === 3. Moderate-to-heavy rain: exclude the current day + subsequent n days ===
        pre_affected2 = pre_event2.copy()

        for i in range(1, n_days + 1):
            pre_affected2 |= pre_event2.shift(i, fill_value=False)

        # === 4. Combine all precipitation effects ===
        pre_affected = (pre_affected1 | pre_affected2).to_numpy()  # Combine and convert to numpy array

        # === 5. Final valid data ===
        valid_precip_mask = ~pre_affected

        mask = (
                np.isfinite(sm_lag) &
                np.isfinite(vpd) &
                np.isfinite(vpd_lag) &
                valid_precip_mask
        )

    # elif filter_condition == '6':
    #     mask = (sm_lag_valid < 0)
    #
    # elif filter_condition == '7':
    #     mask = np.isfinite(sm_lag_valid)


    ### 6. Get final valid points
    final_indices = np.where(mask)[0]
    n_final = len(final_indices)

    ### 7. Check and compute
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
        denominator = np.sqrt((1 - r_xz ** 2) * (1 - r_yz ** 2))
        if denominator == 0:
            return np.nan, np.nan
        else:
            # print(f'r_xy={r_xy} r_xz={r_xz} r_yz={r_yz}')
            pcorr = (r_xy - r_xz * r_yz) / denominator

            if abs(pcorr) < 1.0:
                t_stat = pcorr * np.sqrt((n_final - 3) / (1 - pcorr ** 2))

                # Compute two-tailed p-value using t-distribution
                # Degrees of freedom df = n_final - 3
                from scipy import stats
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_final - 3))
            else:
                # When |r|=1, set p-value to 0
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


def process_pixel_sm_vpd_coupling(i, j, year, pos, sos, year_dates, sm_data, vpd_data, sm_origin_data, ta_origin_data, srad_origin_data, vpd_origin_data, pre_origin_data):
    """Function to process a single pixel (for parallel execution)"""
    sos_pixel = sos[i, j]
    pos_pixel = pos[i, j]
    # print(f'sos_pixel:{sos_pixel}, pos_pixel:{pos_pixel}')

    if not pd.isna(pos_pixel) and not pd.isna(sos_pixel) and not pd.isna(sm_data[0, i, j]):
        if coupling_method == 'Partial':
            if partial_for == 'SM_VPDlag':
                valid_indices = extract_time_window(year, sos_pixel, pos_pixel, year_dates)
            elif partial_for == 'SMlag_VPDlag':
                valid_indices = extract_time_window(year, sos_pixel-1, pos_pixel, year_dates)

            sm_series = sm_data[valid_indices, i, j].flatten()
            vpd_series = vpd_data[valid_indices, i, j].flatten()

            sm_origin_series = sm_origin_data[valid_indices, i, j].flatten()
            ta_origin_series = ta_origin_data[valid_indices, i, j].flatten()
            srad_origin_series = srad_origin_data[valid_indices, i, j].flatten()
            vpd_origin_series = vpd_origin_data[valid_indices, i, j].flatten()
            pre_origin_series = pre_origin_data[valid_indices, i, j].flatten()

            # Calculate Partial correlation
            if partial_for == 'SM_VPDlag':
                r, p = compute_partial_correlation_for_pixel_SM_VPDlag(sm_series, vpd_series,
                                                                       sm_origin_series, ta_origin_series,
                                                                       srad_origin_series, vpd_origin_series,
                                                                       pre_origin_series)
            elif partial_for == 'SMlag_VPDlag':
                r, p = compute_partial_correlation_for_pixel_SM_VPD(sm_series, vpd_series, sm_origin_series,
                                                                       ta_origin_series, srad_origin_series,
                                                                       vpd_origin_series, pre_origin_series)

            mean1 = np.nanmean(sm_series[lag_day:])
            mean2 = np.nanmean(vpd_series[lag_day:])

        if coupling_method == 'Pearson':
            valid_indices = extract_time_window(year, sos_pixel, pos_pixel, year_dates)
            sm_series = sm_data[valid_indices, i, j].flatten()
            vpd_series = vpd_data[valid_indices, i, j].flatten()

            sm_origin_series = sm_origin_data[valid_indices, i, j].flatten()
            ta_origin_series = ta_origin_data[valid_indices, i, j].flatten()
            srad_origin_series = srad_origin_data[valid_indices, i, j].flatten()
            vpd_origin_series = vpd_origin_data[valid_indices, i, j].flatten()
            pre_origin_series = pre_origin_data[valid_indices, i, j].flatten()

            # Calculate Pearson correlation coefficient
            if lag_day == 0:
                r, p = compute_pearson_for_pixel(sm_series, vpd_series, sm_origin_series, ta_origin_series,
                                                 srad_origin_series, vpd_origin_series, pre_origin_series)
            elif lag_day != 0:
                r, p = compute_pearson_for_pixel(sm_series[:-lag_day], vpd_series[lag_day:], sm_origin_series, ta_origin_series,
                                                 srad_origin_series, vpd_origin_series, pre_origin_series[:-lag_day])

            mean1 = np.nanmean(sm_series)
            mean2 = np.nanmean(vpd_series)

        def cal_daily_diff(sm_series, vpd_series):
            sm_vpd_change = np.full(len(sm_series), np.nan)

            if sum(np.isfinite(sm_series) & np.isfinite(vpd_series))>3:
                for i in range(len(sm_series)):
                    sm_diff = sm_series[i]- sm_series[i-1]
                    vpd_diff = vpd_series[i] - vpd_series[i - 1]
                    if sm_diff < 0 and vpd_diff > 0:
                        sm_vpd_change[i] = 1
                    elif sm_diff > 0 and vpd_diff < 0:
                        sm_vpd_change[i] = 2
                    elif sm_diff < 0 and vpd_diff < 0:
                        sm_vpd_change[i] = 3
                    elif sm_diff > 0 and vpd_diff > 0:
                        sm_vpd_change[i] = 4
                sm_vpd_change_clean = sm_vpd_change[np.isfinite(sm_vpd_change)]

                if len(sm_vpd_change_clean) == 0:
                    sm_vpd_change_mode = np.nan
                else:
                    sm_vpd_change_mode = int(stats.mode(sm_vpd_change_clean).mode)
                # print(f'sm_vpd_change_mode:{sm_vpd_change_mode}')
            elif sum(np.isfinite(sm_series) & np.isfinite(vpd_series)) <= 3:
                sm_vpd_change_mode = np.nan

            return sm_vpd_change_mode

        def cal_period_slope(sm_series, vpd_series):

            time_index = np.arange(len(sm_series))
            sm_slope, sm_intercept, sm_r_value, sm_p_value, sm_std_err = linregress(time_index, sm_series)
            vpd_slope, vpd_intercept, vpd_r_value, vpd_p_value, vpd_std_err = linregress(time_index, vpd_series)

            # Determine trend relationship
            if np.sum(np.isfinite(sm_series) & np.isfinite(vpd_series))>3:
                if sm_p_value < 0.05 and vpd_p_value < 0.05 :  # Both trends are significant
                    if sm_slope < 0 and vpd_slope > 0:
                        trend_rel = 1
                    elif sm_slope > 0 and vpd_slope < 0:
                        trend_rel = 2
                    elif sm_slope > 0 and vpd_slope > 0:
                        trend_rel = 3
                    elif sm_slope < 0 and vpd_slope < 0:
                        trend_rel = 4
                    else:
                        trend_rel = np.nan
                elif (sm_p_value < 0.05 and vpd_p_value >= 0.05) or (sm_p_value >= 0.05 and vpd_p_value < 0.05):
                    trend_rel = 5
                elif sm_p_value >= 0.05 and vpd_p_value >= 0.05:
                    trend_rel = 6
                else:
                    trend_rel = np.nan
                # print(f'trend_rel:{trend_rel}')
            elif np.sum(np.isfinite(sm_series) & np.isfinite(vpd_series)) <= 3:
                trend_rel = np.nan

            return  trend_rel

        if np.count_nonzero(sm_series)>0 and np.count_nonzero(vpd_series)>0:
            # Calculate day-to-day differences to analyze causes of coupling
            result_diff_pattern = cal_daily_diff(sm_series, vpd_series)
            # Calculate slope of SM and VPD within the time period
            result_slope_pattern = cal_period_slope(sm_series, vpd_series)
        else:
            result_diff_pattern = np.nan
            result_slope_pattern = np.nan


        # print('Cor:', r)
        # print('SM:', mean1)
        # print('VPD:', mean2)

        return (i, j, r, p, result_diff_pattern, result_slope_pattern, mean1, mean2)

    else:
        return (i, j, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)




def process_pixel2(i, j, year, pos, sos, dates, ta_data, pre_data, srad_data):
    """Function to process a single pixel (for parallel execution)"""

    pos_pixel = pos[i, j]
    sos_pixel = sos[i, j]

    if not pd.isna(pos_pixel) and not pd.isna(sos_pixel) and not pd.isna(ta_data[0, i, j]):
        valid_indices = extract_time_window(year, sos_pixel, pos_pixel, dates)
    # if not pd.isna(pos_pixel) and not pd.isna(ta_data[0, i, j]):
    #     valid_indices = extract_time_window(year, pos_pixel, dates)
        # print('valid_indices:\n', valid_indices)

        # Extract time series data for SM and VPD (shape: [time, 1, 1] → flattened to [time])
        # print('ta_data:\n', np.where(ta_data)[0])
        # print('pre_data:\n', np.where(pre_data)[0])
        # print('srad_data:\n', np.where(srad_data)[0])
        ta_series = ta_data[valid_indices, i, j].flatten()
        pre_series = pre_data[valid_indices, i, j].flatten()
        srad_series = srad_data[valid_indices, i, j].flatten()
        # srad_series = srad_series / (60 * 60 * 24)   # Convert Srad units to daily scale units

        mean1 = np.nanmean(ta_series)
        sum1 = np.nansum(pre_series)
        sum2 = np.nansum(srad_series)

        return (i, j, mean1, sum1, sum2)

    else:
        return (i, j, np.nan, np.nan, np.nan)


def save_tif_gdal(output_path, data, crs, transform):
    """Save TIFF file, automatically retrieving data dimensions and applying geotransform"""
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
    output_band.SetNoDataValue(np.nan)  # Set NoData value to NaN

    output_ds.SetProjection(crs)
    output_ds.SetGeoTransform(transform)  # Apply adjusted transform parameters
    output_ds = None
    return True




###################################### 1 Data Loading & Output Settings ################################################

###################### ===================== Input Settings ======================== ########################
#### Input SM and VPD tif files    ### Please carefully modify here ⬇⬇⬇⬇⬇⬇⬇⬇

star_year = 2001
end_year = 2024

data_detrend = 'Yes'  ### 'Yes' means using detrended data; 'No' means using raw data

sos = 'SOS'
pos = 'POS'

test_number = '3'  ### Please carefully modify here!!!!!!
if test_number == '1':
    interval = 30
elif test_number == '2':
    interval = 60
elif test_number == '3':
    interval = 90

filter_condition = '5'  ### '1': Ta > 5℃; Srad > 110W/2; VPD > 0.5kPa; day i detrend-SM < day i-1 detrend-SM;
                        ### '2': Ta > 5℃; Srad > 110W/2; VPD > 0.5kPa; day i origin-SM < day i-1 origin-SM
                        ### '3': Ta > 5℃; Srad > 110W/2; VPD > 0.5kPa; day i detrend-SM < 0
                        ### '4': Ta > 5℃; Srad > 110W/2; VPD > 0.5kPa
                        ### '5': Exclude days with Pre > 0.001m and the following day
                        ### '6': Exclude days with SManomaly > 0 (i.e., require detrend-SM < 0)
                        ### '7': Exclude days with 0.01m > Pre > 0.001m, days with Pre > 0.01m + next 7 days; Exclude SManomaly > 0 (i.e., require detrend-SM < 0)
                        ### '8': No filtering conditions
n_days =7 ##### Needs adjustment for precipitation effect on SM when filter_condition = '5'

coupling_method = 'Pearson'  ### 'Pearson'/'Partial'
lag_day = 0

partial_for = 'SM_VPDlag'  ### SM and lagged VPD: SM_VPDlag ; SM and VPD: SMlag_VPDlag

scale = 55

######################## The above parameters need to be carefully checked and modified ########################
years_length = end_year - star_year + 1
print('years_length:', years_length)
years = range(star_year, end_year + 1)

if data_detrend == 'Yes':
    input_path = f'I:\Data\ERA5_Land\ERA5_Land_NH_{scale}km_daily_deseason_detrend'

    folder1 = fr'{input_path}\ERA5_Land_NH_{scale}km_daily_0-100cmSM({star_year}-{end_year})'  ### Please carefully modify here   SM:0-100cmSM  Ta
    folder2 = fr'{input_path}\ERA5_Land_NH_{scale}km_daily_VPD({star_year}-{end_year})'  ### Please carefully modify here   VPD           Pre
    folder3 = fr'{input_path}\ERA5_Land_NH_{scale}km_daily_Srad(2001-2024)'
elif data_detrend == 'No':
    input_path = f'I:\Data\ERA5_Land\ERA5_Land_NH_{scale}km_daily'

    folder1 = fr'{input_path}\ERA5_Land_NH_{scale}km_daily_0-100cmSM'   ### Please carefully modify here   SM:0-100cmSM  Ta
    folder2 = fr'{input_path}\ERA5_Land_NH_{scale}km_daily_VPD'  ### Please carefully modify here   VPD           Pre
    folder3 = fr'{input_path}\ERA5_Land_NH_{scale}km_daily_Srad(2001-2024)'


pos_folder = fr'D:\{pos}_{scale}km'
sos_folder = fr'D:\CAU\phenology_swc_vpd\Global_test6_11000m\Data\Pheno\{sos}_{scale}km'
# sos_folder = 'no'

# ###### Data used for filtering:
# sm_origin_folder = r'I:\Data\ERA5_Land\ERA5_Land_NH_55km_daily\ERA5_Land_NH_55km_daily_0-100cmSM'
# vpd_origin_folder = r'I:\Data\ERA5_Land\ERA5_Land_NH_55km_daily\ERA5_Land_NH_55km_daily_VPD'
# ta_origin_folder = r'I:\Data\ERA5_Land\ERA5_Land_NH_55km_daily\ERA5_Land_NH_55km_daily_Ta'
# srad_origin_folder = r'I:\Data\ERA5_Land\ERA5_Land_NH_55km_daily\ERA5_Land_NH_55km_daily_Srad'
pre_origin_folder = rf'I:\Data\ERA5_Land\ERA5_Land_NH_{scale}km_daily\ERA5_Land_NH_{scale}km_daily_Pre_30_84({star_year}-{end_year})'


###################### ===================== Output Settings ======================== ########################
output_cor_tif_path = fr'D:\Correlation(SM_VPD_pearson){test_number}'
output_sm_tif_path = fr'D:\SM_preseason_mean{test_number}'
output_vpd_tif_path = fr'D:\VPD_preseason_mean{test_number}'
output_ta_tif_path = fr'D:\Ta_preseason_mean{test_number}'
output_pre_tif_path = fr'D:\Pre_preseason_sum{test_number}'
output_srad_tif_path = fr'D:\Srad_preseason_sum{test_number}'

output_cor_mean_slope_tif_path = fr'D:\Result'


#################################################################################################################
####### Whether to calculate Cor
if data_detrend == 'Yes':
    if folder1 == fr'{input_path}\ERA5_Land_NH_{scale}km_daily_0-100cmSM({star_year}-{end_year})':
        calculate_cor = 1  # 1 indicates that Cor needs to be calculated; 0 means Cor is not calculated
    if folder1 == fr'{input_path}\ERA5_Land_NH_{scale}km_daily_Ta(2001-2024)':
        calculate_cor = 0
if data_detrend == 'No':
    if folder1 == fr'{input_path}\ERA5_Land_NH_{scale}km_daily_0-100cmSM':
        calculate_cor = 1  # 1 indicates that Cor needs to be calculated; 0 means Cor is not calculated
    if folder1 == fr'{input_path}\ERA5_Land_NH_{scale}km_daily_Ta(2001-2024)':
        calculate_cor = 0

####################################
tif_files1 = sorted(glob.glob(os.path.join(folder1, '*.tif')))
tif_files2 = sorted(glob.glob(os.path.join(folder2, '*.tif')))
if calculate_cor ==0:
    tif_files3 = sorted(glob.glob(os.path.join(folder3, '*.tif')))