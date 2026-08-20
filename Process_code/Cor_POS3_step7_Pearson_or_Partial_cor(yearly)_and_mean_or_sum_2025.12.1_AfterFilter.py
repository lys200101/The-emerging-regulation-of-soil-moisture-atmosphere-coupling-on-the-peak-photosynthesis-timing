
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
import pingouin as pg
from scipy.stats import theilslopes


import pymannkendall as mk





######### Function ##########

### 读取时间信息，提取每个 TIF 文件的日期（最后8位）
def extract_date_from_filename(filename):
    # 提取纯文件名（不含路径和扩展名）
    basename = os.path.basename(filename)
    filename_without_ext = os.path.splitext(basename)[0]

    # 截取最后8位
    date_str = filename_without_ext[-8:]

    # 验证日期格式
    if not date_str.isdigit() or len(date_str) != 8:
        raise ValueError(f"文件 {filename} 的最后8位不是有效日期（需为YYYYMMDD）！")

    # 转换为 datetime 对象
    return datetime.datetime.strptime(date_str, "%Y%m%d")



### 读取SM和VPD波段
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



### 提取季前生长季内的数据
def extract_time_window(year, sos, pos, dates):
    # """根据像元的sos和pos，提取年份year对应的时间窗口索引"""
    # 计算该像元在年份year的生长季起止日期

    # print('pos：', pos, flush=True)

    ### Preseason考虑SOS-POS
    start_date = pd.to_datetime(f"{year}-01-01") + pd.Timedelta(days=int(pos - interval))
    end_date = pd.to_datetime(f"{year}-01-01") + pd.Timedelta(days=int(pos))


    # ###  Preseason只考虑pos
    # start_date = pd.to_datetime(f"{year}-01-01") + pd.Timedelta(days=int(pos) - 30)
    # end_date = pd.to_datetime(f"{year}-01-01") + pd.Timedelta(days=int(pos))

    # print(f'start_date:{start_date}\n'
    #       f'end_date:{end_date}')


    # 找到时间序列中落在[start_date, end_date]内的索引
    # print('start_date:', start_date, 'end_date:', end_date)
    valid_mask = (dates >= start_date) & (dates < end_date)
    valid_indices = np.where(valid_mask)[0]
    # print('valid_indices:', valid_indices)
    return valid_indices


def compute_pearson_for_pixel(sm_series, vpd_series, sm_origin_series, ta_origin_series, srad_origin_series, vpd_origin_series, pre_origin_series):
    """对一个像元的时间序列计算 Pearson r 和 p"""

    if filter_condition == '1':
        sm_decreasing = np.full(len(sm_series), False)
        sm_decreasing[0] = True  # 第一天无法比较，设为True
        for t in range(1, len(sm_series)):
            sm_decreasing[t] = sm_series[t] < sm_series[t - 1]

        mask = (np.isfinite(sm_series) & np.isfinite(vpd_series) &
                (ta_origin_series > 5) & (srad_origin_series > 110) & (vpd_origin_series > 0.5)  & sm_decreasing)

    elif filter_condition == '2':
        sm_decreasing = np.full(len(sm_origin_series), False)
        sm_decreasing[0] = True  # 第一天无法比较，设为True
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

        pre_origin_series = pd.Series(pre_origin_series) #先转为series，否则shift无法运行

        # === 1. 定义降水事件 ===
        pre_event1 = (pre_origin_series.notna()) & (pre_origin_series > 0.001) & (pre_origin_series <= 0.01)  # 小雨
        pre_event2 = (pre_origin_series.notna()) & (pre_origin_series > 0.01)  # 中大雨

        # === 2. 小雨：只剔除当日 ===
        pre_affected1 = pre_event1.copy()

        # === 3. 中大雨：剔除当日 + 后n天 ===
        pre_affected2 = pre_event2.copy()
        # print(f'pre_affected2未检测当日与{n_days}天后：{pre_affected2}')

        for i in range(1, n_days + 1):
            pre_affected2 |= pre_event2.shift(i, fill_value=False)
        # print(f'pre_affected2检测当日与{n_days}天后：{pre_affected2}')

        # === 4. 合并所有降水影响 ===
        pre_affected = (pre_affected1 | pre_affected2).to_numpy()  #合并，并转为numpy

        # === 5. 最终有效数据 ===
        valid_precip_mask = ~pre_affected

        mask = (
                np.isfinite(sm_series) &
                np.isfinite(vpd_series) &
                valid_precip_mask
        )
        # print(f'不受降水影响：{mask}')
        # print(f'mask天数：{np.count_nonzero(mask)}')

    elif filter_condition == '6':

        mask = (np.isfinite(sm_series) & np.isfinite(vpd_series) & (sm_series < 0))

    elif filter_condition == '7':

        pre_origin_series = pd.Series(pre_origin_series) #先转为series，否则shift无法运行

        # === 1. 定义降水事件 ===
        pre_event1 = (pre_origin_series.notna()) & (pre_origin_series > 0.001) & (pre_origin_series <= 0.01)  # 小雨
        pre_event2 = (pre_origin_series.notna()) & (pre_origin_series > 0.01)  # 中大雨

        # === 2. 小雨：只剔除当日 ===
        pre_affected1 = pre_event1.copy()

        # === 3. 中大雨：剔除当日 + 后n天 ===
        pre_affected2 = pre_event2.copy()
        print(f'pre_affected2未检测当日与{n_days}天后：{pre_affected2}')

        for i in range(1, n_days + 1):
            pre_affected2 |= pre_event2.shift(i, fill_value=False)
        print(f'pre_affected2检测当日与{n_days}天后：{pre_affected2}')

        # === 4. 合并所有降水影响 ===
        pre_affected = (pre_affected1 | pre_affected2).to_numpy()  #合并，并转为numpy

        # === 5. SManomaly<0 ===
        sm_mask = sm_series < 0

        # === 5. 最终有效数据 ===
        valid_precip_mask = (~pre_affected)


        mask = (
                np.isfinite(sm_series) &
                np.isfinite(vpd_series) &
                sm_mask &
                valid_precip_mask
        )
        # print(f'不受降水影响：{mask}')
        # print(f'mask天数：{np.count_nonzero(mask)}')

    elif filter_condition == '8':
        mask = np.isfinite(sm_series) & np.isfinite(vpd_series)


    if np.count_nonzero(mask) > len(sm_series)/5 and np.count_nonzero(mask) > 2:
        # print(f'sm_series[mask]:{sm_series[mask]}\nvpd_series[mask]:{vpd_series[mask]}')
        r, p = pearsonr(sm_series[mask], vpd_series[mask])
        # print(f'sos-pos天数：{len(sm_series)},有效天数：{np.count_nonzero(mask)},r:{r}')
        # print(f'降水量：{pre_origin_series}\n土壤水分：{sm_series[mask]}')
        return r, p
    else:
        # print(f'mask数量少于sos-pos天数的一半，sos-pos天数：{len(sm_series)},有效天数：{np.count_nonzero(mask)}')
        return np.nan, np.nan



def compute_partial_correlation_for_pixel_SM_VPDlag(sm_series, vpd_series, sm_origin_series, ta_origin_series, srad_origin_series, vpd_origin_series, pre_origin_series):
    """对一个像元的时间序列计算 Pearson r 和 p"""

    ### 滞后配对
    ### 1. 原始滞后配对（您已有的代码）
    sm = sm_series[:-lag_day].flatten()  # SMi
    vpd = vpd_series[:-lag_day].flatten()  # VPDi
    vpd_lag = vpd_series[lag_day:].flatten()  # VPDi+1

    sm_origin = sm_origin_series[:-lag_day].flatten()
    ta_origin = ta_origin_series[:-lag_day].flatten()
    srad_origin = srad_origin_series[:-lag_day].flatten()
    vpd_origin = vpd_origin_series[:-lag_day].flatten()
    pre_origin = pre_origin_series[:-lag_day].flatten()

    n = len(sm)  # 截断后的长度

    ### 2. 创建新的连续索引（从0开始）

    # ### 3. 基础有效性检查
    # base_mask = (
    #         np.isfinite(sm) &
    #         np.isfinite(vpd) &
    #         np.isfinite(vpd_lag)
    # )
    #
    # # 获取有效索引
    # valid_indices = np.where(base_mask)[0]
    #
    # if len(valid_indices) == 0:
    #     return np.nan, np.nan
    #
    # ### 4. 提取有效数据（使用新索引）
    # sm_valid = sm[valid_indices]
    # vpd_valid = vpd[valid_indices]
    # vpd_lag_valid = vpd_lag[valid_indices]
    #
    # # 控制变量也使用相同的索引
    # sm_origin_valid = sm_origin[valid_indices]
    # ta_valid = ta_origin[valid_indices]
    # srad_valid = srad_origin[valid_indices]
    # vpd_origin_valid = vpd_origin[valid_indices]
    # pre_valid = pre_origin[valid_indices]
    #
    # n_valid = len(sm_valid)

    ### 5. 应用筛选条件
    # if filter_condition == '1':
    #     # SM递减条件
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
    #     # SM原始值递减
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

        pre_origin_series = pd.Series(pre_origin)  # 先转为series，否则shift无法运行

        # === 1. 定义降水事件 ===
        pre_event1 = (pre_origin_series > 0.001) & (pre_origin_series <= 0.01)  # 小雨
        pre_event2 = pre_origin_series > 0.01  # 中大雨

        # === 2. 小雨：只剔除当日 ===
        pre_affected1 = pre_event1.copy()

        # === 3. 中大雨：剔除当日 + 后n天 ===
        pre_affected2 = pre_event2.copy()

        for i in range(1, n_days + 1):
            pre_affected2 |= pre_event2.shift(i, fill_value=False)

        # === 4. 合并所有降水影响 ===
        pre_affected = (pre_affected1 | pre_affected2).to_numpy()  # 合并，并转为numpy

        # === 5. 最终有效数据 ===
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


    ### 6. 获取最终有效点
    final_indices = np.where(mask)[0]
    n_final = len(final_indices)

    ### 7. 检查并计算
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

        # 计算偏相关
        denominator = np.sqrt((1 - r_xz ** 2) * (1 - r_yz ** 2))
        if denominator == 0:
            return np.nan, np.nan
        else:
            # print(f'r_xy={r_xy} r_xz={r_xz} r_yz={r_yz}')
            pcorr = (r_xy - r_xz * r_yz) / denominator

            if abs(pcorr) < 1.0:
                t_stat = pcorr * np.sqrt((n_final - 3) / (1 - pcorr ** 2))

                # 使用t分布计算双尾p值
                # 自由度 df = n_final - 3
                from scipy import stats
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_final - 3))
            else:
                # 当|r|=1时，p值设为0
                p_value = 0.0

            # print(f'sm_valid:{sm_valid}\n'
            #       f'vpd_valid:{vpd_valid}\n'
            #       f'vpd_lag_valid:{vpd_lag_valid}\n'
            #       f'sm_final:{sm_final}\n'
            #       f'vpd_final:{vpd_final}\n'
            #       f'vpd_lag_final:{vpd_lag_final}')

        return pcorr, p_value
    else:
        # print(f'mask数量少于sos-pos天数的1/5，sos-pos天数：{len(sm_series)},有效天数：{np.count_nonzero(mask)}')
        return np.nan, np.nan


def compute_partial_correlation_for_pixel_SM_VPD(sm_series, vpd_series, sm_origin_series, ta_origin_series, srad_origin_series, vpd_origin_series, pre_origin_series):
    """对一个像元的时间序列计算 Pearson r 和 p"""

    ### 滞后配对
    ### 1. 原始滞后配对（您已有的代码）
    sm_lag = sm_series[1:]  # SMi
    vpd = vpd_series[:-1]  # VPDi
    vpd_lag = vpd_series[1:]  # VPDi+1

    sm_origin = sm_origin_series[1:]
    ta_origin = ta_origin_series[1:]
    srad_origin = srad_origin_series[1:]
    vpd_origin = vpd_origin_series[1:]
    pre_origin = pre_origin_series[1:]

    n = len(sm_lag)  # 截断后的长度
    #
    # ### 2. 创建新的连续索引（从0开始）
    #
    # ### 3. 基础有效性检查
    # base_mask = (
    #         np.isfinite(sm_lag) &
    #         np.isfinite(vpd) &
    #         np.isfinite(vpd_lag)
    # )
    #
    # # 获取有效索引
    # valid_indices = np.where(base_mask)[0]
    #
    # if len(valid_indices) == 0:
    #     return np.nan, np.nan
    #
    # ### 4. 提取有效数据（使用新索引）
    # sm_lag_valid = sm_lag[valid_indices]
    # vpd_valid = vpd[valid_indices]
    # vpd_lag_valid = vpd_lag[valid_indices]
    #
    # # 控制变量也使用相同的索引
    # sm_origin_valid = sm_origin[valid_indices]
    # ta_valid = ta_origin[valid_indices]
    # srad_valid = srad_origin[valid_indices]
    # vpd_origin_valid = vpd_origin[valid_indices]
    # pre_valid = pre_origin[valid_indices]
    #
    # n_valid = len(sm_lag_valid)
    #
    # ### 5. 应用筛选条件
    # if filter_condition == '1':
    #     # SM递减条件
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
    #     # SM原始值递减
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
        # 降水影响
        pre_origin_series = pd.Series(pre_origin)  # 先转为series，否则shift无法运行

        # === 1. 定义降水事件 ===
        pre_event1 = (pre_origin_series > 0.001) & (pre_origin_series <= 0.01)  # 小雨
        pre_event2 = pre_origin_series > 0.01  # 中大雨

        # === 2. 小雨：只剔除当日 ===
        pre_affected1 = pre_event1.copy()

        # === 3. 中大雨：剔除当日 + 后n天 ===
        pre_affected2 = pre_event2.copy()

        for i in range(1, n_days + 1):
            pre_affected2 |= pre_event2.shift(i, fill_value=False)

        # === 4. 合并所有降水影响 ===
        pre_affected = (pre_affected1 | pre_affected2).to_numpy()  # 合并，并转为numpy

        # === 5. 最终有效数据 ===
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


    ### 6. 获取最终有效点
    final_indices = np.where(mask)[0]
    n_final = len(final_indices)

    ### 7. 检查并计算
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

        # 计算偏相关
        denominator = np.sqrt((1 - r_xz ** 2) * (1 - r_yz ** 2))
        if denominator == 0:
            return np.nan, np.nan
        else:
            # print(f'r_xy={r_xy} r_xz={r_xz} r_yz={r_yz}')
            pcorr = (r_xy - r_xz * r_yz) / denominator

            if abs(pcorr) < 1.0:
                t_stat = pcorr * np.sqrt((n_final - 3) / (1 - pcorr ** 2))

                # 使用t分布计算双尾p值
                # 自由度 df = n_final - 3
                from scipy import stats
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_final - 3))
            else:
                # 当|r|=1时，p值设为0
                p_value = 0.0

            # print(f'sm_valid:{sm_valid}\n'
            #       f'vpd_valid:{vpd_valid}\n'
            #       f'vpd_lag_valid:{vpd_lag_valid}\n'
            #       f'sm_final:{sm_final}\n'
            #       f'vpd_final:{vpd_final}\n'
            #       f'vpd_lag_final:{vpd_lag_final}')

        return pcorr, p_value
    else:
        # print(f'mask数量少于sos-pos天数的1/5，sos-pos天数：{len(sm_series)},有效天数：{np.count_nonzero(mask)}')
        return np.nan, np.nan


def process_pixel_sm_vpd_coupling(i, j, year, pos, sos, year_dates, sm_data, vpd_data, sm_origin_data, ta_origin_data, srad_origin_data, vpd_origin_data, pre_origin_data):
    """处理单个像元的函数（供并行调用）"""
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

            # 计算Partial correlation
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

            # 计算Pearson相关系数
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

            # 判断趋势关系
            if np.sum(np.isfinite(sm_series) & np.isfinite(vpd_series))>3:
                if sm_p_value < 0.05 and vpd_p_value < 0.05 :  # 两个趋势都显著
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
            # 计算天与天之间差异，统计耦合发生原因
            result_diff_pattern = cal_daily_diff(sm_series, vpd_series)
            # 计算时间段内 SM和VPD的slope
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
    """处理单个像元的函数（供并行调用）"""

    pos_pixel = pos[i, j]
    sos_pixel = sos[i, j]

    if not pd.isna(pos_pixel) and not pd.isna(sos_pixel) and not pd.isna(ta_data[0, i, j]):
        valid_indices = extract_time_window(year, sos_pixel, pos_pixel, dates)
    # if not pd.isna(pos_pixel) and not pd.isna(ta_data[0, i, j]):
    #     valid_indices = extract_time_window(year, pos_pixel, dates)
        # print('valid_indices:\n', valid_indices)

        # 提取SM和VPD的时间序列数据（形状：[time, 1, 1] → 展平为[time]）
        # print('ta_data:\n', np.where(ta_data)[0])
        # print('pre_data:\n', np.where(pre_data)[0])
        # print('srad_data:\n', np.where(srad_data)[0])
        ta_series = ta_data[valid_indices, i, j].flatten()
        pre_series = pre_data[valid_indices, i, j].flatten()
        srad_series = srad_data[valid_indices, i, j].flatten()
        # srad_series = srad_series / (60 * 60 * 24)   #对Srad进行单位换算，变成日尺度的单位

        mean1 = np.nanmean(ta_series)
        sum1 = np.nansum(pre_series)
        sum2 = np.nansum(srad_series)

        return (i, j, mean1, sum1, sum2)

    else:
        return (i, j, np.nan, np.nan, np.nan)


def save_tif_gdal(output_path, data, crs, transform):
    """保存TIFF文件，自动获取数据尺寸并应用地理变换"""
    rows, cols = data.shape
    driver = gdal.GetDriverByName("GTiff")

    output_ds = driver.Create(
        output_path,
        cols, rows, 1, gdal.GDT_Float32
    )
    if not output_ds:
        raise RuntimeError(f"无法创建输出文件: {output_path}")

    output_band = output_ds.GetRasterBand(1)
    output_band.WriteArray(data, 0, 0)
    output_band.SetNoDataValue(np.nan)  # 设置NaN值

    output_ds.SetProjection(crs)
    output_ds.SetGeoTransform(transform)  # 使用调整后的变换参数
    output_ds = None
    return True




###################################### 1 数据读取及输出设定 ################################################

###################### ===================== 输入设定 ======================== ########################
#### 输入的SM和VPD的tif    ### 请仔细修改这里 ⬇⬇⬇⬇⬇⬇⬇⬇

star_year = 2001
end_year = 2024

data_detrend = 'Yes'  ###'Yes'表示使用去趋势数据；'No'表示使用原始数据

sos = 'SOS'
pos = 'POS'

test_number = '3'  ###请仔细修改这里!!!!!!
if test_number == '1':
    interval = 30
elif test_number == '2':
    interval = 60
elif test_number == '3':
    interval = 90

filter_condition = '5'  ###'1':Ta＞5℃；Srad>110W/2；VPD>0.5kPa；第i天detrend-SM<第i-1天detrend-SM；
                        ###'2':Ta＞5℃；Srad>110W/2；VPD>0.5kPa；第i天origin-SM<第i-1天origin-SM
                        ###'3':Ta＞5℃；Srad>110W/2；VPD>0.5kPa；第i天detrend-SM<0
                        ###'4':Ta＞5℃；Srad>110W/2；VPD>0.5kPa
                        ###'5':剔除Pre>0.001m当天与后一日
                        ###'6':剔除SManomaly当天>0(即要求detrend-SM<0）
                        ###'7':剔除0.01m>Pre>0.001m当天，Pre>0.01m当天与后7日；剔除SManomaly>0(即要求detrend-SM<0）
                        ###'8':无筛选条件
n_days =7 #####当filter_condition = '5'需要调整降水对SM的影响

coupling_method = 'Pearson'  ### 'Pearson'/'Partial'
lag_day = 0

partial_for = 'SM_VPDlag'  ###SM与滞后的VPD：SM_VPDlag  ； SM与VPD：SMlag_VPDlag

scale = 55

########################以上部分需要仔细核对修改########################
years_length = end_year - star_year + 1
print('years_length:', years_length)
years = range(star_year, end_year + 1)

if data_detrend == 'Yes':
    input_path = f'I:\Data\ERA5_Land\ERA5_Land_NH_{scale}km_daily_deseason_detrend'

    folder1 = fr'{input_path}\ERA5_Land_NH_{scale}km_daily_0-100cmSM({star_year}-{end_year})'  ### 请仔细修改这里   SM:0-100cmSM  Ta
    folder2 = fr'{input_path}\ERA5_Land_NH_{scale}km_daily_VPD({star_year}-{end_year})'  ### 请仔细修改这里   VPD           Pre
    folder3 = fr'{input_path}\ERA5_Land_NH_{scale}km_daily_Srad(2001-2024)'
elif data_detrend == 'No':
    input_path = f'I:\Data\ERA5_Land\ERA5_Land_NH_{scale}km_daily'

    folder1 = fr'{input_path}\ERA5_Land_NH_{scale}km_daily_0-100cmSM'   ### 请仔细修改这里   SM:0-100cmSM  Ta
    folder2 = fr'{input_path}\ERA5_Land_NH_{scale}km_daily_VPD'  ### 请仔细修改这里   VPD           Pre
    folder3 = fr'{input_path}\ERA5_Land_NH_{scale}km_daily_Srad(2001-2024)'


pos_folder = fr'D:\{pos}_{scale}km'
sos_folder = fr'D:\CAU\phenology_swc_vpd\Global_test6_11000m\Data\Pheno\{sos}_{scale}km'
# sos_folder = 'no'

# ###### 用于筛选的数据：
# sm_origin_folder = r'I:\Data\ERA5_Land\ERA5_Land_NH_55km_daily\ERA5_Land_NH_55km_daily_0-100cmSM'
# vpd_origin_folder = r'I:\Data\ERA5_Land\ERA5_Land_NH_55km_daily\ERA5_Land_NH_55km_daily_VPD'
# ta_origin_folder = r'I:\Data\ERA5_Land\ERA5_Land_NH_55km_daily\ERA5_Land_NH_55km_daily_Ta'
# srad_origin_folder = r'I:\Data\ERA5_Land\ERA5_Land_NH_55km_daily\ERA5_Land_NH_55km_daily_Srad'
pre_origin_folder = rf'I:\Data\ERA5_Land\ERA5_Land_NH_{scale}km_daily\ERA5_Land_NH_{scale}km_daily_Pre_30_84({star_year}-{end_year})'


###################### ===================== 输出设定 ======================== ########################
output_cor_tif_path = fr'D:\Correlation(SM_VPD_pearson){test_number}'
output_sm_tif_path = fr'D:\SM_preseason_mean{test_number}'
output_vpd_tif_path = fr'D:\VPD_preseason_mean{test_number}'
output_ta_tif_path = fr'D:\Ta_preseason_mean{test_number}'
output_pre_tif_path = fr'D:\Pre_preseason_sum{test_number}'
output_srad_tif_path = fr'D:\Srad_preseason_sum{test_number}'

output_cor_mean_slope_tif_path = fr'D:\Result'


#################################################################################################################
####### 是否计算Cor
if data_detrend == 'Yes':
    if folder1 == fr'{input_path}\ERA5_Land_NH_{scale}km_daily_0-100cmSM({star_year}-{end_year})':
        calculate_cor = 1  # 1是表示需要计算Cor，0则不计算Cor
    if folder1 == fr'{input_path}\ERA5_Land_NH_{scale}km_daily_Ta(2001-2024)':
        calculate_cor = 0
if data_detrend == 'No':
    if folder1 == fr'{input_path}\ERA5_Land_NH_{scale}km_daily_0-100cmSM':
        calculate_cor = 1  # 1是表示需要计算Cor，0则不计算Cor
    if folder1 == fr'{input_path}\ERA5_Land_NH_{scale}km_daily_Ta(2001-2024)':
        calculate_cor = 0

####################################
tif_files1 = sorted(glob.glob(os.path.join(folder1, '*.tif')))
tif_files2 = sorted(glob.glob(os.path.join(folder2, '*.tif')))
if calculate_cor ==0:
    tif_files3 = sorted(glob.glob(os.path.join(folder3, '*.tif')))

pos_tif_files = sorted(glob.glob(os.path.join(pos_folder, '*.tif')))
if sos_folder != 'no':
    sos_tif_files = sorted(glob.glob(os.path.join(sos_folder, '*.tif')))

# sm_origin_files = sorted(glob.glob(os.path.join(sm_origin_folder, '*.tif')))
# vpd_origin_files = sorted(glob.glob(os.path.join(vpd_origin_folder, '*.tif')))
# ta_origin_files = sorted(glob.glob(os.path.join(ta_origin_folder, '*.tif')))
# srad_origin_files = sorted(glob.glob(os.path.join(srad_origin_folder, '*.tif')))
pre_origin_files = sorted(glob.glob(os.path.join(pre_origin_folder, '*.tif')))

print('files done!')

if not tif_files1:
    raise FileNotFoundError("未找到任何 tif_files1 TIF 文件！")
    raise FileNotFoundError("未找到任何 tif_files1 TIF 文件！")
if not tif_files2:
    raise FileNotFoundError("未找到任何 tif_files2 TIF 文件！")
if not pos_tif_files:
    raise FileNotFoundError("未找到任何 POS TIF 文件！")



####################### 2 提取tif信息 ############################
first_tif = tif_files1[0]
sample_tif = gdal.Open(first_tif)

if sample_tif is None:
    raise RuntimeError(f"无法打开 TIF 文件：{sample_tif}（驱动不支持或文件损坏）")

# 获取地理变换参数：投影、像素大小
#坐标和投影         坐标参考系：即数据所在的空间参考框架
crs = sample_tif.GetProjectionRef()          # 自动获取输入的 CRS
# print('crs:', crs)
gt = sample_tif.GetGeoTransform()  #地理坐标：经纬度。将像素坐标转换为实际地理坐标的数学变换参数。
proj = sample_tif.GetProjection()  #投影坐标：xy（单位m）

#像素
pixel_width = gt[1]
pixel_height = gt[5]

top_left_x = gt[0]
top_left_y = gt[3]

#行列数
sample_tif = sample_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)

rows = sample_tif.shape[0]
cols = sample_tif.shape[1]
print('rows:', rows, 'cols:', cols)

row_indices = np.repeat(np.arange(rows), cols)  # 行索引重复cols次
col_indices = np.tile(np.arange(cols), rows)  # 列索引平铺rows次

# 计算经纬度范围（修正 pixel_height 为负的情况）
lon_min = top_left_x
lon_max = top_left_x + cols * pixel_width  # 右边界经度
lat_min = top_left_y + rows * pixel_height  # 下边界纬度（最南端，可能更小）
lat_max = top_left_y  # 上边界纬度（最北端，可能更大）
print(f"经度范围: {lon_min:.6f} -> {lon_max:.6f}")
print(f"纬度范围: {lat_min:.6f} -> {lat_max:.6f}")


############################################ 3 时间-堆叠 ###################################################
## 时间
tif_dates = []
for tif_file in tif_files1:
    date = extract_date_from_filename(tif_file)
    tif_dates.append(date)

print('前五个日期：', [d.strftime('%Y%m%d') for d in tif_dates[:5]])

print('All stack start!')

## 数据堆叠
stack1 = []
stack2 = []
stack3 = []

pos_stack = []
sos_stack = []

# sm_origin_stack = []
# vpd_origin_stack = []
# ta_origin_stack = []
# srad_origin_stack = []
pre_origin_stack = []

if data_detrend == 'No':
    for tif_file in tif_files1:
        get_band(tif_file, stack1)

    for tif_file in tif_files2:
        get_band(tif_file, stack2)

elif data_detrend == 'Yes':
    for tif_file in tif_files1:
        get_band(tif_file, stack1)

    print('SM stack done!')

    for tif_file in tif_files2:
        get_band(tif_file, stack2)

    print('VPD stack done!')


for tif_file in pos_tif_files:
    get_band(tif_file, pos_stack)
print('POS stack done!')

if sos_folder != 'no':
    for tif_file in sos_tif_files:
        get_band(tif_file, sos_stack)
    print('SOS stack done!')
# for tif_file in sm_origin_files:
#     get_band_clip(tif_file, sm_origin_stack, row_start, row_end)
# for tif_file in vpd_origin_files:
#     get_band_clip(tif_file, vpd_origin_stack, row_start, row_end)
# for tif_file in ta_origin_files:
#     get_band_clip(tif_file, ta_origin_stack, row_start, row_end)
# for tif_file in srad_origin_files:
#     get_band_clip(tif_file, srad_origin_stack, row_start, row_end)

for tif_file in pre_origin_files:
    # get_band_clip(tif_file, pre_origin_stack, row_start, row_end)
    get_band(tif_file, pre_origin_stack)
print('Pre stack done!')

stack1 = np.stack(stack1, axis=0)#[:, 505:510, 505:510]
stack2 = np.stack(stack2, axis=0)#[:, 505:510, 505:510]
if calculate_cor == 0:

    for tif_file in tif_files3:
        get_band(tif_file, stack3)

    stack3 = np.stack(stack3, axis=0)

pos_stack = np.stack(pos_stack, axis=0)#[:, 505:510, 505:510]

if sos_folder != 'no':
    sos_stack = np.stack(sos_stack, axis=0)#[:, 505:510, 505:510]
print('pos_stack shape:\n', pos_stack.shape)


# sm_origin_stack = np.stack(sm_origin_stack, axis=0)
# vpd_origin_stack = np.stack(vpd_origin_stack, axis=0)
# ta_origin_stack = np.stack(ta_origin_stack, axis=0)
# srad_origin_stack = np.stack(srad_origin_stack, axis=0)
pre_origin_stack = np.stack(pre_origin_stack, axis=0)

days = pre_origin_stack.shape[0]
sm_origin_stack = np.full_like(pre_origin_stack, np.nan)
vpd_origin_stack = np.full_like(pre_origin_stack, np.nan)
ta_origin_stack = np.full_like(pre_origin_stack, np.nan)
srad_origin_stack = np.full_like(pre_origin_stack, np.nan)


print('All stack done!')


######################################### 5 季前生长季均值计算 ###########################################
###### 先去异常值
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
    # print(rf'len(x_flatten):{len(x_flatten)}')
    if len(x_flatten) < time_lengths/2:
        # 如果没有有效数据，直接返回全NaN
        return np.full_like(x, np.nan), i, j
    else:
        # remove Outliers
        upper_quartile, lower_quartile = np.percentile(x_flatten, [qmax, qmin])
        IQR = (upper_quartile - lower_quartile)

        # from scipy.stats import iqr
        # x1= x.copy()
        # x1= np.where(x!=fillvalue, x1, np.nan)
        # IQR = iqr(x1, nan_policy='omit')

        lower_range = lower_quartile - (1.5 * IQR)
        upper_range = upper_quartile + (1.5 * IQR)

        # maxv = np.max(x_flatten)
        # minv = np.min(x_flatten)
        valid_mask = np.logical_and(x <= upper_range, x >= lower_range)
        x_masked = np.where(valid_mask, x, np.nan)

        if (len(np.isfinite(x_masked)) > (years_length/2)) & (len(np.isfinite(x_masked)) <= years_length):
            # print(f'去异常值后无效像元<3个')
            return x_masked, i, j  # IQR, lower_range, upper_range,  minv, maxv
        else:
            # print(f'IQR threshold：{lower_range:.4f} ~ {upper_range:.4f}\n'
            #       f'像元的原始数据:{x}\n'
            # print(f'去异常值后无效像元超3个')
            return np.full_like(x, np.nan), i, j



time_lengths = pos_stack.shape[0]
print(f'time_lengths:{time_lengths}')

outlier_pos_stack = np.full((time_lengths, rows, cols), np.nan)
outlier_sos_stack = np.full((time_lengths, rows, cols), np.nan)


dates = pd.to_datetime(tif_dates)
print('前五个日期：', [d.strftime('%Y%m%d') for d in dates[:5]])

if calculate_cor == 1:

    cor_each_year = []
    p_each_year = []

    cor_matrix = np.full((years_length, rows, cols), np.nan)  # 3维数组    , dtype=object
    p_matrix = np.full((years_length, rows, cols), np.nan)  # 3维数组   , dtype=object

    cor_pattern_byDiff = np.full((years_length, rows, cols), np.nan)
    cor_pattern_bySlope = np.full((years_length, rows, cols), np.nan)


var1_mean = np.full((years_length, rows, cols), np.nan)
var2_mean = np.full((years_length, rows, cols), np.nan)
var1_sum = np.full((years_length, rows, cols), np.nan)
var2_sum = np.full((years_length, rows, cols), np.nan)



for year in years:
    print(f"正在处理年份：{year}")

    k = year - star_year

    year_mask = (dates >= f"{year}-01-01") & (dates <= f"{year}-12-31")

    ## 用顺序索引
    year_idx = np.where(year_mask)[0]
    # print('year_idx:\n', year_idx)
    # 这一年的日期与数据切片（注意：这里不涉及 reset_index）
    year_dates = dates[year_idx]
    stack1_year = stack1[year_idx, :, :]
    stack2_year = stack2[year_idx, :, :]
    if calculate_cor == 0:
        stack3_year = stack3[year_idx, :, :]

    sm_origin_stack_year = sm_origin_stack[year_idx, :, :]
    vpd_origin_stack_year = vpd_origin_stack[year_idx, :, :]
    ta_origin_stack_year = ta_origin_stack[year_idx, :, :]
    srad_origin_stack_year = srad_origin_stack[year_idx, :, :]
    pre_origin_stack_year = pre_origin_stack[year_idx, :, :]


    pos_year = pos_stack[k, :, :]  # 你的 pos_stack 已按 year 索引
    if sos_folder != 'no':
        sos_year = sos_stack[k, :, :]  # 你的 pos_stack 已按 year 索引


    ############ 并行处理

    if calculate_cor == 1:

        print(f'year={year}pixel Pearson cor and mean calculate start')
        #beijing_time = datetime.now(timezone(timedelta(hours=8)))
        #print(f'北京时间：{beijing_time}')

        # 并行处理所有像元（使用所有CPU核心）
        if sos_folder != 'no':
            with parallel_backend("threading", n_jobs=18):
                results = Parallel()(
                    delayed(process_pixel_sm_vpd_coupling)(
                        i, j, year, pos_year, sos_year, year_dates,
                        stack1_year, stack2_year,
                        sm_origin_stack_year, ta_origin_stack_year, srad_origin_stack_year, vpd_origin_stack_year, pre_origin_stack_year
                    )
                    for i, j in zip(row_indices, col_indices)
                )
        if sos_folder == 'no':
            with parallel_backend("threading", n_jobs=18):
                results = Parallel()(
                    delayed(process_pixel2)(
                        i, j, year, pos_year, year_dates, stack1_year, stack2_year
                    )
                    for i, j in zip(row_indices, col_indices)
                )
        print(f'year={year}pixel Pearson cor, sm mean and vpd mean calculate end')


        for i, j, r, p, result_diff_pattern, result_slope_pattern, mean1, mean2 in results:
            cor_matrix[k, i, j] = r
            p_matrix[k, i, j] = p
            cor_pattern_byDiff[k, i, j] = result_diff_pattern
            cor_pattern_bySlope[k, i, j] = result_slope_pattern
            var1_mean[k, i, j] = mean1
            var2_mean[k, i, j] = mean2


        print(f"{year}年耦合效应、SM、VPD结果已准备好")

    if calculate_cor == 0:
        print(f'year={year}pixel mean and sum calculate start')
        # 并行处理所有像元（使用所有CPU核心）
        if sos_folder != 'no':
            with parallel_backend("threading", n_jobs=18):
                results = Parallel(verbose=10)(
                    delayed(process_pixel2)(
                        i, j, year, pos_year, sos_year, year_dates, stack1_year, stack2_year, stack3_year
                    )
                    for i, j in zip(row_indices, col_indices)
                )
        if sos_folder == 'no':
            with parallel_backend("threading", n_jobs=18):
                results = Parallel(verbose=10)(
                    delayed(process_pixel2)(
                        i, j, year, pos_year, year_dates, stack1_year, stack2_year, stack3_year
                    )
                    for i, j in zip(row_indices, col_indices)
                )
        print(f'year={year}pixel mean and sum calculate end')

        for i, j, mean1, sum1, sum2 in results:
            var1_mean[k, i, j] = mean1
            var1_sum[k, i, j] = sum1
            var2_sum[k, i, j] = sum2

        output_path1 = os.path.join(output_ta_tif_path, f"Ta_pearson_mean_{year}.tif")
        save_tif_gdal(
            output_path1,
            var1_mean[k, :, :],
            crs, gt
        )

        output_path2 = os.path.join(output_pre_tif_path, f"Pre_pearson_sum_{year}.tif")
        save_tif_gdal(
            output_path2,
            var1_sum[k, :, :],
            crs, gt
        )

        output_path3 = os.path.join(output_srad_tif_path, f"Srad_pearson_sum_{year}.tif")
        save_tif_gdal(
            output_path3,
            var2_sum[k, :, :],
            crs, gt
        )

        print(f"{year}年Ta、Pre、Srad 结果已保存")




########################## Save cor ###############################
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
import matplotlib.gridspec as gridspec



def plot_cor_mean_combine_forvegType(data, colorbarmin, colorbarmax):
    # 创建5个子图：4个地图 + 1个colorbar
    fig = plt.figure(figsize=(14, 14.2))
    gs = gridspec.GridSpec(7, 3,
                           width_ratios=[7, 0.5, 1],  #三列的宽度比
                           height_ratios=[1, 1, 1, 1, 1, 1, 0.1], # 最后一个给colorbar
                           hspace=0.5, wspace=0.2)

    # 统一设置所有字体大小
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 13,
        'axes.titlesize': 13,
        'axes.labelsize': 13,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 13,
    })

    # 定义四种植被类型
    veg_types = ['All', 'Forest', 'Shrub', 'Savanna', 'Grass', 'Wet']
    titles = ['All Vegetation Types', 'Forest', 'Shrub', 'Savanna', 'Grass', 'Wet']

    plots = []  # 存储每个子图的plot对象

    for i, (veg_type, title) in enumerate(zip(veg_types, titles)):
        ax1 = plt.subplot(gs[i, 0])
        ax2 = plt.subplot(gs[i, 1])
        ax3 = plt.subplot(gs[i, 2])

        # 创建植被类型掩码
        if veg_type == 'All':
            plot_data = data
        elif veg_type == 'Forest':
            mask = (veg_type_data == 1)
            plot_data = np.where(mask, data, np.nan)
        elif veg_type == 'Shrub':
            mask = (veg_type_data == 2)
            plot_data = np.where(mask, data, np.nan)
        elif veg_type == 'Savanna':
            mask = (veg_type_data == 3)
            plot_data = np.where(mask, data, np.nan)
        elif veg_type == 'Grass':
            mask = (veg_type_data == 4)
            plot_data = np.where(mask, data, np.nan)
        elif veg_type == 'Wet':
            mask = (veg_type_data == 5)
            plot_data = np.where(mask, data, np.nan)


        ########### 子图1：空间分布
        ### 创建地图
        m = Basemap(ax=ax1, projection='cyl', resolution='l',
                    llcrnrlon=lon_min, llcrnrlat=lat_min,
                    urcrnrlon=lon_max, urcrnrlat=lat_max)

        # 生成网格坐标
        lons = np.linspace(lon_min, lon_max, cols)
        lats = np.linspace(lat_max, lat_min, rows)
        lons, lats = np.meshgrid(lons, lats)
        X, Y = m(lons, lats)

        ### 设置经纬度刻度
        m.drawparallels(np.arange(30, 90, 20), dashes=[0, 1])
        m.drawmeridians(np.arange(-180, 180, 30), dashes=[0, 1])

        # 设置刻度标签
        xticks = []
        xlabels = []
        for lon in range(-180, 181, 30):
            if lon < 0:
                xlabels.append(f'{abs(lon)}°W')
            elif lon > 0:
                xlabels.append(f'{lon}°E')
            else:
                xlabels.append('0°')
            xticks.append(lon)

        color_list = ['#a50f15', '#de2d26', '#fb6a4a', '#fc9272',
                      '#fcbba1', '#fee5d9', '#9ecae1']

        cmap = mpl.colors.ListedColormap(color_list)
        bins = np.linspace(colorbarmin, colorbarmax, 8)
        norm = mpl.colors.BoundaryNorm(bins, cmap.N)

        # 绘制数据
        yticks = np.arange(30, 90, 20)

        ax1.set_xticks(xticks)
        ax1.set_yticks(yticks)
        ax1.set_xticklabels(xlabels, rotation=0)
        ax1.set_yticklabels([f'{y}°N' for y in yticks])

        ax1.xaxis.set_ticks_position('top')
        ax1.yaxis.set_ticks_position('left')
        ax1.tick_params(axis='both', which='major',
                        length = 2, pad=3,
                        top=True, bottom=False, left=True, right=False)

        ### 绘制边界
        terrence = r'D:\CAU\phenology_swc_vpd\Global_test\制图\border\NH_Terrence'
        m.readshapefile(terrence, 'NH Terrence', drawbounds=True, linewidth=0.2, color='black', zorder=3)

        ### 颜色映射
        # color_list = plt.cm.RdBu(np.linspace(0, 1, 9))

        plot = m.pcolormesh(X, Y, plot_data, cmap=cmap, norm=norm, zorder=1)
        plots.append(plot)  # 保存plot对象

        # 添加显著性标记 #

        ########### 子图2：逐纬度变化趋势
        plot_data_slope_lat = np.nanmean(plot_data, axis=1)

        # 使用实际纬度值作为y轴
        lat_centers = np.linspace(lat_max, lat_min, rows)

        ax2.plot(plot_data_slope_lat, lat_centers, color='red', linewidth=1, alpha=0.8)
        ax2.axvline(x=0, color='gray', linestyle='--', linewidth=1)

        if veg_type == 'Wet':
            ax2.set_xlabel('SM-VPD correlation')

        ax2.set_xlim(-0.6, 0.3)
        ax2.set_xticks(np.arange(-0.3, 0.301, 0.3))
        ax2.set_xticklabels(['-0.3', '0', '0.3'])  # 手动设置标签


        ax2.set_ylim(30, int(lat_max)+1)
        ax2.set_yticks(np.arange(30, int(lat_max)+1, 10))

        ax2.tick_params(axis='both', which='major', length = 2, pad=3)

        ax2.set_aspect(0.035)  # 数值越小，图形越宽；数值越大，图形越高

        ########### 子图3：频率分布图
        ###### 生成频率柱状图
        plot_data_vaild = plot_data[np.isfinite(plot_data)]

        hist, bin_edges = np.histogram(plot_data_vaild, bins=bins,density=False)  # density=True 自动计算密度, bins=bins,
        print(f'Veg_type:{veg_type}\n'
              f'bin_edges:{bin_edges}\n'
              f'Density:{hist}')

        fraction = (hist/len(plot_data_vaild))*100
        print(f'fraction:{fraction}')

        bar_width = 1  # 设置柱子的宽度

        x_positions = np.arange(len(hist))

        ax3.bar(x_positions, fraction, width=bar_width, color=color_list)

        # 设置x轴刻度
        ax3.set_xticks(x_positions)  # 设置x轴刻度
        ax3.tick_params(axis='x', which='both',
                        length = 2, pad=3,
                        top=False, bottom=False, labelbottom=False)  # 刻度线长度设为0

        ax3.set_ylim(0, 45)
        ax3.set_yticks(np.arange(0, 45.01, 15))

        ax3.tick_params(axis='y')

        # 设置y轴
        ax3.set_ylabel('Fraction(%)')  # 显示密度而非频率

        # 隐藏边框，保留左边框
        for spine in ax3.spines.values():
            spine.set_visible(False)
        ax3.spines['left'].set_visible(True)

        # 占比标注（所有+显著）
        # 计算正负值和零值的统计
        total_count = len(plot_data_vaild)
        count_positive = np.count_nonzero(plot_data_vaild > 0)
        count_negative = np.count_nonzero(plot_data_vaild < 0)
        count_zero = np.count_nonzero(plot_data_vaild == 0)

        percent_positive = (count_positive / total_count) * 100
        percent_negative = (count_negative / total_count) * 100
        percent_zero = (count_zero / total_count) * 100


        # 计算mean
        mean_val = np.mean(plot_data_vaild)

        stats_text  = (f'Cor>0: {percent_positive:.1f}%\n'
                      f'Cor<0: {percent_negative:.1f}%\n'
                      f'Cor=0: {percent_zero:.1f}%')

        ax3.text(0.05, 1.25, stats_text,
                 transform=ax3.transAxes, fontsize=8,
                 verticalalignment='top')

        ax3.set_aspect(0.18)  # 数值越小，图形越宽；数值越大，图形越高


        # 设置子图标题
        ax1.set_title(f'{title}(Mean = {mean_val:.2f})')

        ### 导出分植被类型的tif结果
        tif_output_path1 = os.path.join(output_cor_mean_slope_tif_path, rf'mean\SM_VPD_Cor{test_number}\Cor_mean_{scale}km_{veg_type}.tif')

        save_tif_gdal(
            tif_output_path1,
            plot_data,
            crs,
            gt  # 使用新的地理变换参数
        )


    ### 生成Colorbar（使用最后一个位置）
    cbar_ax = plt.subplot(gs[-1, :])
    cbar = fig.colorbar(plots[0], cax=cbar_ax, orientation='horizontal')

    cbar.set_label('SM-VPD correlation')


    # cbar.set_ticks(np.arange(colorbarmin, colorbarmax+0.01, 0.2))
    cbar.set_ticklabels([f'{x:.1f}' for x in bins])

    # 手动调整colorbar位置使其居中
    cbar.ax.set_position([0.18, 0.13, 0.4, 0.02])  # [left, bottom, width, height]

    plt.tight_layout()

    # plt.show()

    # 保存图片
    fig_path = os.path.join(output_cor_mean_slope_png_path, rf'mean\Cor17(Filter_precipitation)\Cor{test_number}_mean_{scale}km_vegType.png')


    plt.savefig(fig_path, dpi=600, bbox_inches='tight')
    print('Plot done!')


def plot_cor_mean_combine_forKB(data, colorbarmin, colorbarmax):
    # 创建5个子图：4个地图 + 1个colorbar
    fig = plt.figure(figsize=(12, 8.2))
    gs = gridspec.GridSpec(5, 3,
                           width_ratios=[7, 0.5, 1],  # 三列的宽度比
                           height_ratios=[1, 1, 1, 1, 0.1],  # 最后一个给colorbar
                           hspace=0.5, wspace=0.2)

    # 统一设置所有字体大小
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 13,
        'axes.titlesize': 13,
        'axes.labelsize': 13,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 13,
    })

    # 定义四种植被类型
    # 定义四种植被类型
    # KB_types = ['Arid', 'Temperate', 'Continental', 'Polar']
    # titles = ['Arid region', 'Temperate region', 'Continental region', 'Polar region']
    KB_types = ['All', 'Arid', 'Temperate', 'Continental']
    titles = ['All climate zones', 'Arid zone', 'Temperate zone', 'Continental zone']

    plots = []  # 存储每个子图的plot对象

    for i, (KB_type, title) in enumerate(zip(KB_types, titles)):
        ax1 = plt.subplot(gs[i, 0])
        ax2 = plt.subplot(gs[i, 1])
        ax3 = plt.subplot(gs[i, 2])

        # 创建植被类型掩码
        if KB_type == 'All':
            plot_data = data
        elif KB_type == 'Arid':
            mask = (climate_type == 2)
            plot_data = np.where(mask, data, np.nan)
        elif KB_type == 'Arid':
            mask = (climate_type == 2)
            plot_data = np.where(mask, data, np.nan)
        elif KB_type == 'Temperate':
            mask = (climate_type == 3)
            plot_data = np.where(mask, data, np.nan)
        elif KB_type == 'Continental':
            mask = (climate_type == 4)
            plot_data = np.where(mask, data, np.nan)
        # elif KB_type == 'Polar':
        #     mask = (climate_type == 5)
        #     plot_data = np.where(mask, data, np.nan)


        ########### 子图1：空间分布
        ### 创建地图
        m = Basemap(ax=ax1, projection='cyl', resolution='l',
                    llcrnrlon=lon_min, llcrnrlat=lat_min,
                    urcrnrlon=lon_max, urcrnrlat=lat_max)

        # 生成网格坐标
        lons = np.linspace(lon_min, lon_max, cols)
        lats = np.linspace(lat_max, lat_min, rows)
        lons, lats = np.meshgrid(lons, lats)
        X, Y = m(lons, lats)

        ### 设置经纬度刻度
        m.drawparallels(np.arange(30, 90, 20), dashes=[0, 1])
        m.drawmeridians(np.arange(-180, 180, 30), dashes=[0, 1])

        # 设置刻度标签
        xticks = []
        xlabels = []
        for lon in range(-180, 181, 30):
            if lon < 0:
                xlabels.append(f'{abs(lon)}°W')
            elif lon > 0:
                xlabels.append(f'{lon}°E')
            else:
                xlabels.append('0°')
            xticks.append(lon)
        yticks = np.arange(30, 90, 20)

        ax1.set_xticks(xticks)
        ax1.set_yticks(yticks)
        ax1.set_xticklabels(xlabels, rotation=0)
        ax1.set_yticklabels([f'{y}°N' for y in yticks])

        ax1.xaxis.set_ticks_position('top')
        ax1.yaxis.set_ticks_position('left')
        ax1.tick_params(axis='both', which='major',
                        length = 2, pad=3,
                        top=True, bottom=False, left=True, right=False)

        ### 绘制边界
        terrence = r'D:\CAU\phenology_swc_vpd\Global_test\制图\border\NH_Terrence'
        m.readshapefile(terrence, 'NH Terrence', drawbounds=True, linewidth=0.2, color='black', zorder=3)

        ### 颜色映射
        # color_list = plt.cm.RdBu(np.linspace(0, 1, 9))
        color_list = ['#a50f15', '#de2d26', '#fb6a4a', '#fc9272',
                      '#fcbba1', '#fee5d9', '#9ecae1']

        cmap = mpl.colors.ListedColormap(color_list)
        bins = np.linspace(colorbarmin, colorbarmax, 8)
        norm = mpl.colors.BoundaryNorm(bins, cmap.N)

        # 绘制数据
        plot = m.pcolormesh(X, Y, plot_data, cmap=cmap, norm=norm, zorder=1)
        plots.append(plot)  # 保存plot对象

        # 添加显著性标记 #

        ########### 子图2：逐纬度变化趋势
        plot_data_slope_lat = np.nanmean(plot_data, axis=1)

        # 使用实际纬度值作为y轴
        lat_centers = np.linspace(lat_max, lat_min, rows)

        ax2.plot(plot_data_slope_lat, lat_centers, color='red', linewidth=1, alpha=0.8)
        ax2.axvline(x=0, color='gray', linestyle='--', linewidth=1)

        if KB_type == 'Continental':
            ax2.set_xlabel('SM-VPD correlation')

        ax2.set_xlim(-0.6, 0.3)
        ax2.set_xticks(np.arange(-0.3, 0.301, 0.3))
        ax2.set_xticklabels(['-0.3', '0', '0.3'])  # 手动设置标签


        ax2.set_ylim(30, int(lat_max)+1)
        ax2.set_yticks(np.arange(30, int(lat_max)+1, 10))

        ax2.tick_params(axis='both', which='major', length = 2, pad=3)

        ax2.set_aspect(0.034)  # 数值越小，图形越宽；数值越大，图形越高

        ########### 子图3：频率分布图
        ###### 生成频率柱状图
        plot_data_vaild = plot_data[np.isfinite(plot_data)]

        hist, bin_edges = np.histogram(plot_data_vaild, bins=bins,density=False)  # density=True 自动计算密度, bins=bins,
        print('Density:', hist)
        print('bin_edges:', bin_edges)

        fraction = (hist/len(plot_data_vaild))*100
        print(f'fraction:{fraction}')

        bar_width = 1  # 设置柱子的宽度

        x_positions = np.arange(len(hist))

        ax3.bar(x_positions, fraction, width=bar_width, color=color_list)

        # 设置x轴刻度
        ax3.set_xticks(x_positions)  # 设置x轴刻度
        ax3.tick_params(axis='x', which='both',
                        length = 2, pad=3,
                        top=False, bottom=False, labelbottom=False)  # 刻度线长度设为0

        ax3.set_ylim(0, 45)
        ax3.set_yticks(np.arange(0, 45.01, 15))

        ax3.tick_params(axis='y')

        # 设置y轴
        ax3.set_ylabel('Fraction(%)')  # 显示密度而非频率

        # 隐藏边框，保留左边框
        for spine in ax3.spines.values():
            spine.set_visible(False)
        ax3.spines['left'].set_visible(True)

        # 占比标注（所有+显著）
        # 计算正负值和零值的统计
        total_count = len(plot_data_vaild)
        count_positive = np.count_nonzero(plot_data_vaild > 0)
        count_negative = np.count_nonzero(plot_data_vaild < 0)
        count_zero = np.count_nonzero(plot_data_vaild == 0)

        percent_positive = (count_positive / total_count) * 100
        percent_negative = (count_negative / total_count) * 100
        percent_zero = (count_zero / total_count) * 100


        # 计算mean
        mean_val = np.mean(plot_data_vaild)

        stats_text  = (f'Cor>0: {percent_positive:.1f}%\n'
                      f'Cor<0: {percent_negative:.1f}%\n'
                      f'Cor=0: {percent_zero:.1f}%')

        ax3.text(0.05, 1.25, stats_text,
                 transform=ax3.transAxes, fontsize=8,
                 verticalalignment='top')

        ax3.set_aspect(0.17)  # 数值越小，图形越宽；数值越大，图形越高


        # 设置子图标题
        ax1.set_title(f'{title}(Mean = {mean_val:.2f})')

        ### 导出分植被类型的tif结果
        tif_output_path1 = os.path.join(output_cor_mean_slope_tif_path, rf'mean\SM_VPD_Cor{test_number}\Cor_mean_{scale}km_KB({KB_type}).tif')

        save_tif_gdal(
            tif_output_path1,
            plot_data,
            crs,
            gt  # 使用新的地理变换参数
        )


    ### 生成Colorbar（使用最后一个位置）
    cbar_ax = plt.subplot(gs[-1, :])
    cbar = fig.colorbar(plots[0], cax=cbar_ax, orientation='horizontal')

    cbar.set_label('SM-VPD correlation')


    # cbar.set_ticks(np.arange(colorbarmin, colorbarmax+0.01, 0.2))
    cbar.set_ticklabels([f'{x:.1f}' for x in bins])

    # 手动调整colorbar位置使其居中
    cbar.ax.set_position([0.18, 0.13, 0.4, 0.018])  # [left, bottom, width, height]

    plt.tight_layout()

    # plt.show()

    # 保存图片
    fig_path = os.path.join(output_cor_mean_slope_png_path, rf'mean\Cor17(Filter_precipitation)\Cor{test_number}_mean_{scale}km_KB.png')


    plt.savefig(fig_path, dpi=600, bbox_inches='tight')
    print('Plot done!')




def plot_cor_slope_and_pvalue_combine_forvegType(data_slope, data_pvalue, colorbarmin, colorbarmax):
    # 创建5个子图：4个地图 + 1个colorbar
    fig = plt.figure(figsize=(14, 14.2))
    gs = gridspec.GridSpec(7, 3,
                           width_ratios=[7, 0.5, 1],  #三列的宽度比
                           height_ratios=[1, 1, 1, 1, 1, 1, 0.1], # 最后一个给colorbar
                           hspace=0.5, wspace=0.18)

    # 统一设置所有字体大小
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 13,
        'axes.titlesize': 13,
        'axes.labelsize': 13,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 13,
    })

    # 定义四种植被类型
    veg_types = ['All', 'Forest', 'Shrub', 'Savanna', 'Grass', 'Wet']
    titles = ['All Vegetation Types', 'Forest', 'Shrub', 'Savanna', 'Grass', 'Wet']

    plots = []  # 存储每个子图的plot对象

    for i, (veg_type, title) in enumerate(zip(veg_types, titles)):
        ax1 = plt.subplot(gs[i, 0])
        ax2 = plt.subplot(gs[i, 1])
        ax3 = plt.subplot(gs[i, 2])

        # 创建植被类型掩码
        if veg_type == 'All':
            plot_data_slope = data_slope
            plot_data_pvalue = data_pvalue
        elif veg_type == 'Forest':
            mask = (veg_type_data == 1)
            plot_data_slope = np.where(mask, data_slope, np.nan)
            plot_data_pvalue = np.where(mask, data_pvalue, np.nan)
        elif veg_type == 'Shrub':
            mask = (veg_type_data == 2)
            plot_data_slope = np.where(mask, data_slope, np.nan)
            plot_data_pvalue = np.where(mask, data_pvalue, np.nan)
        elif veg_type == 'Savanna':
            mask = (veg_type_data == 3)
            plot_data_slope = np.where(mask, data_slope, np.nan)
            plot_data_pvalue = np.where(mask, data_pvalue, np.nan)
        elif veg_type == 'Grass':
            mask = (veg_type_data == 4)
            plot_data_slope = np.where(mask, data_slope, np.nan)
            plot_data_pvalue = np.where(mask, data_pvalue, np.nan)
        elif veg_type == 'Wet':
            mask = (veg_type_data == 5)
            plot_data_slope = np.where(mask, data_slope, np.nan)
            plot_data_pvalue = np.where(mask, data_pvalue, np.nan)

        ########### 子图1：空间分布
        ### 创建地图
        m = Basemap(ax=ax1, projection='cyl', resolution='l',
                    llcrnrlon=lon_min, llcrnrlat=lat_min,
                    urcrnrlon=lon_max, urcrnrlat=lat_max)

        # 生成网格坐标
        lons = np.linspace(lon_min, lon_max, cols)
        lats = np.linspace(lat_max, lat_min, rows)
        lons, lats = np.meshgrid(lons, lats)
        X, Y = m(lons, lats)

        ### 设置经纬度刻度
        m.drawparallels(np.arange(30, 90, 20), dashes=[0, 1])
        m.drawmeridians(np.arange(-180, 180, 30), dashes=[0, 1])

        # 设置刻度标签
        xticks = []
        xlabels = []
        for lon in range(-180, 181, 30):
            if lon < 0:
                xlabels.append(f'{abs(lon)}°W')
            elif lon > 0:
                xlabels.append(f'{lon}°E')
            else:
                xlabels.append('0°')
            xticks.append(lon)
        yticks = np.arange(30, 90, 20)

        ax1.set_xticks(xticks)
        ax1.set_yticks(yticks)
        ax1.set_xticklabels(xlabels, rotation=0)
        ax1.set_yticklabels([f'{y}°N' for y in yticks])

        ax1.xaxis.set_ticks_position('top')
        ax1.yaxis.set_ticks_position('left')
        ax1.tick_params(axis='both', which='major',
                        length = 2, pad=3,
                        top=True, bottom=False, left=True, right=False)

        ### 绘制边界
        terrence = r'D:\CAU\phenology_swc_vpd\Global_test\制图\border\NH_Terrence'
        m.readshapefile(terrence, 'NH Terrence', drawbounds=True, linewidth=0.2, color='black', zorder=3)

        ### 颜色映射
        color_list = ['#762a83', '#9970ab', '#c2a5cf', '#e7d4e8',
                      '#d9f0d3', '#a6dba0', '#5aae61', '#1b7837']
        cmap = mpl.colors.ListedColormap(color_list)
        bins = np.linspace(colorbarmin, colorbarmax, 9)
        norm = mpl.colors.BoundaryNorm(bins, cmap.N)

        # 绘制数据
        plot = m.pcolormesh(X, Y, plot_data_slope, cmap=cmap, norm=norm, zorder=1)
        plots.append(plot)  # 保存plot对象

        # 添加显著性标记 #
        significant_mask = (plot_data_pvalue < 0.05) & np.isfinite(plot_data_pvalue) & np.isfinite(plot_data_slope)

        if np.any(significant_mask):
            sig_rows, sig_cols = np.where(significant_mask)
            sig_lons = lons[sig_rows, sig_cols]
            sig_lats = lats[sig_rows, sig_cols]

            # 将地理坐标转为投影坐标
            sig_x, sig_y = m(sig_lons, sig_lats)

            ax1.scatter(sig_x, sig_y, marker='.', color='black', s=0.5,
                       linewidth=0.1, zorder=2)


        ########### 子图2：逐纬度变化趋势
        plot_data_slope_lat = np.nanmean(plot_data_slope, axis=1)

        # 使用实际纬度值作为y轴
        lat_centers = np.linspace(lat_max, lat_min, rows)

        ax2.plot(plot_data_slope_lat, lat_centers, color='red', linewidth=1, alpha=0.8)
        ax2.axvline(x=0, color='gray', linestyle='--', linewidth=1)

        if veg_type == 'Wet':
            ax2.set_xlabel('SM-VPD correlation trend(per year)')

        ax2.set_xlim(-0.02, 0.02)
        ax2.set_xticks(np.arange(-0.02, 0.0201, 0.02))
        ax2.set_xticklabels(['-0.02', '0', '0.02'])  # 手动设置标签


        ax2.set_ylim(30, int(lat_max)+1)
        ax2.set_yticks(np.arange(30, int(lat_max)+1, 10))

        ax2.tick_params(axis='both', which='major', length = 2, pad=3)

        ax2.set_aspect(0.0015)  # 数值越小，图形越宽；数值越大，图形越高

        ########### 子图3：频率分布图
        ###### 生成频率柱状图
        plot_data_slope_vaild = plot_data_slope[np.isfinite(plot_data_slope)]

        hist, bin_edges = np.histogram(plot_data_slope_vaild, bins=bins,density=False)  # density=True 自动计算密度, bins=bins,
        print('Density:', hist)
        print('bin_edges:', bin_edges)

        fraction = (hist/len(plot_data_slope_vaild))*100
        print(f'fraction:{fraction}')

        bar_width = 1  # 设置柱子的宽度

        x_positions = np.arange(len(hist))

        ax3.bar(x_positions, fraction, width=bar_width, color=color_list)

        # 设置x轴刻度
        ax3.set_xticks(x_positions)  # 设置x轴刻度
        ax3.tick_params(axis='x', which='both',
                        length = 2, pad=3,
                        top=False, bottom=False, labelbottom=False)  # 刻度线长度设为0

        ax3.set_ylim(0, 45)
        ax3.set_yticks(np.arange(0, 45.01, 15))

        ax3.tick_params(axis='y')

        # 设置y轴
        ax3.set_ylabel('Fraction(%)')  # 显示密度而非频率

        # 隐藏边框，保留左边框
        for spine in ax3.spines.values():
            spine.set_visible(False)
        ax3.spines['left'].set_visible(True)

        # 占比标注（所有+显著）
        # 计算正负值和零值的统计
        total_count = len(plot_data_slope_vaild)
        count_positive = np.count_nonzero(plot_data_slope_vaild > 0)
        count_negative = np.count_nonzero(plot_data_slope_vaild < 0)
        count_zero = np.count_nonzero(plot_data_slope_vaild == 0)
        print(veg_type)
        percent_positive = (count_positive / total_count) * 100
        percent_negative = (count_negative / total_count) * 100
        percent_zero = (count_zero / total_count) * 100

        # 计算显著性比例
        significant_positive = np.count_nonzero(
            (plot_data_slope > 0) & (plot_data_pvalue < 0.05) & np.isfinite(plot_data_slope) & np.isfinite(
                plot_data_pvalue))
        significant_negative = np.count_nonzero(
            (plot_data_slope < 0) & (plot_data_pvalue < 0.05) & np.isfinite(plot_data_slope) & np.isfinite(
                plot_data_pvalue))

        percent_sig_positive = (significant_positive / total_count) * 100
        percent_sig_negative = (significant_negative / total_count) * 100

        # 计算mean
        mean_val = np.mean(plot_data_slope_vaild)

        stats_text  = (f'P: {percent_positive:.1f}%({percent_sig_positive:.1f}%)\n'
                      f'N: {percent_negative:.1f}%({percent_sig_negative:.1f}%)\n'
                      f'No trend: {percent_zero:.1f}%')

        ax3.text(0.05, 1.25, stats_text,
                 transform=ax3.transAxes, fontsize=8,
                 verticalalignment='top')

        ax3.set_aspect(0.2)  # 数值越小，图形越宽；数值越大，图形越高


        # 设置子图标题
        ax1.set_title(f'{title}(Mean = {mean_val:.4f})')

        ### 导出分植被类型的tif结果
        tif_output_path1 = os.path.join(output_cor_mean_slope_tif_path, rf'slope\SM_VPD_Cor{test_number}\Cor_slope_{scale}km_{veg_type}.tif')
        tif_output_path2 = os.path.join(output_cor_mean_slope_tif_path, rf'pvalue\SM_VPD_Cor{test_number}\Cor_pvalue_{scale}km_{veg_type}.tif')

        save_tif_gdal(
            tif_output_path1,
            plot_data_slope,
            crs,
            gt  # 使用新的地理变换参数
        )

        save_tif_gdal(
            tif_output_path2,
            plot_data_pvalue,
            crs,
            gt  # 使用新的地理变换参数
        )


    ### 生成Colorbar（使用最后一个位置）
    cbar_ax = plt.subplot(gs[-1, :])
    cbar = fig.colorbar(plots[0], cax=cbar_ax, orientation='horizontal')

    cbar.set_label('SM-VPD correlation trend(per year)')


    # cbar.set_ticks(np.arange(colorbarmin, colorbarmax+0.01, 0.2))
    cbar.set_ticklabels([f'{x:.3f}' for x in bins])

    # 手动调整colorbar位置使其居中
    cbar.ax.set_position([0.18, 0.13, 0.4, 0.02])  # [left, bottom, width, height]

    plt.tight_layout()

    # plt.show()

    # 保存图片
    fig_path = os.path.join(output_cor_mean_slope_png_path, rf'slope\Cor17(Filter_precipitation)\Cor{test_number}_Slope_pvalue_{scale}km_vegType.png')


    plt.savefig(fig_path, dpi=600, bbox_inches='tight')
    print('Plot done!')


def plot_cor_slope_and_pvalue_combine_forKB(data_slope, data_pvalue, colorbarmin, colorbarmax):
    # 创建5个子图：4个地图 + 1个colorbar
    fig = plt.figure(figsize=(12, 8.2))
    gs = gridspec.GridSpec(5, 3,
                           width_ratios=[7, 0.5, 1],  #三列的宽度比
                           height_ratios=[1, 1, 1, 1, 0.1], # 最后一个给colorbar
                           hspace=0.5, wspace=0.18)

    # 统一设置所有字体大小
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 12,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 12,
    })

    # 定义四种植被类型
    # KB_types = ['Arid', 'Temperate', 'Continental', 'Polar']
    # titles = ['Arid region', 'Temperate region', 'Continental region', 'Polar region']
    KB_types = ['All', 'Arid', 'Temperate', 'Continental']
    titles = ['All climate zones', 'Arid zone', 'Temperate zone', 'Continental zone']

    plots = []  # 存储每个子图的plot对象

    for i, (KB_type, title) in enumerate(zip(KB_types, titles)):
        ax1 = plt.subplot(gs[i, 0])
        ax2 = plt.subplot(gs[i, 1])
        ax3 = plt.subplot(gs[i, 2])

        # 创建植被类型掩码
        if KB_type == 'All':
            plot_data_slope = data_slope
            plot_data_pvalue = data_pvalue
        elif KB_type == 'Arid':
            mask = (climate_type == 2)
            plot_data_slope = np.where(mask, data_slope, np.nan)
            plot_data_pvalue = np.where(mask, data_pvalue, np.nan)
        elif KB_type == 'Temperate':
            mask = (climate_type == 3)
            plot_data_slope = np.where(mask, data_slope, np.nan)
            plot_data_pvalue = np.where(mask, data_pvalue, np.nan)
        elif KB_type == 'Continental':
            mask = (climate_type == 4)
            plot_data_slope = np.where(mask, data_slope, np.nan)
            plot_data_pvalue = np.where(mask, data_pvalue, np.nan)
        # elif KB_type == 'Polar':
        #     mask = (climate_type == 5)
        #     plot_data_slope = np.where(mask, data_slope, np.nan)
        #     plot_data_pvalue = np.where(mask, data_pvalue, np.nan)

        ########### 子图1：空间分布
        ### 创建地图
        m = Basemap(ax=ax1, projection='cyl', resolution='l',
                    llcrnrlon=lon_min, llcrnrlat=lat_min,
                    urcrnrlon=lon_max, urcrnrlat=lat_max)

        # 生成网格坐标
        lons = np.linspace(lon_min, lon_max, cols)
        lats = np.linspace(lat_max, lat_min, rows)
        lons, lats = np.meshgrid(lons, lats)
        X, Y = m(lons, lats)

        ### 设置经纬度刻度
        m.drawparallels(np.arange(30, 90, 20), dashes=[0, 1])
        m.drawmeridians(np.arange(-180, 180, 30), dashes=[0, 1])

        # 设置刻度标签
        xticks = []
        xlabels = []
        for lon in range(-180, 181, 30):
            if lon < 0:
                xlabels.append(f'{abs(lon)}°W')
            elif lon > 0:
                xlabels.append(f'{lon}°E')
            else:
                xlabels.append('0°')
            xticks.append(lon)
        yticks = np.arange(30, 90, 20)

        ax1.set_xticks(xticks)
        ax1.set_yticks(yticks)
        ax1.set_xticklabels(xlabels, rotation=0)
        ax1.set_yticklabels([f'{y}°N' for y in yticks])

        ax1.xaxis.set_ticks_position('top')
        ax1.yaxis.set_ticks_position('left')
        ax1.tick_params(axis='both', which='major',
                        length = 2, pad=3,
                        top=True, bottom=False, left=True, right=False)

        ### 绘制边界
        terrence = r'D:\CAU\phenology_swc_vpd\Global_test\制图\border\NH_Terrence'
        m.readshapefile(terrence, 'NH Terrence', drawbounds=True, linewidth=0.2, color='black', zorder=3)

        ### 颜色映射
        color_list = ['#762a83', '#9970ab', '#c2a5cf', '#e7d4e8',
                      '#d9f0d3', '#a6dba0', '#5aae61', '#1b7837']
        cmap = mpl.colors.ListedColormap(color_list)
        bins = np.linspace(colorbarmin, colorbarmax, 9)
        norm = mpl.colors.BoundaryNorm(bins, cmap.N)

        # 绘制数据
        plot = m.pcolormesh(X, Y, plot_data_slope, cmap=cmap, norm=norm, zorder=1)
        plots.append(plot)  # 保存plot对象

        # 添加显著性标记 #
        significant_mask = (plot_data_pvalue < 0.05) & np.isfinite(plot_data_pvalue) & np.isfinite(plot_data_slope)

        if np.any(significant_mask):
            sig_rows, sig_cols = np.where(significant_mask)
            sig_lons = lons[sig_rows, sig_cols]
            sig_lats = lats[sig_rows, sig_cols]

            # 将地理坐标转为投影坐标
            sig_x, sig_y = m(sig_lons, sig_lats)

            ax1.scatter(sig_x, sig_y, marker='.', color='black', s=0.5,
                       linewidth=0.1, zorder=2)


        ########### 子图2：逐纬度变化趋势
        plot_data_slope_lat = np.nanmean(plot_data_slope, axis=1)

        # 使用实际纬度值作为y轴
        lat_centers = np.linspace(lat_max, lat_min, rows)

        ax2.plot(plot_data_slope_lat, lat_centers, color='red', linewidth=1, alpha=0.8)
        ax2.axvline(x=0, color='gray', linestyle='--', linewidth=1)

        if KB_type == 'Continental':
            ax2.set_xlabel('SM-VPD correlation trend(per year)')

        ax2.set_xlim(-0.02, 0.02)
        ax2.set_xticks(np.arange(-0.02, 0.021, 0.02))
        ax2.set_xticklabels(['-0.1', '0', '0.1'])  # 手动设置标签


        ax2.set_ylim(30, int(lat_max)+1)
        ax2.set_yticks(np.arange(30, int(lat_max)+1, 10))

        ax2.tick_params(axis='both', which='major', length = 2, pad=3)

        ax2.set_aspect(0.0015)  # 数值越小，图形越宽；数值越大，图形越高

        ########### 子图3：频率分布图
        ###### 生成频率柱状图
        plot_data_slope_vaild = plot_data_slope[np.isfinite(plot_data_slope)]

        hist, bin_edges = np.histogram(plot_data_slope_vaild, bins=bins,density=False)  # density=True 自动计算密度, bins=bins,
        print('Density:', hist)
        print('bin_edges:', bin_edges)

        fraction = (hist/len(plot_data_slope_vaild))*100
        print(f'fraction:{fraction}')

        bar_width = 1  # 设置柱子的宽度

        x_positions = np.arange(len(hist))

        ax3.bar(x_positions, fraction, width=bar_width, color=color_list)

        # 设置x轴刻度
        ax3.set_xticks(x_positions)  # 设置x轴刻度
        ax3.tick_params(axis='x', which='both',
                        length = 2, pad=3,
                        top=False, bottom=False, labelbottom=False)  # 刻度线长度设为0

        ax3.set_ylim(0, 45)
        ax3.set_yticks(np.arange(0, 45.01, 15))

        ax3.tick_params(axis='y')

        # 设置y轴
        ax3.set_ylabel('Fraction(%)')  # 显示密度而非频率

        # 隐藏边框，保留左边框
        for spine in ax3.spines.values():
            spine.set_visible(False)
        ax3.spines['left'].set_visible(True)

        # 占比标注（所有+显著）
        # 计算正负值和零值的统计
        total_count = len(plot_data_slope_vaild)
        count_positive = np.count_nonzero(plot_data_slope_vaild > 0)
        count_negative = np.count_nonzero(plot_data_slope_vaild < 0)
        count_zero = np.count_nonzero(plot_data_slope_vaild == 0)
        print('KB_type',KB_type)
        percent_positive = (count_positive / total_count) * 100
        percent_negative = (count_negative / total_count) * 100
        percent_zero = (count_zero / total_count) * 100

        # 计算显著性比例
        significant_positive = np.count_nonzero(
            (plot_data_slope > 0) & (plot_data_pvalue < 0.05) & np.isfinite(plot_data_slope) & np.isfinite(
                plot_data_pvalue))
        significant_negative = np.count_nonzero(
            (plot_data_slope < 0) & (plot_data_pvalue < 0.05) & np.isfinite(plot_data_slope) & np.isfinite(
                plot_data_pvalue))

        percent_sig_positive = (significant_positive / total_count) * 100
        percent_sig_negative = (significant_negative / total_count) * 100

        # 计算mean
        mean_val = np.mean(plot_data_slope_vaild)

        stats_text  = (f'P: {percent_positive:.1f}%({percent_sig_positive:.1f}%)\n'
                      f'N: {percent_negative:.1f}%({percent_sig_negative:.1f}%)\n'
                      f'No trend: {percent_zero:.1f}%')

        ax3.text(0.05, 1.25, stats_text,
                 transform=ax3.transAxes, fontsize=8,
                 verticalalignment='top')

        ax3.set_aspect(0.2)  # 数值越小，图形越宽；数值越大，图形越高


        # 设置子图标题
        ax1.set_title(f'{title}(Mean = {mean_val:.4f})')

        ### 导出分植被类型的tif结果
        tif_output_path1 = os.path.join(output_cor_mean_slope_tif_path, rf'slope\SM_VPD_Cor{test_number}\Cor_slope_{scale}km_KB({KB_type}).tif')

        tif_output_path2 = os.path.join(output_cor_mean_slope_tif_path, rf'pvalue\SM_VPD_Cor{test_number}\Cor_pvalue_{scale}km_KB({KB_type}).tif')

        save_tif_gdal(
            tif_output_path1,
            plot_data_slope,
            crs,
            gt  # 使用新的地理变换参数
        )

        save_tif_gdal(
            tif_output_path2,
            plot_data_pvalue,
            crs,
            gt  # 使用新的地理变换参数
        )


    ### 生成Colorbar（使用最后一个位置）
    cbar_ax = plt.subplot(gs[-1, :])
    cbar = fig.colorbar(plots[0], cax=cbar_ax, orientation='horizontal')


    cbar.set_label('Peak of Growing Season trend (per year)')

    # cbar.set_ticks(np.arange(colorbarmin, colorbarmax+0.01, 0.2))
    cbar.set_ticklabels([f'{x:.3f}' for x in bins])

    # 手动调整colorbar位置使其居中
    cbar.ax.set_position([0.18, 0.13, 0.4, 0.018])  # [left, bottom, width, height]

    plt.tight_layout()

    # plt.show()

    # 保存图片
    fig_path = os.path.join(output_cor_mean_slope_png_path, rf'slope\Cor17(Filter_precipitation)\Cor{test_number}_Slope_pvalue_{scale}km_KB.png')

    plt.savefig(fig_path, dpi=600, bbox_inches='tight')
    print('Plot done!')



if calculate_cor == 1:

    cor_mean = np.full((rows, cols), np.nan)
    cor_slope = np.full((rows, cols), np.nan)
    cor_slope_pvalue = np.full((rows, cols), np.nan)


    #### 筛选标准为：有效年份>=12
    cor_valid_mask = np.count_nonzero(np.isfinite(cor_matrix), axis=0) >= (years_length / 2)

    cor_matrix_valid = np.where(np.isfinite(cor_matrix) & cor_valid_mask, cor_matrix, np.nan)
    p_matrix_valid = np.where(np.isfinite(p_matrix) & cor_valid_mask, p_matrix, np.nan)

    ###仅导出有效像元
    for year in years:
        print(f"正在导出年份：{year}")
        k = year - 2001  # 年份索引

        output_path1 = os.path.join(output_cor_tif_path, f"Cor_pearson_{year}.tif")
        save_tif_gdal(
            output_path1,
            cor_matrix_valid[k, :, :],
            crs, gt
        )

        output_path1_1 = os.path.join(output_cor_tif_path, rf"Pvalue\Cor_pearson_pvalue_{year}.tif")
        save_tif_gdal(
            output_path1_1,
            p_matrix_valid[k, :, :],
            crs, gt
        )
        print(f"{year}年耦合效应、SM、VPD结果已保存")
