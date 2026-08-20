
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


def clip_by_latitude(gt, rows, cols, lat_min, lat_max):
    """
    根据纬度范围裁剪数据
    返回: (row_start, row_end, new_gt)
    """
    # 计算每行的中心纬度
    row_centers = np.arange(rows) * gt[5] + gt[3] + gt[5] / 2

    # 找到纬度在30-90度范围内的行
    valid_rows = (row_centers >= lat_min) & (row_centers <= lat_max)

    if not np.any(valid_rows):
        raise ValueError(f"在纬度范围 {lat_min}-{lat_max} 内没有找到有效数据")

    # 找到第一个和最后一个有效行
    valid_row_indices = np.where(valid_rows)[0]
    row_start = valid_row_indices[0]
    row_end = valid_row_indices[-1] + 1  # 切片是左闭右开

    # 计算新的左上角坐标
    new_top_left_x = gt[0] + row_start * gt[2]  # 通常gt[2]=0
    new_top_left_y = gt[3] + row_start * gt[5]  # gt[5]是像素高度

    # 创建新的地理变换参数
    new_gt = (
        new_top_left_x,
        gt[1],  # 像素宽度不变
        gt[2],  # 行旋转不变
        new_top_left_y,
        gt[4],  # 列旋转不变
        gt[5]  # 像素高度不变
    )

    # print(f"纬度范围: {lat_min}-{lat_max}°N")
    # print(f"对应的行范围: {row_start} - {row_end - 1}")
    # print(f"原始行数: {rows}, 裁剪后行数: {row_end - row_start}")
    # print(f"原始左上角纬度: {gt[3]:.2f}°N, 新左上角纬度: {new_top_left_y:.2f}°N")

    return row_start, row_end, new_gt


### 读取SM和VPD波段
def get_band(tif_file, stack):
    tif = gdal.Open(tif_file)
    data = tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
    stack.append(data)


def get_band_clip(tif_file, stack, row_start, row_end):
    tif = gdal.Open(tif_file)
    climate_data = tif.GetRasterBand(1).ReadAsArray().astype(np.float32)

    climate_data_cropped = climate_data[row_start:row_end, :]
    stack.append(climate_data_cropped)

    tif = None  # 及时释放资源



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

    if (len(data_clean) >= (years_length/2)) and (len(data_clean) <= years_length):   ### >10 / >17/ >19

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



### 提取季前生长季内的数据
# def extract_time_window(year, pos, dates):
def extract_time_window(year, sos, pos, dates):
    # """根据像元的sos和pos，提取年份year对应的时间窗口索引"""
    # 计算该像元在年份year的生长季起止日期

    # print('pos：', pos, flush=True)

    ### Preseason考虑SOS-POS
    start_date = pd.to_datetime(f"{year}-01-01") + pd.Timedelta(days=int(pos-90))
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


def compute_pearson_for_pixel(sm_series, vpd_series, pre_origin_series):
    """对一个像元的时间序列计算 Pearson r 和 p"""

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


def process_pixel_sm_vpd_coupling(i, j, year, pos, sos, year_dates, sm_data, vpd_data, pre_origin_data):
# def process_pixel1(i, j, year, pos, dates, sm_data, vpd_data):
    """处理单个像元的函数（供并行调用）"""
    sos_pixel = sos[i, j]
    pos_pixel = pos[i, j]
    # print(f'sos_pixel:{sos_pixel}, pos_pixel:{pos_pixel}')

    if not pd.isna(pos_pixel) and not pd.isna(sos_pixel) and not pd.isna(sm_data[0, i, j]):

        valid_indices = extract_time_window(year, sos_pixel, pos_pixel, year_dates)
        sm_series = sm_data[valid_indices, i, j].flatten()
        vpd_series = vpd_data[valid_indices, i, j].flatten()

        pre_origin_series = pre_origin_data[valid_indices, i, j].flatten()

        # 计算Pearson相关系数
        if lag_day == 0:
            r, p = compute_pearson_for_pixel(sm_series, vpd_series, pre_origin_series)
        elif lag_day != 0:
            r, p = compute_pearson_for_pixel(sm_series[:-lag_day], vpd_series[lag_day:], pre_origin_series[:-lag_day])


        # print('Cor:', r)
        # print('SM:', mean1)
        # print('VPD:', mean2)

        return (i, j, r, p)

    else:
        return (i, j, np.nan, np.nan)




def process_pixel2(i, j, year, pos, sos, dates, ta_data, pre_data, srad_data):
# def process_pixel2(i, j, year, pos, dates, ta_data, pre_data, srad_data):
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




##################################### 6 Plot #########################################
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
import matplotlib.gridspec as gridspec

scale = 55

same_input_path = rf'D:\CAU\phenology_swc_vpd\paper\Data_and_code_online\FigShare_data\{scale}km'

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

############# 4.2 植被类型 ################
veg_type_data = gdal.Open(veg_type_tif)
veg_type_data = veg_type_data.GetRasterBand(1).ReadAsArray().astype(np.float32)

# ############# 4.4 AI ################
ai_type_data = gdal.Open(ai_tif)
ai_type_data = ai_type_data.GetRasterBand(1).ReadAsArray().astype(np.float32)


sample_tif = gdal.Open(cor_mean_path)

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




## a left
def plot_cor_mean_or_slope_and_pvalue_forAllvegType(plot_data, plot_data_pvalue,  colorbarmin, colorbarmax, data_type, name, ax):
    # # 创建5个子图：4个地图 + 1个colorbar
    # fig = plt.figure(figsize=(6, 4))
    # gs = gridspec.GridSpec(1, 3,
    #                        width_ratios=[4, 1, 1],  #三列的宽度比
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

    # 隐藏父级 ax，因为它只是个占位符
    ax.axis('off')

    plots = []  # 存储每个子图的plot对象

    # ax1 = plt.subplot(gs[0, 0])
    # ax2 = plt.subplot(gs[0, 1])
    # ax3 = plt.subplot(gs[0, 2])
    # ax3 = plt.subplot(gs[0, 2])
    # 创建内部真正的三个子轴
    ax1 = fig.add_subplot(gs_inner[0, 0])  # 地图
    ax2 = fig.add_subplot(gs_inner[0, 1])  # 纬度曲线
    ax3 = fig.add_subplot(gs_inner[1, :])  # Colorbar横跨两列

    # word = 'a'

    ########### 子图1：空间分布 #################
    ax1.set_box_aspect(1)  #强制地图轴的形状为正方，使其直径撑满格子高度
    ax1.axis('off')
    ### 创建地图
    m = Basemap(ax=ax1,
                projection='npstere',   # 北极投影
                boundinglat=30,         # 最低显示纬度（你现在是30N）
                lon_0=0,                # 中心经度（可以改） 180:太平洋居中；90：亚洲居中
                resolution='l')

    # 生成网格坐标
    lons = np.linspace(-180, 180, cols, endpoint=False)
    lats = np.linspace(lat_max , lat_min , rows)
    lons, lats = np.meshgrid(lons, lats)

    # 设置经纬度刻度
    m.drawparallels(np.arange(30, 90, 30),
                    labels=[0, 0, 0, 0],
                    linewidth=0.5,
                    # fontsize=8,
                    color="black")

    m.drawmeridians(np.arange(0, 360, 60),
                    latmax=90,  #使经度线在北极交汇
                    labels=[0, 0, 0, 0],  #labels=[left, right, top, bottom] 控制经度显示与否
                    linewidth=0.5,
                    # fontsize=8,
                    color="gray")

    # # 填充大陆
    # m.fillcontinents(color='white', lake_color='white', zorder=1)

    # map_boundary = m.drawmapboundary(linewidth=0)  # 不显示边界线


    ### 绘制数据
    # 颜色映射
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
                        zorder=1)  # 避免极区撕裂

    plots.append(plot)  # 保存plot对象

    if data_type == 'cor slope':
        # 添加显著性标记
        significant_mask = (plot_data_pvalue < 0.05) & np.isfinite(plot_data_pvalue) & np.isfinite(plot_data)

        if np.any(significant_mask):
            sig_rows, sig_cols = np.where(significant_mask)
            sig_lons = lons[sig_rows, sig_cols]
            sig_lats = lats[sig_rows, sig_cols]

            # 将地理坐标转为投影坐标
            sig_x, sig_y = m(sig_lons, sig_lats)

            ax1.scatter(sig_x, sig_y, marker='.', color='black', s=0.5,
                       linewidth=0.1, zorder=2)
    ax1.set_frame_on(False)

    ### 绘制边界
    terrence = r'D:\CAU\Border\NH_terrence\NH_Terrence_30_90'
    # m.readshapefile(terrence, 'NH_Terrence', drawbounds=True, linewidth=0.2, color='black', zorder=3)
    m.readshapefile(terrence, 'NH_Terrence', drawbounds=False)
    for shape in m.NH_Terrence:
        # 将列表转为 numpy 数组方便计算
        points = np.array(shape)
        x, y = points[:, 0], points[:, 1]

        # 核心逻辑：计算相邻点之间的投影距离
        # 如果相邻两个点在投影平面上的距离突然变得非常大，说明这是一条“跨圆心”的回环线
        dist = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)

        # 设定一个阈值（投影坐标通常很大，比如 100000 级）
        # 只要相邻点距离超过地图直径的 1/10，就判定为异常跳变
        threshold = (ax1.get_xlim()[1] - ax1.get_xlim()[0]) * 0.1

        # 找到跳变点的索引
        break_indices = np.where(dist > threshold)[0]

        if len(break_indices) == 0:
            # 没有跳变，直接画整条线
            ax1.plot(x, y, color='black', linewidth=0.3, zorder=3)
        else:
            # 有跳变，将线段切断，分段画出
            # 这样既能去掉横跨圆心的直线，又能保留正常的边界
            start_idx = 0
            for break_idx in break_indices:
                ax1.plot(x[start_idx:break_idx + 1], y[start_idx:break_idx + 1],
                         color='black', linewidth=0.3, zorder=3)
                start_idx = break_idx + 1
            # 画最后一段
            ax1.plot(x[start_idx:], y[start_idx:], color='black', linewidth=0.3, zorder=3)

    # x_pole, y_pole = m(0, 90)
    # ax1.scatter(x_pole, y_pole, color='black', s=5, zorder=3)

    ### 最外边界的裁剪
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
        edgecolor='black',  # 颜色
        linewidth=0.8,
        clip_on = False,
        zorder=4  # 放最上层
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


    ## 统计 ##
    data_gte0 = plot_data[(plot_data >= 0) & np.isfinite(plot_data)]
    data_lt0 = plot_data[(plot_data < 0) & np.isfinite(plot_data)]
    sum_count = np.sum(np.isfinite(plot_data))

    data_gte0_count = np.sum(np.isfinite(data_gte0))
    data_lt0_count = np.sum(np.isfinite(data_lt0))

    data_gte0_ratio = data_gte0_count/sum_count * 100
    data_lt0_ratio = data_lt0_count/sum_count * 100


    if data_type == 'cor mean':
        h = 0.25
        v = 0.82
        ax1.text(h, v,
                 f'Mean = {np.nanmean(plot_data):.2f}\n'
                 f'Pos = {np.nanmean(data_gte0):.2f} ({data_gte0_ratio:.1f}%)\n'
                 f'Neg = {np.nanmean(data_lt0):.2f} ({data_lt0_ratio:.1f}%)',
                 transform=ax1.transAxes,  # 使用相对坐标，方便定位
                 multialignment='center',  # 垂直居中
                 fontsize = 6)
    elif data_type == 'cor slope':
        h = 0.21
        v = 0.82
        ax1.text(h, v,
                 f'Mean = {np.nanmean(plot_data):.3f}\n'
                 f'Pos trend = {np.nanmean(data_lt0):.3f} ({data_lt0_ratio:.1f}%)\n'
                 f'Neg trend = {np.nanmean(data_gte0):.3f} ({data_gte0_ratio:.1f}%)',
                 transform=ax1.transAxes,  # 使用相对坐标，方便定位
                 multialignment='center',  # 垂直居中
                 fontsize = 6)

    ########### 子图2：逐纬度变化趋势

    # 使用实际纬度值作为y轴
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
        ax2.set_xticklabels(['-1', '0', '1'])  # 手动设置标签

        tick_size = plt.rcParams['xtick.labelsize']
        ax2.text(
            0.98,  # x = 刻度位置（数据坐标）
            -0.02,  # y = 稍微往下（轴坐标）
            r'$×10^{-2}$',  # 你想要的内容
            transform=ax2.transAxes,
            ha='left',  # 向右展开（避免压缩）
            va='top',
            fontsize=8,
            clip_on=False
        )
    elif data_type == 'cor mean' :
        ax2.set_xlim(-0.5, 0)
        ax2.set_xticks(np.arange(-0.3, 0.01, 0.3))
        ax2.set_xticklabels(['-0.3','0'])  # 手动设置标签

    ax2.set_ylim(30, 90)
    ticks = np.arange(30, 91, 10)
    ax2.set_yticks(ticks)
    ax2.set_yticklabels(f'{x}°' for x in ticks)


    ax2.tick_params(axis='both', which='major', length = 2, pad=3)

    ########### 子图3：Colorbar
    ### 生成Colorbar（使用最后一个位置）
    cbar = fig.colorbar(plots[0], cax=ax3, orientation='horizontal')

    cbar.set_ticks(bins)

    if data_type == 'cor slope':
        cbar.set_label('SM-VPD coupling trend (per year)', labelpad = 13)
        cbar.set_ticklabels([ '0' if x == 0 else
            f'{int(x*100)}' if x*100 == int(x*100) else
            f'{x*100:.1f}'
            for x in bins])
        tick_size = plt.rcParams['xtick.labelsize']
        ax3.text(
            0.9,  # x = 刻度位置（数据坐标）
            -2.05,  # y = 稍微往下（轴坐标）
            r'$×10^{-2}$',  # 你想要的内容
            transform=ax3.transAxes,
            ha='left',  # 向右展开（避免压缩）
            va='top',
            fontsize=8,
            clip_on=False
        )
    if data_type == 'cor mean':
        cbar.set_label('SM-VPD coupling', labelpad = 13)
        cbar.set_ticklabels(['0' if x == 0 else
            f'{x:.1f}' for x in bins])

    ax3.tick_params(axis='both', length=2, pad=3)

    plt.tight_layout()

    # 当前 ax1 左下角
    pos1 = ax1.get_position()

    pos2 = ax2.get_position()

    pos3 = ax3.get_position()

    if name == 'All':
        # 重新设置 ax1
        ax1.set_position([
            pos1.x0 - 0.04,  # 左边不变
            pos2.y0,  # 和 ax2 对齐底部
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
                pos1.x0 - xpos,  # 左边不变
                pos2.y0,  # 和 ax2 对齐底部
                pos2.height,
                pos2.height
            ])  # [left, bottom, width, height]
        if data_type == 'cor slope':
            ax1.set_position([
                pos1.x0 - 0.145,  # 左边不变
                pos2.y0,  # 和 ax2 对齐底部
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

## a right
def plot_cor_mean_or_slope_lt0_sig_and_lt0_nosig_and_gt0_forAllvegType(plot_data, data_pvalue, data_type, colorbarmin, colorbarmax, name, ax):
    if data_type == 'cor slope':
        ### 数据划分
        #三个分类的划分
        plot_data_lt0_mask = plot_data<0
        plot_data_gte0_mask = plot_data>=0

        pvalue_sig_mask = data_pvalue<=0.05
        #
        # plot_data_lt0_sig = np.where(plot_data_lt0_mask & pvalue_sig_mask, plot_data, np.nan)
        # plot_data_lt0_all = np.where(plot_data_lt0_mask , plot_data, np.nan)
        # plot_data_gte0_sig = np.where(plot_data_gte0_mask & pvalue_sig_mask, plot_data, np.nan)
        # plot_data_gte0_all = np.where(plot_data_gte0_mask , plot_data, np.nan)

        plot_data_sig = np.where(pvalue_sig_mask, plot_data, np.nan)

        bins = np.arange(colorbarmin, colorbarmax+0.006, 0.006)
        # count_lt0_sig, _ = np.histogram(plot_data_lt0_sig, bins=bins)
        # count_lt0_all, _ = np.histogram(plot_data_lt0_all, bins=bins)
        # count_gte0_sig, _ = np.histogram(plot_data_gte0_sig, bins=bins)
        # count_gte0_all, _ = np.histogram(plot_data_gte0_all, bins=bins)

        count_sig, _ = np.histogram(plot_data_sig, bins=bins)
        count_all, _ = np.histogram(plot_data, bins=bins)



    elif data_type == 'cor mean':
        bins = np.arange(colorbarmin, colorbarmax+0.1, 0.1)
        count_mean, _ = np.histogram(plot_data, bins=bins)

    ### plot
    fig = ax.get_figure()
    gs_inner = ax.get_subplotspec().subgridspec(2, 1,
                                                height_ratios=[5, 0.3],
                                                hspace=0.15)

    # 隐藏父级 ax，因为它只是个占位符
    ax.axis('off')

    #设置截断y轴
    # if data_type == 'cor slope':
    #     ax.axis('off')
    #     bax = brokenaxes(
    #         ylims=((0, 5000), (10000, 11000)),
    #         hspace=0.1,
    #         height_ratios=[1, 5],  # 上图占1份，下图占2份。数值越大，对应的部分占地越广
    #         subplot_spec=sub_gs
    #     )
    #
    #     # 设置bar的位置
    #     bin_centers = (bins[:-1] + bins[1:]) / 2
    #     print(f'bin_centers:{bin_centers}')
    #
    #     total_width = 0.007  # 一个刻度位内柱子的总占用宽度
    #     n = 2  # 类别数量
    #     width = total_width / n  # 单个柱子的宽度
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
    #     bax.set_xlabel('VPD-SM coupling trend (per decade)', labelpad=20) #控制标签与刻度距离
    #
    #     bax.set_ylabel('Frequency', labelpad=31)  # 控制标签与刻度距离

    #     # Colorbar
    #     bax.legend(
    #         loc='upper right',
    #         bbox_to_anchor=(1.2, 1),
    #         ncol=1,
    #         frameon=False,  # 控制 legend（图例）外框是否显示
    #         handlelength=1,
    #         handleheight=1
    #     )


    # 不设置截断y轴
    if data_type == 'cor slope':

        ax1 = fig.add_subplot(gs_inner[0])

        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        # 设置bar的位置
        bin_centers = (bins[:-1] + bins[1:]) / 2
        print(f'bin_centers:{bin_centers}')

        color_list = ['#67001f', '#b2182b', '#d6604d', '#f4a582', '#fddbc7',
                      '#d1e5f0', '#92c5de', '#4393c3', '#2166ac', '#053061']

        cmap = mpl.colors.ListedColormap(color_list)

        # bins = np.linspace(colorbarmin, colorbarmax, 11)

        norm = mpl.colors.BoundaryNorm(bins, cmap.N)

        bin_colors = cmap(norm(bin_centers))

        total_width = 0.006  # 一个刻度位内柱子的总占用宽度
        n = 2  # 类别数量
        width = total_width / n  # 单个柱子的宽度

        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

        # ax1.bar(bin_centers, count_lt0_sig, width=0.004, linewidth = 0.4, hatch='///', facecolor='none', edgecolor='#f46d43', label='Sig-strong', zorder= 2)
        # ax1.bar(bin_centers, count_lt0_all, width=0.004, linewidth = 0.5, color='#fee090', label='Nonsig-strong', zorder= 1)
        # ax1.bar(bin_centers, count_gte0_all, width=0.004, linewidth = 0.5, color='#e0f3f8', label='Nonsig-weaken', zorder= 1)
        # ax1.bar(bin_centers, count_gte0_sig, width=0.004, linewidth = 0.4, hatch='///', facecolor='none', edgecolor='#74add1', label='Sig-weaken', zorder= 2)

        # 逐个柱子绘制，以确保颜色严格对应
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

        ### x轴设定
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
            y_pos =-0.15
        else:
            x_pos = 0.78
            y_pos = -0.15
        ax1.text(
            x_pos,  # x = 刻度位置（数据坐标）
            y_pos,  # y = 稍微往下（轴坐标）
            r'$×10^{-2}$',  # 你想要的内容
            transform=ax1.transAxes,
            ha='left',  # 向右展开（避免压缩）
            va='top',
            rotation=0,
            fontsize=8,
            clip_on=False
        )

        # ax1.set_xticklabels(labels, rotation=45)  # 核心：将 0.007 物理位置显示为 0.07 文字

        if name == 'All':
            ax1.set_ylim(0, 5000)
            ticks = np.arange(0, 5000.1, 1000)

            ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(abs(x))}"))

            ### y轴设定
            ax1.set_yticklabels(f'{int(x * 0.001)}' for x in ticks)



        elif name in ['Forests', 'Shrublands', 'Savannas', 'Grasslands']:
            ax1.set_ylim(0, 1500)
            ticks = np.arange(0, 1500.1, 300)
            ax1.set_yticks(ticks)

            ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(abs(x))}"))

            ### y轴设定
            ax1.set_yticklabels('0' if x == 0 else
                                f'{x * 0.001:.1f}' for x in ticks)

        elif name in ['Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']:
            ax1.set_ylim(0, 2500)
            ticks = np.arange(0, 2500.1, 500)
            ax1.set_yticks(ticks)

            ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(abs(x))}"))

            ### y轴设定
            ax1.set_yticklabels('0' if x == 0 else
                f'{x * 0.001:.1f}' for x in ticks)

        ax1.set_ylabel('Frequency', labelpad=5)  # 控制标签与刻度距离

        ax1.text(
            -0.1,  # x = 刻度位置（数据坐标）
            1.12,  # y = 稍微往下（轴坐标）
            r'$×10^{3}$',  # 你想要的内容
            transform=ax1.transAxes,
            ha='left',  # 向右展开（避免压缩）
            va='top',
            fontsize=8,
            clip_on=False
        )

        if name == 'All':
            x_position = 0.52
        else:
            x_position = 0.4

        # Colorbar
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
            frameon=False,  # 控制 legend（图例）外框是否显示
            handlelength=1,
            handleheight=1,
            columnspacing=0.5
        )



    elif data_type == 'cor mean':
        if name == 'All':
            bax = brokenaxes(
                ylims=((0, 4500), (6000, 7000)),
                hspace=0.1,
                height_ratios=[1, 5],  # 上图占1份，下图占2份。数值越大，对应的部分占地越广
                subplot_spec=gs_inner[0],
                d=0.005
            )

            total_width = 0.2  # 一个刻度位内柱子的总占用宽度
            n = 2  # 类别数量
            width = total_width / n  # 单个柱子的宽度

            bin_centers = (bins[:-1] + bins[1:]) / 2


            color_list = ['#a50f15', '#de2d26', '#fb6a4a', '#fc9272',
                          '#fcbba1', '#fee5d9', '#9ecae1']

            # 逐个柱子绘制，以确保颜色严格对应
            for j in range(len(count_mean)):
                bax.bar(bin_centers[j], count_mean[j], width=0.08,
                        color=color_list[j], edgecolor='none')

            # bax.set_xlim(colorbarmin, colorbarmax)
            # bax.set_xticks(np.arange(colorbarmin, colorbarmax, 0.1))
            # bax.axs[1].set_xticklabels(bax.axs[1].get_xticklabels())

            bax.tick_params(axis='both', length=2, pad=3)

            bax.set_xlim(colorbarmin, colorbarmax)
            xticks = np.arange(colorbarmin, colorbarmax+0.1, 0.1)

            bax.axs[1].set_xticks(xticks)
            bax.axs[1].set_xticklabels(
                ['0' if np.isclose(x, 0) else f'{x:.1f}'
                    for x in xticks],
                rotation=45
            )


            # bax.set_xlabel('VPD-SM coupling', labelpad=20) #控制标签与刻度距离

            bax.axs[0].set_yticks([6000, 7000])
            bax.axs[1].set_yticks([0, 1000, 2000, 3000, 4000])

            for ax in bax.axs:
                # 获取当前y轴刻度值
                y_ticks = ax.get_yticks()
                # 根据y轴刻度值生成标签
                ax.set_yticklabels([
                    '0' if y == 0 else
                    f'{int(y * 0.001)}'
                    for y in y_ticks
                ])

            ## 控制y轴 10的n次方
            bax.axs[0].text(
                -0.1,  # x = 刻度位置（数据坐标）
                1.65,  # y = 稍微往下（轴坐标）
                r'$×10^{3}$',  # 你想要的内容
                transform=bax.axs[0].transAxes,
                ha='left',  # 向右展开（避免压缩）
                va='top',
                rotation=0,
                fontsize=8,
                clip_on=False
            )


            bax.set_ylabel('Frequency', labelpad=15)  # 控制标签与刻度距离




        else:
            ax1 = fig.add_subplot(gs_inner[0])

            ax1.spines['top'].set_visible(False)

            ax1.spines['right'].set_visible(False)

            total_width = 0.2  # 一个刻度位内柱子的总占用宽度
            n = 2  # 类别数量
            width = total_width / n  # 单个柱子的宽度

            bin_centers = (bins[:-1] + bins[1:]) / 2

            color_list = ['#a50f15', '#de2d26', '#fb6a4a', '#fc9272',
                          '#fcbba1', '#fee5d9', '#9ecae1']

            # 逐个柱子绘制，以确保颜色严格对应
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

            # ax1.set_xlabel('VPD-SM coupling', labelpad=3)  # 控制标签与刻度距离
            if name in ['Forests', 'Shrublands', 'Savannas', 'Grasslands']:
                ax1.set_ylim(0, 2500)
                ticks = np.arange(0, 2500.1, 500)
                ax1.set_yticks(ticks)
            elif name in ['Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']:
                ax1.set_ylim(0, 4000)
                ticks = np.arange(0, 4000.1, 1000)
                ax1.set_yticks(ticks)

            ### y轴设定
            ax1.set_yticklabels(f'{x * 0.001:.1f}' for x in ticks)
            tick_size = plt.rcParams['xtick.labelsize']
            if data_type == 'cor mean':
                if name != 'All':
                    ypos =1.1
            else:
                ypos = 1
            ax1.text(
                -0.1,  # x = 刻度位置（数据坐标）
                ypos,  # y = 稍微往下（轴坐标）
                r'$×10^{3}$',  # 你想要的内容
                transform=ax1.transAxes,
                ha='left',  # 向右展开（避免压缩）
                va='top',
                fontsize=8,
                clip_on=False
            )

            ax1.set_ylabel('Frequency', labelpad=3)  # 控制标签与刻度距离

            ax1.tick_params(axis='both', length=2, pad=3)


    # if name == 'All':
    #     if data_type == 'cor mean':
    #         bax.set_title(f'(b)', pad=10, fontweight='bold')
    #     elif data_type == 'cor slope':
    #         ax.set_title(f'(d)', pad=10, fontweight='bold')
    # plt.show()

## b
def plot_cor_mean_or_slope_forDiffvegType_and_AI(plot_data, veg_data, ai_data, data_type, ax):

    # 1. 数据准备
    veg_list = [plot_data[(veg_data == i) & np.isfinite(plot_data)] for i in [1, 2, 3, 4]]
    veg_labels = ['Forests', 'Shrublands', 'Savannas', 'Grasslands']

    ai_list = [
        plot_data[(ai_data == 2) & np.isfinite(plot_data)],  # Arid
        plot_data[((ai_data == 3) | (ai_data == 4)) & np.isfinite(plot_data)],  # Semi-Arid (合并 3 和 4)
        plot_data[(ai_data == 5) & np.isfinite(plot_data)],  # Dry sub-humid
        plot_data[(ai_data == 6) & np.isfinite(plot_data)]  # Humid
    ]
    ai_labels = ['Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']

    full_data = veg_list + ai_list
    x_positions = [1, 2, 3, 4, 5, 6, 7, 8]  # X轴物理位置

    # 修正：安全计算均值
    full_data_mean = [np.mean(d) if len(d) > 0 else 0 for d in full_data]

    ### plot
    # fig, ax = plt.subplots(figsize=(4, 4))  # 稍微调宽一点，文字不拥挤
    # plt.subplots_adjust(bottom=0.2)  # 底部留白给旋转后的标签

    fig = ax.get_figure()

    gs_inner = ax.get_subplotspec().subgridspec(
        2, 1,
        height_ratios=[5, 0.3],
        hspace=0.15
    )

    ax1 = fig.add_subplot(gs_inner[0])
    ax.axis('off')

    # 增加 showmedians=True
    vio = ax1.violinplot(full_data, positions=x_positions,
                        showmeans=True, showextrema=False)

    # 颜色设置
    # v_colors = ['#0ebeff', '#ae63e4', '#ffd200', '#ff3c41',  # 植被颜色
    #             '#0ebeff', '#ae63e4', '#ffd200', '#ff3c41']  # AI 颜色建议区分开
    # 获取 Paired 颜色映射
    paired_colors = plt.cm.Paired(np.linspace(0, 1, 12))  # 先取8个颜色
    indices = [1, 3, 7, 9]
    colors = [paired_colors[i] for i in indices]
    v_colors = list(colors) + list(colors)

    for i, pc in enumerate(vio['bodies']):
        pc.set_facecolor(v_colors[i])
        pc.set_edgecolor('none')
        # pc.set_linewidth(0.5)
        # pc.set_alpha(0.7)

    # 分别设置均值线和中位线颜色
    # 均值线 (Means) - 红色
    vio['cmeans'].set_edgecolor('red')
    vio['cmeans'].set_linestyle('-')
    vio['cmeans'].set_linewidth(1.5)


    # 坐标轴美化
    ax1.set_xticks(x_positions)
    # if data_type == 'cor mean':
    #     ax1.set_xticklabels([])
    # elif data_type == 'cor slope':
    ax1.set_xticklabels(veg_labels + ai_labels, rotation=90)

    # if data_type == 'cor slope':
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)

    ax1.axvline(4.5, color='black', linestyle='-', linewidth=1)  # 垂直线分界

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

### All,Fig4
def plot_fig3(data_mean, data_slope, data_pvalue):

    # 统一设置所有字体大小
    plt.rcParams.update({
        'font.family': 'Arial',

        'mathtext.fontset': 'custom',

        'mathtext.rm': 'Arial',  # 正常
        'mathtext.it': 'Arial:italic',  # 斜体
        'mathtext.bf': 'Arial:bold',  # 粗体

        # 可选（推荐加）
        'mathtext.default': 'regular',  # 避免自动变斜体

        'font.size': 10,
        'axes.titlesize': 10,
        'axes.labelsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        # 'text.usetex': False,  # 不使用外部LaTeX
    })

    # 创建5个子图：4个地图 + 1个colorbar
    fig = plt.figure(figsize=(8.2, 6.5))
    gs = gridspec.GridSpec(2, 3,
                           width_ratios=[6, 3.5, 3.5],  # 三列的宽度比
                           height_ratios=[1, 1],  # 最后一个给colorbar
                           hspace=0.6, wspace=0.33)

    ax1 = plt.subplot(gs[0, 0])  ## fig a left
    ax2 = plt.subplot(gs[0, 1])  ## fig a right
    ax3 = plt.subplot(gs[0, 2])  ## fig b

    ax4 = plt.subplot(gs[1, 0])  ## fig a left
    ax5 = plt.subplot(gs[1, 1])  ## fig a right
    ax6 = plt.subplot(gs[1, 2])  ## fig b

    ### fig1 a left
    plot_cor_mean_or_slope_and_pvalue_forAllvegType(data_mean, data_pvalue,  -0.6, 0.1,  'cor mean', 'All',ax = ax1)

    ### fig1 a right
    plot_cor_mean_or_slope_lt0_sig_and_lt0_nosig_and_gt0_forAllvegType(data_mean,  data_pvalue, 'cor mean', -0.6, 0.1, 'All',ax = ax2)

    ### fig1 b
    plot_cor_mean_or_slope_forDiffvegType_and_AI(data_mean, veg_type_data, ai_type_data, 'cor mean', ax = ax3)

    ### fig1 c left
    plot_cor_mean_or_slope_and_pvalue_forAllvegType(data_slope, data_pvalue, -0.03, 0.03,'cor slope', 'All',ax = ax4)

    ### fig1 c right
    plot_cor_mean_or_slope_lt0_sig_and_lt0_nosig_and_gt0_forAllvegType(data_slope, data_pvalue, 'cor slope', -0.03, 0.03, 'All',ax = ax5)

    ### fig1 d
    plot_cor_mean_or_slope_forDiffvegType_and_AI(data_slope, veg_type_data, ai_type_data, 'cor slope', ax = ax6)

    # plt.savefig(fr'D:\CAU\phenology_swc_vpd\Global_test4\Fig\3Cor_mean_slope\Cor17_8_2_mean_Slope_pvalue.png', dpi=600, bbox_inches='tight')
    plt.savefig(fr'D:\CAU\phenology_swc_vpd\Global_test5\Fig\Fig 3 SM-VPD coupling mean and trend\All\Cor17_8_2_mean_Slope_pvalue.png', dpi=300, bbox_inches='tight')

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

    # 1. 数据准备
    veg_data_list = [np.where(veg_type_data == i, data, np.nan) for i in [1, 2, 3, 4]]
    veg_pvalue_list = [np.where(veg_type_data == i, data_pvalue, np.nan) for i in [1, 2, 3, 4]]
    veg_labels = ['Forests', 'Shrublands', 'Savannas', 'Grasslands']

    ai_data_list = [
        np.where(ai_type_data == 2, data, np.nan),  # Arid
        np.where((ai_type_data == 3) | (ai_type_data == 4), data, np.nan),  # Semi-Arid (合并 3 和 4)
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

    # 统一设置所有字体大小
    plt.rcParams.update({
        'font.family': 'Arial',

        'mathtext.fontset': 'custom',

        'mathtext.rm': 'Arial',  # 正常
        'mathtext.it': 'Arial:italic',  # 斜体
        'mathtext.bf': 'Arial:bold',  # 粗体

        # 可选（推荐加）
        'mathtext.default': 'regular',  # 避免自动变斜体

        'font.size': 9,
        'axes.titlesize': 9,
        'axes.labelsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        # 'text.usetex': False,  # 不使用外部LaTeX
    })

    ############################################## 植被类型 ##################################################
    fig = plt.figure(figsize=(8.4, 6.5))
    gs = gridspec.GridSpec(2, 5,
                           width_ratios=[6, 4, 0.35, 6, 4],  # 三列的宽度比
                           height_ratios=[1, 1],  # 最后一个给colorbar
                           hspace=0.45, wspace=0.6)

    ax1 = plt.subplot(gs[0, 0])  ## 森林 left
    ax2 = plt.subplot(gs[0, 1])  ## 森林 right
    ax3 = plt.subplot(gs[0, 3])  ## shrub left
    ax4 = plt.subplot(gs[0, 4])  ## shrub right

    ax5 = plt.subplot(gs[1, 0])  ## savanna left
    ax6 = plt.subplot(gs[1, 1])  ## savanna right
    ax7 = plt.subplot(gs[1, 3])  ## grass left
    ax8 = plt.subplot(gs[1, 4])  ## grass right

    ax_pairs = [(ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)]

    for veg_data, veg_pvalue, veg_name, (ax_l, ax_r) in zip(veg_data_list, veg_pvalue_list, veg_labels, ax_pairs):
        # 绘制左侧地图组 (ax_l)
        plot_cor_mean_or_slope_and_pvalue_forAllvegType(veg_data, veg_pvalue, colorbarmin, colorbarmax, data_type, veg_name, ax=ax_l)

        # 绘制右侧柱状图组 (ax_r)
        plot_cor_mean_or_slope_lt0_sig_and_lt0_nosig_and_gt0_forAllvegType(veg_data, veg_pvalue, data_type, colorbarmin, colorbarmax, veg_name, ax=ax_r)

    # plt.savefig(fr'D:\CAU\phenology_swc_vpd\Global_test4\Fig\3Cor_mean_slope\Cor17_8_2_{data_type}_Vegtype.png', dpi=600, bbox_inches='tight')
    plt.savefig(fr'D:\CAU\phenology_swc_vpd\Global_test5\Fig\Fig 3 SM-VPD coupling mean and trend\Veg\Cor17_8_2_{data_type}_Vegtype.png', dpi=300, bbox_inches='tight')

    # plt.show()

    ############################################## AI类型 ##################################################
    fig = plt.figure(figsize=(8.4, 6.5))
    gs = gridspec.GridSpec(2, 5,
                           width_ratios=[6, 4, 0.8, 6, 4],  # 三列的宽度比
                           height_ratios=[1, 1],  # 最后一个给colorbar
                           hspace=0.45, wspace=0.5)

    ax1 = plt.subplot(gs[0, 0])  ## 森林 left
    ax2 = plt.subplot(gs[0, 1])  ## 森林 right
    ax3 = plt.subplot(gs[0, 3])  ## shrub left
    ax4 = plt.subplot(gs[0, 4])  ## shrub right

    ax5 = plt.subplot(gs[1, 0])  ## savanna left
    ax6 = plt.subplot(gs[1, 1])  ## savanna right
    ax7 = plt.subplot(gs[1, 3])  ## grass left
    ax8 = plt.subplot(gs[1, 4])  ## grass right



    ax_pairs = [(ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)]

    for ai_data, ai_pvalue, ai_name, (ax_l, ax_r) in zip(ai_data_list, ai_pvalue_list, ai_labels, ax_pairs):
        # 绘制左侧地图组 (ax_l)
        # 注意：我假设你修改了函数签名，加入了 veg_name
        plot_cor_mean_or_slope_and_pvalue_forAllvegType(ai_data, ai_pvalue, colorbarmin, colorbarmax, data_type, ai_name, ax=ax_l)

        # 绘制右侧柱状图组 (ax_r)
        plot_cor_mean_or_slope_lt0_sig_and_lt0_nosig_and_gt0_forAllvegType(ai_data, ai_pvalue, data_type, colorbarmin, colorbarmax, ai_name, ax=ax_r)

    # plt.savefig(fr'D:\CAU\phenology_swc_vpd\Global_test4\Fig\3Cor_mean_slope\Cor17_8_2_{data_type}_AItype.png', dpi=600, bbox_inches='tight')
    plt.savefig(fr'D:\CAU\phenology_swc_vpd\Global_test5\Fig\Fig 3 SM-VPD coupling mean and trend\AI\Cor17_8_2_{data_type}_AItype.png', dpi=300, bbox_inches='tight')

    # # plt.show()


plot_fig3(cor_mean, cor_slope, cor_slope_pvalue)
print('Fig3 plot done!')
plot_S9_12_forMean_or_Slope(cor_mean, cor_slope, cor_slope_pvalue, 'cor mean')
print('S9-10 plot done!')
plot_S9_12_forMean_or_Slope(cor_mean, cor_slope, cor_slope_pvalue, 'cor slope')
print('S11-12 plot done!')


