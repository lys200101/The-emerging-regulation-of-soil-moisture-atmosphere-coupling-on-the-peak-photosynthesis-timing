import os.path
import glob
import sys
from statistics import linear_regression

import numpy as np
from numpy.core.multiarray import bincount

from osgeo import gdal
import pandas as pd
import pymannkendall as mk
import matplotlib.pyplot as plt
import matplotlib as mpl
from patsy.origin import Origin

from scipy.stats import alpha
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from pygam import LinearGAM, s
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import r2_score

from mpl_toolkits.basemap import Basemap
from joblib import Parallel, delayed
from scipy.stats import theilslopes
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from matplotlib.ticker import FormatStrFormatter


################################ 1 输入及输出 #############################
##### 输入设定 #####

star_year = 2001
end_year = 2024
years_length = end_year - star_year + 1
print('years_length:', years_length)

only_analyses_significant = 'No'   ###Yes则只分析POS或Cor Pvalue<0.05的像元，No则分析所有像元

Outlier = 'No'  # Yes / No
OutnosigCor = 'No'  # Yes / No

scale = 55

same_input_path = 'D:\CAU\phenology_swc_vpd\paper\Data_and_code_online\FigShare_data'

pos_input = rf'{same_input_path}\{scale}km\POS_55km'

folder_cor_1 = fr'{same_input_path}\{scale}km\Climate_data\Correlation(SM_VPD_pearson)1'  #(POS-30) - POS
folder_cor_2 = fr'{same_input_path}\{scale}km\Climate_data\Correlation(SM_VPD_pearson)2'  #(POS-60) - POS
folder_cor_3 = fr'{same_input_path}\{scale}km\Climate_data\Correlation(SM_VPD_pearson)3'  #(POS-90) - POS

folder_cor_pvalue_1 = fr'{same_input_path}\{scale}km\Climate_data\Correlation(SM_VPD_pearson)1\Pvalue'  #(POS-30) - POS
folder_cor_pvalue_2 = fr'{same_input_path}\{scale}km\Climate_data\Correlation(SM_VPD_pearson)2\Pvalue'  #(POS-60) - POS
folder_cor_pvalue_3 = fr'{same_input_path}\{scale}km\Climate_data\Correlation(SM_VPD_pearson)3\Pvalue'  #(POS-90) - POS

pos_tiffiles = sorted(glob.glob(os.path.join(pos_input, '*.tif')))

tif_files_cor_1 = sorted(glob.glob(os.path.join(folder_cor_1, '*.tif')))
tif_files_cor_2 = sorted(glob.glob(os.path.join(folder_cor_2, '*.tif')))
tif_files_cor_3 = sorted(glob.glob(os.path.join(folder_cor_3, '*.tif')))
cor_length_tif = os.path.join(input_path, rf'{same_input_path}\{scale}km\Climate_data\Best_preseason_length\17_8_1\Cor_preseason_length.tif')

tif_files_cor_pvalue_1 = sorted(glob.glob(os.path.join(folder_cor_pvalue_1, '*.tif')))
tif_files_cor_pvalue_2 = sorted(glob.glob(os.path.join(folder_cor_pvalue_2, '*.tif')))
tif_files_cor_pvalue_3 = sorted(glob.glob(os.path.join(folder_cor_pvalue_3, '*.tif')))

#### 输入AI的tif path
ai_tif_file = rf'{same_input_path}\AI\NH30_84_AI(graident)_{scale}km.tif'

#### 输入的植被类型 tif path
veg_type_file = rf'{same_input_path}\Veg_type\NH_veg_type_{scale}km(Python).tif'

#### 输入的耦合梯度 tif path
cor_mean_file = fr'{same_input_path}\mean\SM_VPD_Cor17_8_0\Cor_mean_{scale}km_All.tif'  #SOS - POS

##### 输出设定 #####
fig_output = r'D:\CAU\phenology_swc_vpd\paper\Data_and_code_online\Result'


############################### 基本信息 ###############################
sample = tif_files_cor_1[0]
sample_tif = gdal.Open(sample)

sample_array = sample_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)

crs = sample_tif.GetProjectionRef()
gt = sample_tif.GetGeoTransform()

rows = sample_array.shape[0]
cols = sample_array.shape[1]
print(f'rows={rows} , cols={cols}')

row_indices = np.repeat(np.arange(rows), cols)
col_indices = np.tile(np.arange(cols), rows)

lon_min = gt[0]
lon_max = gt[0] + gt[1]*cols
lat_min = gt[3] + gt[5]*rows
lat_max = gt[3]

################################ 2 堆叠 ################################
def get_band(tif, stack):
    tif_data = gdal.Open(tif)
    tif_array = tif_data.GetRasterBand(1).ReadAsArray().astype(np.float32)
    stack.append(tif_array)

pos_stack = []
cor_stack_1 = []
cor_stack_2 = []
cor_stack_3 = []

cor_pvalue_stack_1 = []
cor_pvalue_stack_2 = []
cor_pvalue_stack_3 = []

for tif_file in pos_tiffiles:
    get_band(tif_file, pos_stack)

for tif_file in tif_files_cor_1:
    get_band(tif_file, cor_stack_1)
for tif_file in tif_files_cor_2:
    get_band(tif_file, cor_stack_2)
for tif_file in tif_files_cor_3:
    get_band(tif_file, cor_stack_3)

cor_length_tif = gdal.Open(cor_length_tif)
cor_length = cor_length_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)

for tif_file in tif_files_cor_pvalue_1:
    get_band(tif_file, cor_pvalue_stack_1)
for tif_file in tif_files_cor_pvalue_2:
    get_band(tif_file, cor_pvalue_stack_2)
for tif_file in tif_files_cor_pvalue_3:
    get_band(tif_file, cor_pvalue_stack_3)

pos_stack = np.stack(pos_stack, axis=0)

cor_stack_1 = np.stack(cor_stack_1, axis=0)#[:, 505:510, 505:510]
cor_stack_2 = np.stack(cor_stack_2, axis=0)
cor_stack_3 = np.stack(cor_stack_3, axis=0)
cor_stack = np.stack([cor_stack_1[:years_length, :, :], cor_stack_2[:years_length, :, :], cor_stack_3[:years_length, :, :]], axis=0)

cor_pvalue_stack_1 = np.stack(cor_pvalue_stack_1, axis=0)#[:, 505:510, 505:510]
cor_pvalue_stack_2 = np.stack(cor_pvalue_stack_2, axis=0)
cor_pvalue_stack_3 = np.stack(cor_pvalue_stack_3, axis=0)
cor_pvalue_stack = np.stack([cor_pvalue_stack_1[:years_length, :, :], cor_pvalue_stack_2[:years_length, :, :], cor_pvalue_stack_3[:years_length, :, :]], axis=0)


#### 植被类型数据
veg_type_tif = gdal.Open(veg_type_tif)
veg_type_data = veg_type_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
print('veg_type_data shape:', veg_type_data.shape)

ai_tif = gdal.Open(ai_tif)
ai_type_data = ai_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)

#### 耦合梯度数据
cor_mean_file = gdal.Open(cor_mean_file)
cor_mean_data = cor_mean_file.GetRasterBand(1).ReadAsArray().astype(np.float32)
print('cor_mean_data shape:', cor_mean_data.shape)

########################### 4 逐像元根据最佳季前长度获取数据 #################################
cor_pre_stack = np.full((years_length, rows, cols), np.nan)
cor_pvalue_pre_stack = np.full((years_length, rows, cols), np.nan)

def get_preseason_data(i, j,
                       cor_len, cor_data, cor_pvalue_data):

    def extract_by_len(length_val, data_stack):
        if np.isnan(length_val):
            return np.full(years_length, np.nan)
        idx = int(length_val) - 1  # 关键：将 1,2,3 转为 0,1,2
        return data_stack[idx, :]

    cor_preseason_value = extract_by_len(cor_len, cor_data)
    cor_pvalue_preseason_value = extract_by_len(cor_len, cor_pvalue_data)

    return (i, j, cor_preseason_value, cor_pvalue_preseason_value)

results = Parallel(n_jobs=15)(
    delayed(get_preseason_data)(
        i, j ,
        cor_length[i, j], cor_stack[:, :, i, j], cor_pvalue_stack[:, :, i, j]
    )for i, j in zip(row_indices, col_indices)
)

for (i, j, cor_preseason_value, cor_pvalue_preseason_value) in results:
    cor_pre_stack[:, i, j] = cor_preseason_value
    cor_pvalue_pre_stack[:, i, j] = cor_pvalue_preseason_value

print('Preseason match done!')


############################### 3 去异常值 ###################################
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
        if len(x_flatten) < (years_length/2):
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
                return x_masked, i, j  # IQR, lower_range, upper_range,  minv, maxv
            else:
                print(f'IQR threshold：{lower_range:.4f} ~ {upper_range:.4f}\n'
                      f'像元的原始数据:{x}\n'
                      f'去异常值后无效像元超一半：{x_masked}')
                return np.full_like(x, np.nan), i, j


    outlier_pos_stack = np.full((years_length, rows, cols), np.nan)

    outlier_cor_stack = np.full((years_length, rows, cols), np.nan)

    ### IQR去 POS 异常值
    results = Parallel(n_jobs=18, verbose=10)(
        delayed(Outlier_array_IQR)(
            pos_stack[:, i, j], i, j, 25, 75
        ) for i, j in zip(row_indices, col_indices)
    )

    for data_mask, i, j in results:
        outlier_pos_stack[:, i, j] = data_mask


    ### IQR去 Cor 异常值
    results = Parallel(n_jobs=18, verbose=10)(
        delayed(Outlier_array_IQR)(
            cor_pre_stack[:, i, j], i, j, 25, 75
        ) for i, j in zip(row_indices, col_indices)
    )

    for data_mask, i, j in results:
        outlier_cor_stack[:, i, j] = data_mask

    mask = np.isfinite(outlier_pos_stack) & np.isfinite(outlier_cor_stack)


elif Outlier == 'No':
    mask = np.isfinite(pos_stack) & np.isfinite(cor_pre_stack)

    # cor_mean = np.nanmean(cor_pre_stack, axis=0)

if OutnosigCor == 'Yes':
    mask = (cor_pvalue_pre_stack <= 0.1) & mask

valid_pixel_count = np.nansum(mask, axis=0)

# 创建有效像素的掩码
valid_pixel_mask = valid_pixel_count > (years_length/2)  # (years_length - 3)
print(f'去异常值且配对后有效年份>12年的像元数量:{np.count_nonzero(valid_pixel_mask)}')

# 应用掩码到所有时间步
if Outlier == 'Yes':
    pos_stack_clean = np.where(valid_pixel_mask, outlier_pos_stack, np.nan)
    cor_stack_clean = np.where(valid_pixel_mask, outlier_cor_stack, np.nan)
if Outlier == 'No':
    pos_stack_clean = np.where(valid_pixel_mask, pos_stack, np.nan)
    cor_stack_clean = np.where(valid_pixel_mask, cor_pre_stack, np.nan)
print(f'pos_stack_clean shape:{pos_stack_clean.shape[0]}')



################################ 4 POS和Cor sen’s slope/mean 计算 #########################
def calculate_senSlope(data):

    mask = np.isfinite(data)

    data_clean = data[mask]
    years_valid = years[mask]

    if (len(data_clean) > (years_length/2)) and (len(data_clean) <= years_length):   ### >10 / >17/ >19

        # Sen slope
        result = theilslopes(data_clean, years_valid)

        slope = result.slope
        intercept = result.intercept

        # Mann-Kendall
        mk_result = mk.original_test(data_clean)

        pvalue = mk_result.p

    else:
        ############# 如果 len(data_nodrought_clean) <= 10 可以考虑用滑动窗口来计算 ###########
        slope = np.nan
        pvalue = np.nan
        intercept = np.nan

    return slope, pvalue


# ###POS mean
pos_mean = np.nanmean(pos_stack_clean, axis=0)


years = np.arange(star_year, star_year + pos_stack.shape[0])
print(f'years shape:{years.shape[0]}')

###POS sen's slope
pos_results = np.apply_along_axis(calculate_senSlope,
                                    axis=0,  # 沿着时间维度
                                    arr=pos_stack_clean)
pos_slope = pos_results[0, :, :]
pos_slope_p = pos_results[1, :, :]


### Cor mean
cor_mean = np.nanmean(cor_stack_clean, axis=0)

###Cor sen's slope
cor_results = np.apply_along_axis(calculate_senSlope,
                                    axis=0,  # 沿着时间维度
                                    arr=cor_stack_clean)
cor_slope = cor_results[0, :, :]
cor_slope_p = cor_results[1, :, :]


print('--此处以下是二维密度图--')

################################ 5 Plot(二维密度图+散点图） #####################################
## ！！！只保留显著像元：
if only_analyses_significant == 'Yes':
    significant_mask = (np.isfinite(pos_slope_p)) & (np.isfinite(cor_slope_p)) & ((pos_slope_p < 0.05)|(cor_slope_p < 0.05))
    print(f'POS或Cor变化显著的像元数量：{np.count_nonzero(significant_mask)}')
    pos_slope = np.where(significant_mask, pos_slope, np.nan)
    cor_slope = np.where(significant_mask, cor_slope, np.nan)


### 线性拟合
def linegres_fit(data1, data2):
    mask = np.isfinite(data1) & np.isfinite(data2)
    x_clean = data1[mask]
    y_clean = data2[mask]

    # # 分割数据
    # x_train, x_test, y_train, y_test = train_test_split(
    #     x_clean, y_clean, test_size=test_size, random_state=random_state)
    slope, intercept, r_value, p_value, stderr = stats.linregress(x_clean, y_clean)
    r2 = r_value ** 2
    # y_pred = slope * x_test + intercept
    #
    # r2_fit_linear = r2_score(y_test, y_pred)

    x_fit_linear = np.linspace(x_clean.min(), x_clean.max(), 100)
    y_fit_linear = slope * x_fit_linear + intercept

    return  x_fit_linear, y_fit_linear, slope, intercept, r2, p_value


def gam_fit_for_spatial_data_onlyXY(data1, data2, n_splines, weights):

    mask = ~(np.isnan(data1) | np.isnan(data2))

    x_clean = data1[mask]
    y_clean = data2[mask]

    if weights is not None:
        weights_clean = weights[mask]
    else:
        weights_clean = None

    # train-test split
    if weights_clean is not None:
        x_train, x_test, y_train, y_test, w_train, w_test = train_test_split(
            x_clean,
            y_clean,
            weights_clean,
            test_size=0.2,
            random_state=42
        )
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            x_clean,
            y_clean,
            test_size=0.2,
            random_state=42
        )

    # sort
    sort_idx = np.argsort(x_train)

    x_train_sorted = x_train[sort_idx]
    y_train_sorted = y_train[sort_idx]

    if weights_clean is not None:
        w_train_sorted = w_train[sort_idx]

    # GAM
    gam = LinearGAM(
        s(0, n_splines=n_splines)
    )

    if weights_clean is not None:
        gam.fit(
            x_train_sorted,
            y_train_sorted,
            weights=w_train_sorted
        )
    else:
        gam.fit(
            x_train_sorted,
            y_train_sorted
        )

    # test
    y_test_pred = gam.predict(x_test)

    if weights_clean is not None:
        test_r2 = r2_score(
            y_test,
            y_test_pred,
            sample_weight=w_test
        )
    else:
        test_r2 = r2_score(
            y_test,
            y_test_pred
        )

    # fit curve
    x_fit = np.linspace(
        x_clean.min(),
        x_clean.max(),
        300
    )

    y_fit = gam.predict(x_fit)

    return x_fit, y_fit, test_r2




# ################################ 6 Plot(二维分布图） #####################################
## ======== 5.1 二维colorbar定义 ======== ##
#### mean:
mean_lower_bound_x = -0.7
mean_upper_bound_x = 0.1
mean_lower_bound_y = 120
mean_upper_bound_y = 280

mean_bin_size_x = 0.1
mean_bin_size_y = 20

mean_bins = (mean_upper_bound_x - mean_lower_bound_x)/mean_bin_size_x
mean_bins = int(round(mean_bins))

#### Slope:
slope_lower_bound_x = -0.024
slope_upper_bound_x = 0.024
slope_lower_bound_y = -1
slope_upper_bound_y = 1

slope_bin_size_x = 0.006
slope_bin_size_y = 0.25

slope_bins = (slope_upper_bound_x - slope_lower_bound_x)/slope_bin_size_x
slope_bins = int(round(slope_bins))


## ======== 5.2 准备双变量颜色矩阵 ======== ##
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import matplotlib.gridspec as gridspec


## 插值
def create_bivariate_colormap(nquantiles, x_left_color, x_right_y_bottom_color, y_top_color):

    n = nquantiles

    col_matrix = np.zeros((n, n, 3))

    for i in range(n):      # i: 上→下
        for j in range(n):  # j: 左→右

            x = j / (n - 1)          # 左→右：0→1
            y = 1-i / (n - 1)      # 上→下：0→1（修正：1-i/(n-1) 需要加括号）

            # 四角权重（标准双线性插值）
            w_ll = (1 - x) * (1 - y)   # 左上
            w_lr = x * (1 - y)         # 右上
            w_ul = (1 - x) * y         # 左下
            w_ur = x * y               # 右下

            # 四个角的颜色
            c_ll = np.array([0, 1, 0])  # 左上
            c_lr = y_top_color   # 右上
            c_ul = x_left_color   # 左下
            c_ur = x_right_y_bottom_color  # 右下

            color = (
                w_ll * c_ll +
                w_lr * c_lr +
                w_ul * c_ul +
                w_ur * c_ur
            )

            col_matrix[i, j] = color

    return col_matrix

def plot_bivariate_colorbar(col_matrix, xlabel='x', ylabel='y'):
    """
    绘制二维 colorbar
    """

    n = col_matrix.shape[0]

    fig, ax = plt.subplots(figsize=(4, 4))

    # imshow 直接画颜色矩阵
    ax.imshow(col_matrix, origin='lower')

    # 设置刻度（可选）
    ax.set_xticks([0, n-1])
    ax.set_yticks([0, n-1])

    ax.set_xticklabels(['Low', 'High'])
    ax.set_yticklabels(['Low', 'High'])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # 去掉网格
    ax.grid(False)

    plt.tight_layout()
    plt.show()


x_left = np.array([1, 1, 0])
y_top = np.array([0, 0, 1])
x_right_y_bottom  = np.array([1, 1, 1])

col_matrix = create_bivariate_colormap(8, x_left, x_right_y_bottom, y_top)


#### ========== 5.3 数据量化 =========== ####
def read_and_process_raster(data_x, data_y, lower_bound_x, upper_bound_x, lower_bound_y, upper_bound_y, bin_size_x, bin_size_y):
    # 过滤数据（将数据限制在lower_bound到upper_bound之间），同时保留 NaN 值
    # 创建一个掩码来标记 NaN 值
    mask_x = np.isnan(data_x)
    mask_y = np.isnan(data_y)

    data_x = np.clip(data_x, lower_bound_x, upper_bound_x)
    data_y = np.clip(data_y, lower_bound_y, upper_bound_y)

    # 使用掩码将 NaN 值恢复到原始位置
    data_x[mask_x] = np.nan
    data_y[mask_y] = np.nan

    quantized_data_x = np.full((rows, cols), np.nan)  # 初始化量化数据为 NaN
    quantized_data_y = np.full((rows, cols), np.nan)  # 初始化量化数据为 NaN

    for i in range(rows):
        for j in range(cols):

            ### 量化x：
            if not np.isnan(data_x[i, j]):  # 检查当前值是否为 NaN

                    # 计算量化值，得到的quantized_value_x对应colorbar的x坐标
                    quantized_value_x = (data_x[i, j] - lower_bound_x) // bin_size_x

                    if quantized_value_x == int((upper_bound_x-lower_bound_x)/bin_size_x):
                        quantized_data_x[i, j] = quantized_value_x-1
                    else:
                        quantized_data_x[i, j] = quantized_value_x

            else:
                quantized_data_x[i, j] = np.nan  # 保留 NaN 值

            ### 量化y：
            if not np.isnan(data_y[i, j]):  # 检查当前值是否为 NaN
                quantized_value_y = (data_y[i, j] - lower_bound_y) // bin_size_y
                if quantized_value_y == int((upper_bound_y-lower_bound_y)/bin_size_y):
                    quantized_data_y[i, j] = quantized_value_y-1
                else:
                    quantized_data_y[i, j] = quantized_value_y
                # print('quantized_data_y[i, j]:', i, j, quantized_data_y[i, j])
            else:
                quantized_data_y[i, j] = np.nan  # 保留 NaN 值

    return quantized_data_x, quantized_data_y


if only_analyses_significant == 'Yes':
    ###  ！！！只保留POSslope显著的像元
    significant_mask = ((pos_slope_p < 0.05)|(cor_slope_p < 0.05)) & np.isfinite(pos_slope_p) & np.isfinite(cor_slope_p)

    pos_slope_significant = np.where(significant_mask, pos_slope, np.nan)
    cor_slope_significant = np.where(significant_mask, cor_slope, np.nan)

    slope_y = pos_slope_significant
    ### 注意，下面的pos_slope要记得更改！！！
    slope_x = cor_slope_significant

    # std_y = pos_std
    # std_x = cor_std
else:
    mean_y = pos_mean
    mean_x = cor_mean

    slope_y = pos_slope
    slope_x = cor_slope


cormean_quantily, posmean_quantily = read_and_process_raster(mean_x, mean_y,
                                                              mean_lower_bound_x, mean_upper_bound_x,
                                                              mean_lower_bound_y, mean_upper_bound_y,
                                                              mean_bin_size_x, mean_bin_size_y)

corslope_quantily, posslope_quantily = read_and_process_raster(slope_x, slope_y,
                                                              slope_lower_bound_x, slope_upper_bound_x,
                                                              slope_lower_bound_y, slope_upper_bound_y,
                                                              slope_bin_size_x, slope_bin_size_y)



def assign_rgb_to_combined_matrix(data_x, data_y, combined_matrix):
    rgb_matrix = np.zeros((rows, cols, 3))  # 存储提取的RGB颜色

    # 根据量化后的坐标组合提取RGB颜色
    for i in range(rows):
        for j in range(cols):
            # 获取第一张栅格量化值 x 和第二张栅格量化值 y
            x = data_x[i, j]
            y = data_y[i, j]

            # 如果 x 或 y 是 NaN，则赋值为空白透明色
            if np.isnan(x) or np.isnan(y):
                rgb_matrix[i, j] = [1, 1, 1]  # 设置透明色（RGB全0）
            else:
                x = int(x)
                y = int(y)
                # 从combined_matrix提取RGB值
                rgb_matrix[i, j] = combined_matrix[y, x, :3]  # 提取RGB颜色，忽略透明度

    return rgb_matrix

biv_map_cormean_posmean = assign_rgb_to_combined_matrix(cormean_quantily, posmean_quantily, col_matrix)
biv_map_corslope_posslope = assign_rgb_to_combined_matrix(corslope_quantily, posslope_quantily, col_matrix)


########### ====== 5.4 像元上的POS与Cor的线性关系 ###########
def cal_pixel_pos_cor_linear(i, j, pos_data, cor_data):

    mask = np.isfinite(pos_data) & np.isfinite(cor_data)

    if np.sum(mask) > years_length/2 :
        x = cor_data[mask].flatten()
        y = pos_data[mask].flatten()

        slope, intercept, r_value, p_value, stderr = stats.linregress(x, y)
        r2 = r_value ** 2
    else:
        slope = r2 = p_value = np.nan

    return i, j, slope, r2, p_value

cor_pos_pixel_linear_slope = np.full_like(pos_stack_clean[0, :, :], np.nan)
cor_pos_pixel_linear_r2 = np.full_like(pos_stack_clean[0, :, :], np.nan)
cor_pos_pixel_linear_p = np.full_like(pos_stack_clean[0, :, :], np.nan)

results = Parallel(n_jobs=18, verbose=10)(
    delayed(cal_pixel_pos_cor_linear)(
        i, j,
        pos_stack_clean[:, i, j],
        cor_stack_clean[:, i, j]
    )
    for i, j in zip(row_indices, col_indices)
)

for i, j, slope, r2, p in results:
    cor_pos_pixel_linear_slope[i, j] = slope
    cor_pos_pixel_linear_r2[i, j] = r2
    cor_pos_pixel_linear_p[i, j] = p



############### ========= 绘制结果 ========== #############
def bivariate_map(ax, data, lower_bound_x, upper_bound_x, size_x,
                  lower_bound_y, upper_bound_y, size_y,
                  colorbar_matrix, xlabel, ylabel):

    plots = []

    plot_data_filter = data


    ########### 子图1：空间分布 #################
    ax.set_box_aspect(1)  # 强制地图轴的形状为正方，使其直径撑满格子高度
    ax.axis('off')
    ### 创建地图
    m = Basemap(ax=ax,
                projection='npstere',  # 北极投影
                boundinglat=30,  # 最低显示纬度（你现在是30N）
                lon_0=0,  # 中心经度（可以改） 180:太平洋居中；90：亚洲居中
                resolution='l')

    # 生成网格坐标
    lons = np.linspace(-180, 180, cols, endpoint=False)
    lats = np.linspace(lat_max, lat_min, rows)
    lons, lats = np.meshgrid(lons, lats)

    # 设置经纬度刻度
    m.drawparallels(np.arange(30, 90, 30),
                    labels=[0, 0, 0, 0],
                    linewidth=0.5,
                    # fontsize=8,
                    color="black")

    m.drawmeridians(np.arange(0, 360, 60),
                    latmax=90,  # 使经度线在北极交汇
                    labels=[0, 0, 0, 0],  # labels=[left, right, top, bottom] 控制经度显示与否
                    linewidth=0.5,
                    # fontsize=8,
                    color="gray")

    ### 绘制数据
    # 绘制
    plot = m.pcolormesh(lons, lats,
                        plot_data_filter,
                        shading='nearest',
                        latlon=True,
                        zorder=1)  #控制所画的图所在层，zorder=1（底层），zorder=3（顶层）

    plots.append(plot)  # 保存plot对象

    # 绘制边界
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
        threshold = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.1

        # 找到跳变点的索引
        break_indices = np.where(dist > threshold)[0]

        if len(break_indices) == 0:
            # 没有跳变，直接画整条线
            ax.plot(x, y, color='black', linewidth=0.3, zorder=3)
        else:
            # 有跳变，将线段切断，分段画出
            # 这样既能去掉横跨圆心的直线，又能保留正常的边界
            start_idx = 0
            for break_idx in break_indices:
                ax.plot(x[start_idx:break_idx + 1], y[start_idx:break_idx + 1],
                         color='black', linewidth=0.3, zorder=3)
                start_idx = break_idx + 1
            # 画最后一段
            ax.plot(x[start_idx:], y[start_idx:], color='black', linewidth=0.3, zorder=3)

    # x_pole, y_pole = m(0, 90)
    # ax.scatter(x_pole, y_pole, color='black', s=5, zorder=3)

    ### 最外边界的裁剪
    from matplotlib.patches import Circle

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    center = [(x0 + x1) / 2, (y0 + y1) / 2]
    radius = (x1 - x0) / 2

    clip_circle = Circle(center, radius, transform=ax.transData)

    for artist in ax.collections + ax.lines + ax.patches:
        artist.set_clip_path(clip_circle)

    boundary_circle = Circle(
        center,
        radius,
        transform=ax.transData,
        facecolor='none',
        edgecolor='black',  # 颜色
        linewidth=0.8,
        zorder=4  # 放最上层
    )

    ax.add_patch(boundary_circle)

    #######  colorbar  ############
    inset_pos = [0.01, 0, 0.28, 0.28]
    axins = ax.inset_axes(inset_pos)
    cbar = axins

    n = colorbar_matrix.shape[0]

    # 设置正确的extent，但通过调整ax_cbar的宽高比来保持正方形
    cbar.imshow(colorbar_matrix,
                interpolation='nearest',
                extent=[0, n, 0, n],
                origin='lower')  # 数组的第 0 行（row=0）显示在图的“底部”

    def value_to_index(value, lower, size):
        return (value - lower) / size

    if ylabel == 'POS trend':
        x0 = value_to_index(0, lower_bound_x, size_x)
        y0 = value_to_index(0, lower_bound_y, size_y)

        cbar.axvline(x=x0, color='black', linestyle='--', linewidth=0.5)
        cbar.axhline(y=y0, color='black', linestyle='--', linewidth=0.5)

    # elif ylabel == 'POS mean':
    #     x_line = value_to_index(-0.3, lower_bound_x, size_x)
    #     y_line = value_to_index(200, lower_bound_y, size_y)
    #
    #     cbar.axvline(x=x_line, color='black', linestyle='--', linewidth=0.5)
    #     cbar.axhline(y=y_line, color='black', linestyle='--', linewidth=0.5)

    ### 修改成隔一个位置取1个刻度
    tick_idx = np.arange(0, n + 1, 2) # 0表示从第一个数开始取

    cbar.set_xticks(tick_idx)
    cbar.set_yticks(tick_idx)
    cbar.tick_params(axis='both', direction='in')

    x_labels = np.linspace(lower_bound_x, upper_bound_x, n+1)
    y_labels = np.linspace(lower_bound_y, upper_bound_y, n+1)

    cbar.set_xlabel(xlabel)

    if ylabel == 'POS mean':
        cbar.set_ylabel('PPT (days)')

        cbar.set_xticklabels([f'{0}' if abs(x) < 1e-6 else f'{x:.1f}' for x in x_labels[tick_idx]], fontsize = 9, rotation=90)
        cbar.set_yticklabels([f'{int(y)}' for y in y_labels[tick_idx]], fontsize = 9)

    elif ylabel == 'POS trend':
        cbar.set_ylabel('PPT trend (days/yr)')

        cbar.set_xticklabels([f'{0}' if abs(x) < 1e-6 else f'{(x * 100):.1f}' for x in x_labels[tick_idx]], fontsize = 9, rotation=90)
        cbar.set_yticklabels([f'{0}' if abs(y) < 1e-6 else
                              f'{int(y)}' if y == int(y) else
                              f'{y:.1f}' for y in y_labels[tick_idx]], fontsize = 9)

        ## x
        cbar.text(1.28, -0.15, r'$×10^{-2}$',
                  transform=cbar.transAxes,
                  ha='center',
                  va='center',
                  fontsize = 9)

    plt.tight_layout()

def plot_combined_hexbin_with_WeightedPointTrendline(data1, data2, xlabel, ylabel, grade_by, type, plot_weighted_point, ax):

    ### === 0 维度划分 === ###


    if grade_by == 'All':

        fig = ax.figure

        # 定义四种植被类型
        if only_analyses_significant == 'Yes':
            vmax_values = [6]  # 对应的vmax值  ### 这行vmax值仅适用于像元POS slope显著时
        else:
            if ylabel == 'POS trend':
                vmax_values = [40]  # Cor50:[55, 20, 12, 25, 12, 6]
            if ylabel == 'POS mean':
                vmax_values = [40]

        types = ['All']

    elif grade_by == 'Veg':

        fig = ax.figure

        types = ['Forest', 'Shrub', 'Savanna', 'Grass']

        if ylabel == 'POS trend':
            vmax_values = [15, 10, 20, 6]  #Cor50:[55, 20, 12, 25, 12, 6]

        if ylabel == 'POS mean':
            vmax_values = [15, 10, 20, 6]

        i = types.index(type)

        vmax_values = vmax_values[i]
        vmax_values = [vmax_values]

        types = types[i]
        types = [types]


    elif grade_by == 'AI':

        fig = ax.figure

        types = ['Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']

        if ylabel == 'POS trend':
            vmax_values = [6, 10, 10, 20]  # Cor50:[55, 20, 12, 25, 12, 6]

        if ylabel == 'POS mean':
            vmax_values = [6, 10, 10, 20]

        i = types.index(type)

        vmax_values = vmax_values[i]
        vmax_values = [vmax_values]

        types = types[i]
        types = [types]

    elif grade_by == 'Cor mean':

        fig = ax.figure

        types = ['Cor(-0.1~0)', 'Cor(-0.2~-0.1)', 'Cor(-0.3~-0.2)',
                 'Cor(-0.4~-0.3)', 'Cor(<-0.4)']

        if ylabel == 'POS trend':
            vmax_values = [10, 10, 20, 20, 6]  # Cor50:[55, 20, 12, 25, 12, 6]

        if ylabel == 'POS mean':
            vmax_values = [10, 10, 10, 20, 6]

        i = types.index(type)

        vmax_values = vmax_values[i]
        vmax_values = [vmax_values]

        types = types[i]
        types = [types]


    # 预先展平数据
    data1_flat = data1.flatten()
    data2_flat = data2.flatten()

    veg_type_flat = veg_type_data.flatten()  # 展平植被类型数据
    ai_type_flat = ai_type_data.flatten()  # 展平植被类型数据
    cor_type_flat = cor_mean_data.flatten()  # 展平植被类型数据
    print(f'veg_type_flat count:{np.count_nonzero(np.isfinite(veg_type_flat))}')

    for i, (type, vmax) in enumerate(zip(types, vmax_values)):
        ax1 = ax

        # word_list = ['a', 'b', 'c', 'd']
        # word = word_list[i]

        # 创建植被类型掩码
        if grade_by == 'All':
            mask = np.isfinite(data1_flat) & np.isfinite(data2_flat)
        elif grade_by == 'Veg':
            if type == 'Forest':
                mask = (veg_type_flat == 1) #& np.isfinite(data1_flat) & np.isfinite(data2_flat) # & (data1_flat >= -0.5) & (data1_flat < 0)
            elif type == 'Shrub':
                mask = (veg_type_flat == 2) #& np.isfinite(data1_flat) & np.isfinite(data2_flat) # & (data1_flat >= -0.5) & (data1_flat < 0)
            elif type == 'Savanna':
                mask = (veg_type_flat == 3) #& np.isfinite(data1_flat) & np.isfinite(data2_flat) # & (data1_flat >= -0.5) & (data1_flat < 0)
            elif type == 'Grass':
                mask = (veg_type_flat == 4) #& np.isfinite(data1_flat) & np.isfinite(data2_flat) # & (data1_flat >= -0.5) & (data1_flat < 0)
        elif grade_by == 'AI':
            if type == 'Hyper Arid':
                mask = (ai_type_flat == 1)  #& np.isfinite(data1_flat) & np.isfinite(data2_flat) # AI 0-0.03
            elif type == 'Arid':
                mask = (ai_type_flat == 2)  #& np.isfinite(data1_flat) & np.isfinite(data2_flat) # AI 0.03-0.2
            elif type == 'Semi-arid':
                mask = ((ai_type_flat == 3) | (ai_type_flat == 4))  #& np.isfinite(data1_flat) & np.isfinite(data2_flat)# AI 0.2-0.35
            elif type == 'Dry sub-humid':
                mask = (ai_type_flat == 5)  #& np.isfinite(data1_flat) & np.isfinite(data2_flat) # AI 0.5-0.65
            elif type == 'Humid':
                mask = (ai_type_flat == 6)  #& np.isfinite(data1_flat) & np.isfinite(data2_flat) # AI > 0.65
        elif grade_by == 'Cor mean':
            if type == 'Cor(-0.1~0)':
                mask = (-0.1 <= cor_type_flat) & (cor_type_flat < 0)  #& np.isfinite(data1_flat) & np.isfinite(data2_flat) # AI 0-0.03
            elif type == 'Cor(-0.2~-0.1)':
                mask = (-0.2 <= cor_type_flat) & (cor_type_flat < -0.1)  #& np.isfinite(data1_flat) & np.isfinite(data2_flat) # AI 0.03-0.2
            elif type == 'Cor(-0.3~-0.2)':
                mask = (-0.3 <= cor_type_flat) & (cor_type_flat < -0.2)  #& np.isfinite(data1_flat) & np.isfinite(data2_flat)# AI 0.2-0.35
            elif type == 'Cor(-0.4~-0.3)':
                mask = (-0.4 <= cor_type_flat) & (cor_type_flat < -0.3)  #& np.isfinite(data1_flat) & np.isfinite(data2_flat) # AI 0.5-0.65
            elif type == 'Cor(<-0.4)':
                mask = (cor_type_flat < -0.4) #& (-0.5 <= cor_mean_data)  #& np.isfinite(data1_flat) & np.isfinite(data2_flat) # AI > 0.65

        valid_mask = (
                np.isfinite(data1_flat) &
                np.isfinite(data2_flat) &
                mask
        )

        x = data1_flat[valid_mask]
        y = data2_flat[valid_mask]

        ### ============= 1 绘制原始散点 ============== ###
        ax1.scatter(
            x,
            y,
            s=8,
            facecolors='none',  # 空心
            edgecolors='gray',  # 红色边框
            linewidths=0.3,
            alpha=0.3,
            zorder=1
        )


        ### ============== 2 KDE ================ ###

        # xy = np.vstack([x, y]).T  # (n,2)
        #
        # # 标准化
        # scaler = StandardScaler()
        # xy_scaled = scaler.fit_transform(xy)
        #
        # # 转成 gaussian_kde 需要的格式
        # xy_scaled_T = xy_scaled.T  # (2,n)

        if ylabel == 'POS trend':
            bandwidth = 0.3
            xmin, xmax = -0.03, 0.03
            ymin, ymax = -0.6, 0.6

        elif ylabel == 'POS mean':
            bandwidth = 0.3
            xmin, xmax = -0.7, 0
            ymin, ymax = 150, 250

        xy = np.vstack([x, y])
        kde = gaussian_kde(xy, bw_method=bandwidth)  #创建 KDE 模型
        print(kde.factor)

        # 创建统一网格
        xgrid = np.linspace(xmin, xmax, 200)
        ygrid = np.linspace(ymin, ymax, 200)

        xx, yy = np.meshgrid(xgrid, ygrid)

        # 网格坐标
        grid_coords = np.vstack([xx.ravel(), yy.ravel()])

        # KDE density
        zz = kde(grid_coords).reshape(xx.shape)

        print(f'{ylabel} - {type}, zz min: {np.nanmin(zz)}')
        print(f'{ylabel} - {type}, zz max: {np.nanmax(zz)}')
        print(f'{ylabel} - {type}, zz max * 70%: {np.nanpercentile(zz, 70)}')
        print(f'{ylabel} - {type}, zz max * 75%: {np.nanpercentile(zz, 75)}')
        print(f'{ylabel} - {type}, zz max * 80%: {np.nanpercentile(zz, 80)}')
        print(f'{ylabel} - {type}, zz max * 90%: {np.nanpercentile(zz, 90)}')

        # 固定 density levels
        if ylabel == 'POS mean':
            if grade_by == 'All' or (grade_by == 'AI' and type == 'Semi-arid') or (grade_by == 'Cor mean' and type == 'Cor(<-0.4)'):
                density_min = 0.01
                density_max = 0.11
            elif (grade_by == 'Veg' and type == 'Grass') or (grade_by == 'AI' and type == 'Arid'):
                density_min = 0.01
                density_max = 0.06
            elif grade_by == 'AI' and type == 'Humid':
                density_min = 0.01
                density_max = 0.13
            elif grade_by == 'Cor mean' and type in ['Cor(-0.1~0)', 'Cor(-0.2~-0.1)']:
                density_min = 0.01
                density_max = 0.25
            elif grade_by == 'Cor mean' and type in ['Cor(-0.3~-0.2)', 'Cor(-0.4~-0.3)']:
                density_min = 0.01
                density_max = 0.15
            else:
                density_min = 0.01
                density_max = 0.2

        elif ylabel == 'POS trend':
            if grade_by == 'All':
                density_min = 10
                density_max = 85
            elif (grade_by == 'Veg' and type == 'Grass') or (grade_by == 'AI' and type == 'Arid'):
                density_min = 15
                density_max = 60
            elif grade_by == 'AI' and type == 'Semi-arid':
                density_min = 15
                density_max = 75
            elif (grade_by == 'AI' and type == 'Dry sub-humid') or (grade_by == 'AI' and type == 'Humid'):
                density_min = 15
                density_max = 105
            elif grade_by == 'Cor mean' and type == 'Cor(-0.2~-0.1)':
                density_min = 10
                density_max = 100
            elif grade_by == 'Cor mean' and type in ['Cor(-0.3~-0.2)', 'Cor(-0.1~0)']:
                density_min = 10
                density_max = 90
            elif grade_by == 'Cor mean' and type == 'Cor(-0.4~-0.3)':
                density_min = 10
                density_max = 80
            elif grade_by == 'Cor mean' and type == 'Cor(<-0.4)':
                density_min = 15
                density_max = 60
            else:
                density_min = 30
                density_max = 110


        # density_min = round(np.nanpercentile(zz, 75), 2)
        # density_max = round(np.nanpercentile(zz, 99), 2)

        levels = np.linspace(
            density_min,
            density_max,
            11
        )

        # 绘制 KDE 密度面
        cf = ax1.contourf(
            xx,
            yy,
            zz,
            levels=levels,
            cmap='YlGnBu',
            extend='max',
            alpha=0.75,
            zorder=2
        )

        ### =========== 3 基于分箱密度加权值============= ###
        # == 1 分箱 == #
        # # 直接创建颜色列表
        # if grade_by == 'Veg' or grade_by == 'AI' or only_analyses_significant == 'Yes':
        #     all_colors = ['lightgray'] * (2 - 1) + list(plt.cm.Reds(np.linspace(0, 1, vmax - 2)))
        #
        # elif grade_by == 'All':
        #     all_colors = ['lightgray'] * (5 - 1) + list(
        #         plt.cm.Reds(np.linspace(0, 1, vmax - 5)))  ###蓝红:RdBu_r; 渐变红：Reds
        #
        # cmap_custom = mcolors.ListedColormap(all_colors)

        if ylabel == 'POS trend':
            extent = [-0.03, 0.03, -2, 2]  ##Cor mean: extent = [-0.8, 0.4, -2, 2]
            gridsize = 120
        if ylabel == 'POS mean':
            extent = [-0.6, 0, 150, 250]  ##Cor mean: extent = [-0.8, 0.4, -15, 10]
            gridsize = 100

        hb = ax1.hexbin(x, y, gridsize=gridsize, extent=extent, visible=False)  # mincnt=1：显示所有包含至少1个数据点的六边形；vmin=1：颜色映射的最小值从1开始
        print(f'type: {type}')

        # == 2 密度加权 == #
        def Density_Average_Specific_Gravity(hb):
            counts = hb.get_array()  # 每个六边形的密度（计数）
            offsets = hb.get_offsets()  # 每个六边形的中心坐标 (x, y)

            # 移除零计数的六边形
            valid_mask = counts > 0
            hex_x = offsets[valid_mask, 0]
            hex_y = offsets[valid_mask, 1]
            hex_weights = counts[valid_mask]

            if xlabel == 'SM-VPD coupling':
                bin_width = 0.005
                # if veg_type == 'All':
                # x_bins = np.linspace(-0.5, -0.1, 101) #用于Cor17_5-8
                # x_bins = np.linspace(-0.5, -0.1, 81)
                if grade_by != 'Cor mean':
                    if type not in ['Grass', 'Arid', 'All']:
                        #     x_bins = np.linspace(-0.5, -0.1, 101) #用于Cor17_5-8
                        bincount = int(0.45 / bin_width)
                        x_bins = np.linspace(-0.45, 0, bincount)
                    elif type in ['Grass']:
                        bincount = int(0.4 / bin_width)
                        x_bins = np.linspace(-0.5, -0.1, bincount)
                        # x_bins = np.linspace(-0.4, 0, 101)
                    elif type in ['Arid']:
                        bincount = int(0.4 / bin_width)
                        x_bins = np.linspace(-0.5, -0.1, bincount)
                    elif type == 'All':
                        bincount = int(0.5 / bin_width)
                        x_bins = np.linspace(-0.5, 0, bincount)
                else:
                    if type in ['Cor(-0.1~0)', 'Cor(-0.2~-0.1)']:
                        bincount = int(0.4 / bin_width)
                        x_bins = np.linspace(-0.4, 0, bincount)
                    elif type == 'Cor(-0.3~-0.2)':
                        bincount = int(0.3 / bin_width)
                        x_bins = np.linspace(-0.4, -0.1, bincount)
                    elif type == 'Cor(-0.4~-0.3)':
                        bincount = int(0.3 / bin_width)
                        x_bins = np.linspace(-0.5, -0.2, bincount)
                    elif type == 'Cor(<-0.4)':
                        bincount = int(0.3 / bin_width)
                        x_bins = np.linspace(-0.6, -0.3, bincount)




            elif xlabel == 'SM-VPD coupling trend':
                # if type == 'All':
                #     x_bins = np.linspace(-0.02, 0.02, 81)
                # else:
                #     x_bins = np.linspace(-0.015, 0.015, 61)  # (-0.01, 0.008, 41)

                bin_width = 0.001
                if type in ['All']:
                    if OutnosigCor == 'No':
                        bincount = int(0.045 / bin_width)
                        x_bins = np.linspace(-0.025, 0.02, bincount)
                    else:
                        bincount = int(0.035 / bin_width)
                        x_bins = np.linspace(-0.02, 0.015, bincount)
                elif type in ['Forest', 'Cor(-0.1~0)', 'Cor(-0.2~-0.1)', 'Cor(-0.3~-0.2)']:
                    bincount = int(0.035 / bin_width)
                    x_bins = np.linspace(-0.02, 0.015, bincount)
                elif type  in ['Arid']:
                    bincount = int(0.03 / bin_width)
                    x_bins = np.linspace(-0.015, 0.015, bincount)
                elif type in ['Semi-arid']:
                    bincount = int(0.035 / bin_width)
                    x_bins = np.linspace(-0.02, 0.015, bincount)
                elif type in ['Dry sub-humid']:
                    bincount = int(0.045 / bin_width)
                    x_bins = np.linspace(-0.025, 0.02, bincount)
                elif type in ['Humid']:
                    bincount = int(0.045 / bin_width)
                    x_bins = np.linspace(-0.025, 0.02, bincount)

                elif type in ['Shrub']:
                    bincount = int(0.03 / bin_width)
                    x_bins = np.linspace(-0.02, 0.01, bincount)
                elif type in ['Savanna']:
                    bincount = int(0.03 / bin_width)
                    x_bins = np.linspace(-0.015, 0.015, bincount)
                elif type  in ['Grass']:
                    bincount = int(0.02 / bin_width)
                    x_bins = np.linspace(-0.01, 0.01, bincount)

                elif type in ['Cor(-0.4~-0.3)', 'Cor(<-0.4)']:
                    bincount = int(0.025 / bin_width)
                    x_bins = np.linspace(-0.015, 0.01, bincount)


            x_centers = (x_bins[:-1] + x_bins[1:]) / 2

            x_weighted = []
            y_weighted = []
            max_densities = []  # 存储每个区间的最大密度
            y_max = []
            y_min = []

            for i in range(len(x_bins) - 1):
                bin_mask = (hex_x >= x_bins[i]) & (hex_x < x_bins[i + 1])

                if i == len(x_bins) - 2:
                    bin_mask = (hex_x >= x_bins[i]) & (hex_x <= x_bins[i + 1])
                else:
                    bin_mask = (hex_x >= x_bins[i]) & (hex_x < x_bins[i + 1])

                if np.sum(bin_mask) > 0:
                    hex_x_bin = hex_x[bin_mask]
                    hex_y_bin = hex_y[bin_mask]
                    hex_weights_bin = hex_weights[bin_mask]

                    # 使用密度作为权重计算加权平均值
                    hex_y_weighted = np.average(hex_y_bin, weights=hex_weights_bin)
                    hex_weight_max = np.max(hex_weights_bin)

                    hex_y_max = np.nanmax(hex_y_bin)
                    hex_y_min = np.nanmin(hex_y_bin)

                    # print(f'hex_x_bin:{hex_x_bin}')
                    # print(f'hex_y_bin:{hex_y_bin}')
                    # print(f'hex_weight_max:{hex_weight_max}')

                else:
                    hex_y_weighted = np.nan
                    hex_weight_max = np.nan
                    hex_y_max = np.nan
                    hex_y_min = np.nan

                x_weighted.append(x_centers[i])
                y_weighted.append(hex_y_weighted)
                max_densities.append(hex_weight_max)
                y_max.append(hex_y_max)
                y_min.append(hex_y_min)

            x_weighted = np.array(x_weighted)
            y_weighted = np.array(y_weighted)
            max_densities = np.array(max_densities)
            y_max = np.array(y_max)
            y_min = np.array(y_min)

            #
            # print(f'vaild_x_weighted: {vaild_x_weighted}')
            # print(f'vaild_y_weighted: {vaild_y_weighted}')
            # print(f'valid_max_densities: {valid_max_densities}')

            return x_weighted, y_weighted, max_densities, y_max, y_min

        x_weighted, y_weighted, max_densities, y_max, y_min = Density_Average_Specific_Gravity(hb)

        # 计算颜色映射（基于最大密度）
        vaild_mask = np.isfinite(x_weighted) & np.isfinite(y_weighted)
        x_vaild = x_weighted[vaild_mask]
        y_vaild = y_weighted[vaild_mask]

        ### ========== 4 拟合 ============  ###
        degree = 3
        n_splines = 15


        x_fit_linear, y_fit_linear, linear_slope, linear_intercept, linear_r2, linear_p = linegres_fit(x_vaild, y_vaild)

        x_weighted_gam, y_weighted_gam, weighted_gam_r2 = gam_fit_for_spatial_data_onlyXY(
                                                            x_vaild,
                                                            y_vaild,
                                                            n_splines=n_splines,
                                                            weights=None
                                                        )

        if linear_p < 0.01:
            ax1.plot(x_fit_linear, y_fit_linear, '#11a579', linewidth=1.5,
                     label=rf'Linear (y = {linear_slope:.1f}x - {abs(linear_intercept):.1f}, $\it{{P}}$ < 0.01)', zorder=3)  #青绿色
        elif linear_p < 0.05:
            ax1.plot(x_fit_linear, y_fit_linear, '#11a579', linewidth=1.5,
                     label=rf'Linear (y = {linear_slope:.1f}x - {abs(linear_intercept):.1f}, $\it{{P}}$ <0.05)',
                     zorder=3)  # 青绿色
        elif linear_p >= 0.05:
            ax1.plot(x_fit_linear, y_fit_linear, '#11a579', linewidth=1.5,
                     label=rf'Linear (y = {linear_slope:.1f}x - {abs(linear_intercept):.1f}, $\it{{P}}$ = {linear_p:.2f})',
                     zorder=3)  # 青绿色

        # weight-GAM的趋势线
        ax1.plot(x_weighted_gam, y_weighted_gam, '#e73f74', alpha = 0.7, linewidth=1.5,
                 label=f'GAM ($R^{2}={weighted_gam_r2:.2f}$)', zorder=3)#玫红色


        ### ========== 5 细节设置 ========== ###
        # 设置坐标轴
        if xlabel == 'SM-VPD coupling':
            ax1.set_xlim(-0.7, 0)  ### 这行vmax值仅适用于像元POS slope显著时
            ax1.set_xticks(np.arange(-0.6, 0.001, 0.2))  ### 这行vmax值仅适用于像元POS slope显著时
            ax1.set_xticklabels('0' if np.isclose(x, 0)  else
                                f'{x:.1f}' for x in np.arange(-0.6, 0.001, 0.2))  ### 这行vmax值仅适用于像元POS slope显著时


        elif xlabel == 'SM-VPD coupling trend':
            ax1.set_xlim(-0.03, 0.03)
            ax1.set_xticks(np.arange(-0.02, 0.0201, 0.02))
            ax1.set_xticklabels('0' if x == 0 else
                                f'{int(x*100)}' for x in np.arange(-0.02, 0.0201, 0.02))  ### 这行vmax值仅适用于像元POS slope显著时
            ## x
            ax1.text(0.95, -0.042, r'$×10^{-2}$',
                      transform=ax1.transAxes,
                      ha='center',
                      va='center')

        if ylabel == 'POS trend':
            ticks = np.arange(-0.6, 0.61, 0.2)
            ax1.set_ylim(-0.6, 0.6)  ### 这行vmax值仅适用于像元POS slope显著时

            ax1.set_yticks(ticks)  ### 这行vmax值仅适用于像元POS slope显著时
            ax1.set_yticklabels('0' if np.isclose(x * 10, 0) else
                                f'{x:.1f}' for x in ticks)  ### 这行vmax值仅适用于像元POS slope显著时

        elif ylabel == 'POS mean':
            ticks = np.arange(150, 250.01, 25)
            ax1.set_ylim(150, 250)
            ax1.set_yticks(ticks)

            ax1.set_yticks(ticks)  ### 这行vmax值仅适用于像元POS slope显著时
            ax1.set_yticklabels(f'{int(x)}' for x in ticks)  ### 这行vmax值仅适用于像元POS slope显著时

        # if type in ['All', 'Savanna', 'Grass', 'Dry sub-humid', 'Humid']:
        ax1.set_xlabel(xlabel)
        # if type in ['All', 'Forest', 'Savanna', 'Arid', 'Dry sub-humid']:
        if ylabel == 'POS trend':
            ax1.set_ylabel('PPT trend (day/yr)')
        if ylabel == 'POS mean':
            ax1.set_ylabel('PPT (days)')


        # 添加colorbar
        cbar1 = fig.colorbar(cf, ax=ax1)

        cbar1.set_label('Density')
        if ylabel== 'POS mean':
            # 保留两位小数
            cbar1.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))



        # 添加参考线
        ax1.axhline(y=0, linestyle='--', color='gray', linewidth=0.8)
        ax1.axvline(x=0, linestyle='--', color='gray', linewidth=0.8)

        # 添加图例
        ax1.legend(loc='upper right', frameon=False, fancybox=True,
                   framealpha=0.1,     # 图例线长度
                   handletextpad=0.3,  # 线和文字距离
                   fontsize = 9)


### Plot Fig5
def plot_fig5(bivariate_data1,
              lower_bound_x1, upper_bound_x1, lower_bound_y1, upper_bound_y1, size_x1, size_y1,
              bivariate_data2,
              lower_bound_x2, upper_bound_x2, lower_bound_y2, upper_bound_y2, size_x2, size_y2,
              colorbar_matrix,  ## Bivariate map
              data_x_1, data_y_1,
              data_x_2, data_y_2,
              grade_by, plot_weighted_point): ## Density scatter plots
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

    ########   主图   #######
    # 创建5个子图：4个地图 + 1个colorbar
    fig = plt.figure(figsize=(8.2, 8.2))
    gs = gridspec.GridSpec(2, 2,
                           width_ratios=[1, 1.1],  # 四列的宽度比
                           height_ratios=[5, 5],  # 最后一个给colorbar
                           hspace=0.2, wspace=0.3)


    ### Cormean 与 POSmean 的二维空间分布
    ax1 = plt.subplot(gs[0, 0])
    bivariate_map(ax1, bivariate_data1, lower_bound_x1, upper_bound_x1, size_x1,
                  lower_bound_y1, upper_bound_y1, size_y1,
                  colorbar_matrix, 'SM-VPD coupling', 'POS mean')

    ### Cormean 与 POSmean 的二维密度图
    ax2 = plt.subplot(gs[0, 1])
    plot_combined_hexbin_with_WeightedPointTrendline(data_x_1, data_y_1, 'SM-VPD coupling', 'POS mean', grade_by, 'All',plot_weighted_point, ax = ax2)


    ### Corslope 与 POSslope 的二维空间分布
    ax3 = plt.subplot(gs[1, 0])
    bivariate_map(ax3, bivariate_data2, lower_bound_x2, upper_bound_x2, size_x2,
                  lower_bound_y2, upper_bound_y2, size_y2,
                  colorbar_matrix, 'SM-VPD coupling trend', 'POS trend')

    ### Corslope 与 POSslope 的二维密度图
    ax4 = plt.subplot(gs[1, 1])
    plot_combined_hexbin_with_WeightedPointTrendline(data_x_2, data_y_2, 'SM-VPD coupling trend', 'POS trend', grade_by, 'All',plot_weighted_point, ax = ax4)

    # plt.savefig(fr'D:\CAU\phenology_swc_vpd\Global_test4\Fig\4Pheno_trend_and_Cor\Bivariate_map\SM_VPD_Cor17\{grade_by}\SM_VPD_Cor(preseason)_BivariateMap_ScatterPlot.png',
    plt.savefig(fr'{fig_output}\SM_VPD_Cor(preseason)_BivariateMap_ScatterPlot_OutnosigCor({OutnosigCor}).png',
                dpi=300, bbox_inches='tight')
    print('Fig5 plot done!')
    # plt.show()

plot_fig5(biv_map_cormean_posmean,
          mean_lower_bound_x, mean_upper_bound_x, mean_lower_bound_y, mean_upper_bound_y, mean_bin_size_x, mean_bin_size_y,
          biv_map_corslope_posslope,
          slope_lower_bound_x, slope_upper_bound_x, slope_lower_bound_y, slope_upper_bound_y, slope_bin_size_x, slope_bin_size_y,
          col_matrix,
          cor_mean, pos_mean,
          cor_slope, pos_slope,
          'All', 'No')
print('Fig5 plot done!')

# sys.exit()

### Plot S Fig
def plot_S16_21(biv_map_data,
                               lower_bound_x, upper_bound_x, lower_bound_y, upper_bound_y, size_x, size_y,
                               col_matrix,
                               data_x, data_y,
                               xlabel, ylabel,
                               grade_by):

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from mpl_toolkits.basemap import Basemap
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib.gridspec as gridspec
    import numpy as np
    import os
    from brokenaxes import brokenaxes
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

    if grade_by == 'Veg':
        fig = plt.figure(figsize=(8.2, 14.5))
        gs = gridspec.GridSpec(4, 2,
                               width_ratios=[1, 1],  # 四列的宽度比
                               height_ratios=[5, 5, 5, 5],  # 最后一个给colorbar
                               hspace=0.3, wspace=0.15)
        # 定义四种植被类型
        types = ['Forest', 'Shrub', 'Savanna', 'Grass']#, 'Wet']
        titles = ['Forest', 'Shrub', 'Savanna', 'Grass']# 'Wet']

    elif grade_by == 'AI':
        fig = plt.figure(figsize=(8.2, 14.5))
        gs = gridspec.GridSpec(4, 2,
                               width_ratios=[1, 1],  # 四列的宽度比
                               height_ratios=[5, 5, 5, 5],  # 最后一个给colorbar
                               hspace=0.3, wspace=0.15)
        types = ['Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']
        titles = ['Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']

    elif grade_by == 'Cor mean':
        fig = plt.figure(figsize=(8.2, 16.5))
        gs = gridspec.GridSpec(5, 2,
                               width_ratios=[1, 1],  # 四列的宽度比
                               height_ratios=[5, 5, 5, 5, 5],  # 最后一个给colorbar
                               hspace=0.3, wspace=0.15)
        types = ['Cor(-0.1~0)', 'Cor(-0.2~-0.1)', 'Cor(-0.3~-0.2)',
                 'Cor(-0.4~-0.3)', 'Cor(<-0.4)']
        titles = ['Cor(-0.1~0)', 'Cor(-0.2~-0.1)', 'Cor(-0.3~-0.2)',
                 'Cor(-0.4~-0.3)', 'Cor(<-0.4)']



    for i, (type, title) in enumerate(zip(types, titles)):

        if grade_by == 'Veg':
            # 创建植被类型掩码
            if type == 'All':
                mask = np.isfinite(veg_type_data)
            elif type == 'Forest':
                ax1 = plt.subplot(gs[0, 0])  # 第一个变量空间分布
                ax2 = plt.subplot(gs[0, 1])  # 第一个变量空间分布
                mask = (veg_type_data == 1)
            elif type == 'Shrub':
                ax1 = plt.subplot(gs[1, 0])  # 第一个变量空间分布
                ax2 = plt.subplot(gs[1, 1])  # 第一个变量空间分布
                mask = (veg_type_data == 2)
            elif type == 'Savanna':
                ax1 = plt.subplot(gs[2, 0])  # 第一个变量空间分布
                ax2 = plt.subplot(gs[2, 1])  # 第一个变量空间分布
                mask = (veg_type_data == 3)
            elif type == 'Grass':
                ax1 = plt.subplot(gs[3, 0])  # 第一个变量空间分布
                ax2 = plt.subplot(gs[3, 1])  # 第一个变量空间分布
                mask = (veg_type_data == 4)
            elif type == 'Wet':
                mask = (veg_type_data == 5)
        elif grade_by == 'AI':
            # 创建植被类型掩码
            if type == 'All':
                mask = np.isfinite(ai_type_data)
            elif type == 'Hyper Arid':
                mask = (ai_type_data == 1)  # AI 0-0.03
            elif type == 'Arid':
                ax1 = plt.subplot(gs[0, 0])  # 第一个变量空间分布
                ax2 = plt.subplot(gs[0, 1])  # 第一个变量空间分布
                mask = (ai_type_data == 2)  # AI 0.03-0.2
            elif type == 'Semi-arid':
                ax1 = plt.subplot(gs[1, 0])  # 第一个变量空间分布
                ax2 = plt.subplot(gs[1, 1])  # 第一个变量空间分布
                mask = (ai_type_data == 3) | (ai_type_data == 4)  # AI 0.2-0.35
            elif type == 'Dry sub-humid':
                ax1 = plt.subplot(gs[2, 0])  # 第一个变量空间分布
                ax2 = plt.subplot(gs[2, 1])  # 第一个变量空间分布
                mask = (ai_type_data == 5)  # AI 0.5-0.65
            elif type == 'Humid':
                ax1 = plt.subplot(gs[3, 0])  # 第一个变量空间分布
                ax2 = plt.subplot(gs[3, 1])  # 第一个变量空间分布
                mask = (ai_type_data == 6)  # AI > 0.65

        elif grade_by == 'Cor mean':

            if type == 'All':
                mask = np.isfinite(veg_type_data)
            elif type == 'Cor(-0.1~0)':
                ax1 = plt.subplot(gs[0, 0])  # 第一个变量空间分布
                ax2 = plt.subplot(gs[0, 1])  # 第一个变量空间分布
                mask = (-0.1 <= cor_mean_data) & (cor_mean_data < 0)
            elif type == 'Cor(-0.2~-0.1)':
                ax1 = plt.subplot(gs[1, 0])  # 第一个变量空间分布
                ax2 = plt.subplot(gs[1, 1])  # 第一个变量空间分布
                mask = (-0.2 <= cor_mean_data) & (cor_mean_data < -0.1)
            elif type == 'Cor(-0.3~-0.2)':
                ax1 = plt.subplot(gs[2, 0])  # 第一个变量空间分布
                ax2 = plt.subplot(gs[2, 1])  # 第一个变量空间分布
                mask = (-0.3 <= cor_mean_data) & (cor_mean_data < -0.2)
            elif type == 'Cor(-0.4~-0.3)':
                ax1 = plt.subplot(gs[3, 0])  # 第一个变量空间分布
                ax2 = plt.subplot(gs[3, 1])  # 第一个变量空间分布
                mask = (-0.4 <= cor_mean_data) & (cor_mean_data < -0.3)
            elif type == 'Cor(<-0.4)':
                ax1 = plt.subplot(gs[4, 0])  # 第一个变量空间分布
                ax2 = plt.subplot(gs[4, 1])  # 第一个变量空间分布
                mask = (cor_mean_data < -0.4) #& (-0.5 <= cor_mean_data)

        # 应用掩码
        plot_varname_data = np.where(mask[:, :, np.newaxis], biv_map_data, np.nan)

        plot_data_x = np.where(mask, data_x, np.nan)
        plot_data_y = np.where(mask, data_y, np.nan)


        ### 二维空间分布
        bivariate_map(ax1, plot_varname_data, lower_bound_x, upper_bound_x, size_x,
                      lower_bound_y, upper_bound_y, size_y,
                      col_matrix, xlabel, ylabel)

        ### 二维密度图
        plot_combined_hexbin_with_WeightedPointTrendline(plot_data_x, plot_data_y, xlabel, ylabel, grade_by, type,
                                                         'No', ax=ax2)

    # plt.savefig(fr'D:\CAU\phenology_swc_vpd\Global_test4\Fig\4Pheno_trend_and_Cor\Bivariate_map\SM_VPD_Cor17\{grade_by}\SM_VPD_Cor(preseason)_BivariateMap_{ylabel}_{grade_by}.png', dpi = 600, bbox_inches='tight')
    plt.savefig(fr'D:\CAU\phenology_swc_vpd\Global_test5\Fig\Fig 5 Relationship between POS and Cor\{grade_by}\SM_VPD_Cor(preseason)_BivariateMap_{ylabel}_{grade_by}_OutnosigCor({OutnosigCor}).png',
                dpi = 300, bbox_inches='tight')
    print('S Fig plot done!')

    # plt.show()


# plot_S16_21(biv_map_cormean_posmean,
#            mean_lower_bound_x, mean_upper_bound_x, mean_lower_bound_y, mean_upper_bound_y, mean_bin_size_x, mean_bin_size_y,
#            col_matrix,
#            cor_mean, pos_mean,
#            'SM-VPD coupling',
#            'POS mean',
#            'Veg')
# print('S16 plot done!')
# plot_S16_21(biv_map_corslope_posslope,
#            slope_lower_bound_x, slope_upper_bound_x, slope_lower_bound_y, slope_upper_bound_y, slope_bin_size_x, slope_bin_size_y,
#            col_matrix,
#            cor_slope, pos_slope,
#            'SM-VPD coupling trend',
#            'POS trend',
#            'Veg')
# print('S19 plot done!')
#
#
# plot_S16_21(biv_map_cormean_posmean,
#                            mean_lower_bound_x, mean_upper_bound_x, mean_lower_bound_y, mean_upper_bound_y, mean_bin_size_x, mean_bin_size_y,
#                            col_matrix,
#                            cor_mean, pos_mean,
#                            'SM-VPD coupling',
#                            'POS mean',
#                            'AI')
# print('S17 plot done!')
# plot_S16_21(biv_map_corslope_posslope,
#                            slope_lower_bound_x, slope_upper_bound_x, slope_lower_bound_y, slope_upper_bound_y, slope_bin_size_x, slope_bin_size_y,
#                            col_matrix,
#                            cor_slope, pos_slope,
#                            'SM-VPD coupling trend',
#                            'POS trend',
#                            'AI')
# print('S20 plot done!')


# plot_S16_21(biv_map_cormean_posmean,
#                            mean_lower_bound_x, mean_upper_bound_x, mean_lower_bound_y, mean_upper_bound_y, mean_bin_size_x, mean_bin_size_y,
#                            col_matrix,
#                            cor_mean, pos_mean,
#                            'SM-VPD coupling',
#                            'POS mean',
#                            'Cor mean')
# print('S18 plot done!')
# plot_S16_21(biv_map_corslope_posslope,
#                            slope_lower_bound_x, slope_upper_bound_x, slope_lower_bound_y, slope_upper_bound_y, slope_bin_size_x, slope_bin_size_y,
#                            col_matrix,
#                            cor_slope, pos_slope,
#                            'SM-VPD coupling trend',
#                            'POS trend',
#                            'Cor mean')
# print('S21 plot done!')
#
#

