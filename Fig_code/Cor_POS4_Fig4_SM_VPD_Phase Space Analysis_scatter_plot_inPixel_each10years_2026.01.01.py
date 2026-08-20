import os.path
import glob
import sys

import numpy as np

from osgeo import gdal
import pandas as pd
import pymannkendall as mk
import matplotlib.pyplot as plt
import matplotlib as mpl

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
from scipy.ndimage import gaussian_filter1d

from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1 import make_axes_locatable


################################ 1 输入及输出 #############################
##### 输入设定 #####

star_year = 2001
end_year = 2024
years = np.arange(star_year, end_year + 1)
years_length = end_year - star_year + 1
print('years_length:', years_length)

Phase_Space_Analysis_year = 12

scale = 55

same_input_path = rf'D:\CAU\phenology_swc_vpd\paper\Data_and_code_online\FigShare_data'

sm_input = rf'{same_input_path}\{scale}km\Climate_data\SM_preseason_mean2'
vpd_input = rf'{same_input_path}\{scale}km\Climate_data\VPD_preseason_mean2'

sm_tiffiles = sorted(glob.glob(os.path.join(sm_input, '*.tif')))
vpd_tiffiles = sorted(glob.glob(os.path.join(vpd_input, '*.tif')))

#### 输入的植被类型 tif path
veg_type_tif = rf'{input_same_path}\{scale}km\Veg_type\NH_veg_type_{scale}km(Python).tif'

#### 输入AI tif path
ai_type_tif = rf'{input_same_path}\{scale}km\AI\NH30_84_AI(graident)_{scale}km.tif'

#### 输入的耦合梯度 tif path
cor_mean_file = fr'{input_same_path}\{scale}km\mean\SM_VPD_Cor17_8_0\Cor_mean_{scale}km_All.tif'  #SOS - POS

##### 输出设定 #####
output_fig_path = 'D:\CAU\phenology_swc_vpd\paper\Data_and_code_online\Result'

############################### 基本信息 ###############################
sample = cor_tiffiles[0]
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

sm_stack = []
vpd_stack = []

cor_stack = []

for tif_file in sm_tiffiles:
    get_band(tif_file, sm_stack)
for tif_file in vpd_tiffiles:
    get_band(tif_file, vpd_stack)

for tif_file in cor_tiffiles:
    get_band(tif_file, cor_stack)


sm_stack = np.stack(sm_stack, axis=0)
vpd_stack = np.stack(vpd_stack, axis=0)

cor_stack = np.stack(cor_stack, axis=0)


#### 植被类型数据
veg_type_tif = gdal.Open(veg_type_tif)
veg_type_data = veg_type_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
print('veg_type_data shape:', veg_type_data.shape)

#### 干旱类型数据
ai_tif = gdal.Open(ai_tif)
ai_data = ai_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
print('ai_data shape:', ai_data.shape)

#### 耦合梯度数据
cor_mean_file = gdal.Open(cor_mean_file)
cor_mean_data = cor_mean_file.GetRasterBand(1).ReadAsArray().astype(np.float32)
print('cor_mean_data shape:', cor_mean_data.shape)


print(f'SM有效像元：{np.count_nonzero(np.isfinite(sm_stack[0, :, :]))}')
print(f'VPD有效像元：{np.count_nonzero(np.isfinite(vpd_stack[0, :, :]))}')
print(f'Cor有效像元：{np.count_nonzero(np.isfinite(cor_stack[0, :, :]))}')
print('Stack done!')

############################### 3 去异常值 ###################################
time_lengths = sm_stack.shape[0]

############# 检测pos_stack与cor_stack的配对结果
mask = np.isfinite(sm_stack) & np.isfinite(vpd_stack)  & np.isfinite(cor_stack)

# 对每个像素点的时间序列计算有效数据点数量
valid_pixel_count = np.nansum(mask, axis=0)

# 创建有效像素的掩码
valid_pixel_mask = valid_pixel_count > (years_length/2)  # (years_length - 3)
print(f'去异常值且配对后有效年份>12年的像元数量:{np.count_nonzero(valid_pixel_mask)}')


sm_stack_clean = np.where(valid_pixel_mask, sm_stack, np.nan)
vpd_stack_clean = np.where(valid_pixel_mask, vpd_stack, np.nan)

cor_stack_clean = np.where(valid_pixel_mask, cor_stack, np.nan)

print('IQR done!')

########################## 4 标准化 ##################################
def standardize_data(data_stack):
    """时间标准化"""

    def standardize_func(data_pixel, i, j):
        original_nan_mask = np.isnan(data_pixel)

        spatio_mean = np.nanmean(data_pixel)
        spatio_std = np.nanstd(data_pixel)

        standardized_pixel_data = (data_pixel - spatio_mean) / spatio_std

        standardized_pixel_data[original_nan_mask] = np.nan

        return standardized_pixel_data, i, j

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


sm_standardized = standardize_data(sm_stack_clean)
vpd_standardized = standardize_data(vpd_stack_clean)
cor_standardized = standardize_data(cor_stack_clean)



def plot_phase_space_by_veg_AI(x_data, y_data, year_length, gradient_by):

    ##rcParams要在subplots前创建，否则创建的plot的字体不会发生改变
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

    if gradient_by == 'All':
        types = ['All']
        codes = [1]

        fig, axes = plt.subplots(1, 1, figsize=(3.5, 3))

    elif gradient_by == 'Veg':
        types = ['Forest', 'Shrub', 'Savanna', 'Grass']
        codes = [1, 2, 3, 4]

        fig, axes = plt.subplots(
            3, 2,
            figsize=(6.5, 6.1),
            gridspec_kw={'height_ratios': [1, 0.005, 1]}
        )
        fig.subplots_adjust(
            hspace=0.01,
            wspace=0.3
        )
        # 关闭中间两个空白子图
        axes[1, 0].axis('off')
        axes[1, 1].axis('off')

        axes = axes.flatten()

    elif gradient_by == 'AI':
        types = ['Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']
        codes = [2, 3, 5, 6]

        fig, axes = plt.subplots(
            3, 2,
            figsize=(6.5, 6.1),
            gridspec_kw={'height_ratios': [1, 0.005, 1]}
        )
        fig.subplots_adjust(
            hspace=0.01,
            wspace=0.3
        )

        # 关闭中间两个空白子图
        axes[1, 0].axis('off')
        axes[1, 1].axis('off')

        axes = axes.flatten()


    elif gradient_by == 'Cor mean':

        types = ['Cor(-0.1~0)', 'Cor(-0.2~-0.1)', 'Cor(-0.3~-0.2)',
                 'Cor(-0.4~-0.3)', 'Cor(-0.5~-0.4)']#, 'Cor(<-0.5)']
        codes = [0, 1, 2, 3, 4]

        fig, axes = plt.subplots(
            5, 2,
            figsize=(6.5, 9),
            gridspec_kw={'height_ratios': [1, 0.005, 1, 0.005, 1]}
        )
        fig.subplots_adjust(
            hspace=0.01,
            wspace=0.3
        )

        # 关闭中间两个空白子图
        axes[1, 0].axis('off')
        axes[1, 1].axis('off')

        axes[3, 0].axis('off')
        axes[3, 1].axis('off')

        axes[4, 1].axis('off')

        axes = axes.flatten()


    elif gradient_by == 'POS_slope_sig':
        types = ['Significant']
        codes = [1]

        fig, axes = plt.subplots(1, 1, figsize=(5, 4))

    if year_length == 5:
        period_labels = [
            "2001–2005",
            "2006–2010",
            "2011–2015",
            "2016–2020",
            "2021–2024"
        ]

        balance_hex = [
            "#543729",
            "#2baf2b",
            "#00acee",
            "#ef5734",
            "#ffcc2f"
        ]
    elif year_length == 12:
        period_labels = [
            "2001–2012",
            "2013–2024"
        ]

        balance_hex = [
            "#2166AC",
            "#B2182B"
        ]

    for i, (type, code) in enumerate(zip(types, codes)):



        if gradient_by == 'All' or gradient_by == 'POS_slope_sig':
            ax = axes
        else:
            if i < 2:
                ax = axes[i]
            elif i >= 2 and i <4:
                ax = axes[i+2]
            elif i >= 4:
                ax = axes[i + 4]



        legend_handles = []

        centroid_x = []
        centroid_y = []

        # === 创建边缘轴 ===
        ### 构建边缘分布
        divider = make_axes_locatable(ax)
        ax_histx = divider.append_axes("top", size="25%", pad=0.15, sharex=ax)
        ax_histy = divider.append_axes("right", size="25%", pad=0.15, sharey=ax)

        ax_histx.axis("off")
        ax_histy.axis("off")

        ax.set_xlim(-1, 1)
        ax.set_xticks(np.arange(-1, 1.001, 0.25))
        ax.set_xticklabels(f'{int(x)}' if x == int(x) else f'{x}' for x in np.arange(-1, 1.001, 0.25))
        ax.set_ylim(-1, 1)
        ax.set_yticks(np.arange(-1, 1.001, 0.25))
        ax.set_yticklabels(f'{int(x)}' if x == int(x) else f'{x}' for x in np.arange(-1, 1.001, 0.25))

        # print(np.nanmean(x_data[0]), np.nanmean(x_data[1]))
        # print(np.nanmean(y_data[0]), np.nanmean(y_data[1]))

        for t, (label, c) in enumerate(zip(period_labels, balance_hex)):

            if code is None:
                mask = np.isfinite(x_data[t]) & np.isfinite(y_data[t])
            else:
                # === 植被掩膜 ===
                if gradient_by == 'Veg':
                    mask = (
                            (veg_type_data == code) &
                            np.isfinite(x_data[t]) &
                            np.isfinite(y_data[t])
                    )
                # === 干旱区掩膜 ===
                elif gradient_by == 'AI':
                    if code == 3:
                        mask = (
                                ((ai_data == code) | (ai_data == 4)) &
                                np.isfinite(x_data[t]) &
                                np.isfinite(y_data[t])
                        )
                    else:
                        mask = (
                                (ai_data == code) &
                                np.isfinite(x_data[t]) &
                                np.isfinite(y_data[t])
                        )

                # === 耦合梯度掩膜 ===
                elif gradient_by == 'Cor mean':
                    if code == 4:
                        mask = (
                                (cor_mean_data <= -0.4) &
                                np.isfinite(x_data[t]) &
                                np.isfinite(y_data[t])
                        )
                    else:
                        mask = (
                                (cor_mean_data <= -(code*0.1)) & (cor_mean_data > -((code+1)*0.1)) &
                                np.isfinite(x_data[t]) &
                                np.isfinite(y_data[t])
                        )
                    print(f'code={code}, 有效像元数量: {np.sum(mask)}')

                elif gradient_by == 'POS_slope_sig':
                    mask = (
                            (pos_slope_pvalue < 0.05) & np.isfinite(pos_slope_pvalue) &
                            np.isfinite(x_data[t]) &
                            np.isfinite(y_data[t])
                    )
                elif gradient_by == 'All':
                    mask = (
                            np.isfinite(x_data[t]) &
                            np.isfinite(y_data[t])
                    )

            x = x_data[t][mask]
            y = y_data[t][mask]

            # === 所有像元散点 ===
            ax.plot([-1.5, 1.5], [1.5, -1.5], '--', color='black', linewidth=1)

            ax.scatter(
                x, y,
                s=5,
                color=c,
                alpha=0.05,
                linewidths=0,
                zorder=1
            )

            # === 质心 ===
            cx = np.nanmean(x)
            cy = np.nanmean(y)

            centroid_x.append(cx)
            centroid_y.append(cy)

            h = ax.scatter(
                cx, cy,
                s=60,
                color=c,
                edgecolor='k',
                alpha=1,
                zorder=5,
                label=label
            )

            legend_handles.append(h)

            # ===== 2D KDE 等密度线（圈住散点）=====
            # if len(x) > 50:  # 保证稳定
            values = np.vstack([x, y])
            # print(f'code={code}')
            kde2d = gaussian_kde(values, bw_method=0.1)

            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()

            xx, yy = np.mgrid[
                     xmin:xmax:100j,
                     ymin:ymax:100j
                     ]
            grid = np.vstack([xx.ravel(), yy.ravel()])
            zz = kde2d(grid).reshape(xx.shape)

            ax.contour(
                xx, yy, zz,
                levels=[np.percentile(zz, 95)],  # 等值线用于表示联合分布中的高密度核心区域，其阈值定义为核密度估计值的第95百分位。
                colors=[c],
                linewidths=2,
                # alpha=1,
                zorder=3
            )

            # # ===== KDE（边缘密度）=====
            # ## x
            # x_kde = gaussian_kde(x, bw_method=0.1)
            # x_grid = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 200)
            # ax_histx.plot(x_grid, x_kde(x_grid), color=c, lw=1)
            # ax_histx.fill_between(x_grid, x_kde(x_grid), color=c, alpha=0.3)
            #
            # # 添加均值线
            # y1 = x_kde(cx)[0]
            # ax_histx.plot([cx, cx],
            #                  [0, y1],
            #                  color=c, linestyle='--', linewidth=1.5)
            #
            # ## y
            # y_kde = gaussian_kde(y, bw_method=0.1)
            # y_grid = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 200)
            # ax_histy.plot(y_kde(y_grid), y_grid, color=c, lw=1)
            # ax_histy.fill_betweenx(y_grid, y_kde(y_grid), color=c, alpha=0.3)
            #
            # # 添加均值线
            # x1 = y_kde(cy)[0]
            # ax_histy.plot([0, x1],
            #               [cy, cy],
            #               color=c, linestyle='--', linewidth=1.5)


            # ========= 频率分布(边缘分布) ===============
            ## x
            counts, bin_edges, _ = ax_histx.hist(
                x,
                bins=100,
                range=(-1, 1),
                density=True,
                histtype='step',
                linewidth=0
            )

            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            smooth_counts = gaussian_filter1d(counts, sigma=2)

            # 正确方向
            ax_histx.plot(bin_centers, smooth_counts, color=c, lw=1.5)
            ax_histx.fill_between(bin_centers, smooth_counts, color=c, alpha=0.3)

            # 均值线
            idx = np.argmin(np.abs(bin_centers - cx))
            y1 = smooth_counts[idx]

            ax_histx.plot([cx, cx], [0, y1], '--', color=c, linewidth=1.5)

            # 贴紧主图
            ax_histx.set_ylim(0, max(smooth_counts) * 1.05)
            ax_histx.margins(y=0)


            ## y
            # histogram（仅用于统计）
            counts, bin_edges, _ = ax_histy.hist(
                y,
                bins=100,
                range=(-1, 1),
                density=True,
                histtype='step',
                linewidth=0
            )

            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # 平滑
            smooth_counts = gaussian_filter1d(counts, sigma=2)

            # 绘制
            ax_histy.plot(smooth_counts, bin_centers, color=c, lw=1.5)
            ax_histy.fill_betweenx(bin_centers, smooth_counts, color=c, alpha=0.3)

            # 均值线
            idx = np.argmin(np.abs(bin_centers - cy))
            x1 = smooth_counts[idx]

            ax_histy.plot([0, x1], [cy, cy], '--', color=c, linewidth=1.5)

            ax_histy.set_xlim(0, max(smooth_counts) * 1.05)
            ax_histy.margins(x=0)


        # === 均值箭头 ===
        for k in range(len(centroid_x) - 1):
            ax.annotate(
                '',
                xy=(centroid_x[k + 1], centroid_y[k + 1]),
                xytext=(centroid_x[k], centroid_y[k]),
                arrowprops=dict(
                    arrowstyle='->',
                    mutation_scale=20,  # 1=默认大小，>1变大，<1变小
                    color='k',
                    lw=2,
                    alpha=1
                ),
                zorder=4
            )


            # ### 在边缘密度图上连接均值
            # # 获取两个均值点
            # x1, x2 = np.nanmean(x_data[k]), np.nanmean(x_data[k+1])
            # y1, y2 = np.nanmean(y_data[k]), np.nanmean(y_data[k + 1])
            # y1_val = x_kde(x1)[0] if len(x) > 0 else 0  # 需要小心：这里x_kde是最后一个时期的
            # y2_val = x_kde(x2)[0] if len(x) > 0 else 0
            #
            # # 在上边缘图添加箭头
            # ax_histx.annotate('', xy=(x2, 0.5), xytext=(x1, 0.5),)
            #                   # arrowprops=dict(arrowstyle='->', color='gray', lw=1, alpha=0.5))

        # 参考线
        ax.axhline(0, ls='--', lw=0.8, color='gray', zorder=2)
        ax.axvline(0, ls='--', lw=0.8, color='gray', zorder=2)

        if gradient_by == 'Veg' or gradient_by == 'AI' :
            if type == 'Forest' or type == 'Arid':
                word = 'a'
            elif type == 'Shrub' or type == 'Semi-arid':
                word = 'b'
            elif type == 'Savanna' or type == 'Dry sub-humid':
                word = 'c'
            elif type == 'Grass' or type == 'Humid':
                word = 'd'
            # ax_histx.set_title(f'({word}) {type}', fontsize=10, pad=6)

        ax.set_xlabel('SM (standardized)')
        ax.set_ylabel('VPD (standardized)')

        ax.legend(
            handles=legend_handles,
            loc='upper right',
            frameon=False,  ##图例外框和背景色
            # fontsize=7,
            handletextpad=0.1,  # 缩短图例线与文字之间的距离
        )


    plt.tight_layout()

    output_path =rf'{output_fig_path}\SM_VPD{climate_test_number}_Phase_Space_Analysis_{gradient_by}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    # plt.show()
    # plt.close()



########### 12年1平均
sm_2001_2012 = np.nanmean(sm_standardized[:12, :, :], axis=0)
sm_2013_2024 = np.nanmean(sm_standardized[12:, :, :], axis=0)

vpd_2001_2012 = np.nanmean(vpd_standardized[:12, :, :], axis=0)
vpd_2013_2024 = np.nanmean(vpd_standardized[12:, :, :], axis=0)

sm_each_n_year = np.stack([
    sm_2001_2012,
    sm_2013_2024
], axis=0)

vpd_each_n_year = np.stack([
    vpd_2001_2012,
    vpd_2013_2024
], axis=0)
print('Standardized done!')


### Fig4
plot_phase_space_by_veg_AI(sm_each_n_year, vpd_each_n_year, Phase_Space_Analysis_year, 'All')
print('Fig4 plot done!')
# ### Fig S13-15
# plot_phase_space_by_veg_AI(sm_each_n_year, vpd_each_n_year, Phase_Space_Analysis_year, 'Veg')
# print('S13 plot done!')
# plot_phase_space_by_veg_AI(sm_each_n_year, vpd_each_n_year, Phase_Space_Analysis_year, 'AI')
# print('S14 plot done!')
# plot_phase_space_by_veg_AI(sm_each_n_year, vpd_each_n_year, Phase_Space_Analysis_year, 'Cor mean')
# print('S15 plot done!')

