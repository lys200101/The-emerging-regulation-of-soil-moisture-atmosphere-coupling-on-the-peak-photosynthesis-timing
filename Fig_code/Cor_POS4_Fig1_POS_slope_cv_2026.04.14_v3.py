import glob
import os.path
import sys
from tabnanny import verbose

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.pyplot import figure
from osgeo import gdal
from scipy.stats import alpha
from scipy.stats import theilslopes
from brokenaxes import brokenaxes
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import FixedLocator
import matplotlib.patches as mpatches

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import pymannkendall as mk

######### 共用function #############
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


####################################### 1 输入输出 ######################################
##################### 1.1 输入 ##########################
startYear = 2001
endYear = 2024
year_length = endYear-startYear+1

data_type = 'MCD12Q2' # MCD12Q2
# interval_days = 16  # 1/8/16

pheno = 'POS'
scale = 55

input_same_path = rf'D:\CAU\phenology_swc_vpd\paper\Data_and_code_online\FigShare_data'

pheno_path = rf'{input_same_path}\{scale}km\{pheno}_{scale}km'

veg_type_tif = rf'{input_same_path}\{scale}km\Veg_type\NH_veg_type_{scale}km(Python).tif'

ai_type_tif = rf'{input_same_path}\{scale}km\AI\NH30_84_AI(graident)_{scale}km.tif'

pheno_files = sorted(glob.glob(os.path.join(pheno_path, '*.tif')))


##################### 1.2 输出 ##########################
output_same_path = rf'D:\CAU\phenology_swc_vpd\paper\Data_and_code_online\Result'
fig_output = rf'{output_same_path}\Fig 1 POS trend and CV'

####################################### 2 基础信息 #########################################
sample_tif = pheno_files[0]
sample_data = gdal.Open(sample_tif)

sample_array = sample_data.GetRasterBand(1).ReadAsArray().astype(np.float32)

crs = sample_data.GetProjectionRef()
gt = sample_data.GetGeoTransform()

rows = sample_array.shape[0]
cols = sample_array.shape[1]

row_indices = np.repeat(np.arange(rows), cols)
col_indices = np.tile(np.arange(cols), rows)

lon_min = gt[0]               ##起始经度
lon_max = gt[0] + gt[1]*cols  ##起始经度+像元分辨率*列数
lat_max = gt[3]       ##起始纬度
lat_min = gt[3] + gt[5]*rows  ##起始纬度+像元分辨率*行数数
print(f'经度范围：{lon_min}~{lon_max}°')
print(f'纬度范围：{lat_min}~{lat_max}°N')



####################################### 3 堆叠 ######################################
### 读取波段
def get_band(tif_file, stack):
    tif = gdal.Open(tif_file)
    climate_data = tif.GetRasterBand(1).ReadAsArray().astype(np.float32)

    if scale == 11:
        climate_data = np.where(climate_data != 0, climate_data, np.nan)

    stack.append(climate_data)
    tif = None  # 及时释放资源

############ 3.1 物候 ################
pheno_stack = []

for tif_file in pheno_files:
    get_band(tif_file, pheno_stack)

pheno_stack = np.stack(pheno_stack, axis=0)

print('pheno_stack[100:105, 400:405]', pheno_stack[:, 100, 500])

############# 3.2 植被类型 ################
veg_type_data = gdal.Open(veg_type_tif)
veg_type_data = veg_type_data.GetRasterBand(1).ReadAsArray().astype(np.float32)

############### 3.3 AI指数 #################
ai_type_data = gdal.Open(ai_type_tif)
ai_type_data = ai_type_data.GetRasterBand(1).ReadAsArray().astype(np.float32)


####################################### 4 去异常 #########################################
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

        if (len(np.isfinite(x_masked)) > (year_length/2)) & (len(np.isfinite(x_masked)) <= year_length):
            return x_masked, i, j  # IQR, lower_range, upper_range,  minv, maxv
        else:
            print(f'IQR threshold：{lower_range:.4f} ~ {upper_range:.4f}\n'
                  f'像元的原始数据:{x}\n'
                  f'去异常值后无效像元超3个：{x_masked}')
            return np.full_like(x, np.nan), i, j


time_lengths = pheno_stack.shape[0]

outlier_pheno_stack = np.full((time_lengths, rows, cols), np.nan)

### IQR去异常值
results = Parallel(n_jobs=18, verbose=10)(
    delayed(Outlier_array_IQR)(
        pheno_stack[:, i, j], i, j, 25, 75
    ) for i, j in zip(row_indices, col_indices)
)

for data_mask, i, j in results:
    outlier_pheno_stack[:, i, j] = data_mask

print('outlier_pheno_stack[:, 101, 501]:', outlier_pheno_stack[:, 101, 501])


################ 5 ######################
################ 5.1 计算长期趋势+显著性 ##################

def calculate_senSlope(data, i, j):

    mask = np.isfinite(data)

    data_clean = data[mask]
    years_valid = years[mask]

    if (len(data_clean) > (year_length/2)) and (len(data_clean) <= year_length):   ### >10 / >17/ >19

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

    return (i, j, slope, pvalue, intercept)

pheno_slope = np.full((rows, cols), np.nan)
pheno_intercept = np.full((rows, cols), np.nan)
pheno_slope_pvalue = np.full((rows, cols), np.nan)

years = np.arange(startYear, startYear + pheno_stack.shape[0])

results = Parallel(n_jobs = 18, verbose=10)(
    delayed(calculate_senSlope)(
        outlier_pheno_stack[:, i, j], i, j)
            for i, j in zip(row_indices, col_indices)
)

for i, j, slope, pvalue, intercept in results:
    pheno_slope[i, j] = slope
    pheno_intercept[i, j] = intercept
    pheno_slope_pvalue[i, j] = pvalue



################ 5.2 计算多年的std\cv ##################
def calculate_std_cv(data, i, j):

    data_clean = data[~np.isnan(data)]

    if (len(data_clean) > (year_length/2)) and (len(data_clean) <= year_length):

        mean_value = np.nanmean(data_clean)
        std_value = np.nanstd(data_clean)
        cv_value = std_value/mean_value

    else:
        ############# 如果 len(data_nodrought_clean) <= 10 可以考虑用滑动窗口来计算 ###########
        std_value = np.nan
        cv_value = np.nan

    return (i, j, std_value, cv_value)

pheno_std = np.full((rows, cols), np.nan)
pheno_cv = np.full((rows, cols), np.nan)

results = Parallel(n_jobs = 18, verbose=10)(
    delayed(calculate_std_cv)(
        pheno_stack[:, i, j], i, j)
            for i, j in zip(row_indices, col_indices)
)

for i, j, std_value, cv_value in results:
    pheno_std[i, j] = std_value
    pheno_cv[i, j] = cv_value


################ 6 plot ##################
## a left
def plot_pheno_slope_and_pvalue_forAllvegType(plot_data_slope, plot_data_pvalue, plot_data_cv, colorbarmin, colorbarmax, data_type, name, ax):
    # # 创建5个子图：4个地图 + 1个colorbar
    # fig = plt.figure(figsize=(6, 4))
    # gs = gridspec.GridSpec(1, 3,
    #                        width_ratios=[4, 1, 1],  #三列的宽度比
    #                        hspace=0.5, wspace=0.2)

    fig = ax.get_figure()
    if name =='All':
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
    # 创建内部真正的三个子轴
    ax1 = fig.add_subplot(gs_inner[0, 0])  # 地图
    ax2 = fig.add_subplot(gs_inner[0, 1])  # 纬度曲线
    ax3 = fig.add_subplot(gs_inner[1, :])  # Colorbar横跨两列



    ########### 子图1：空间分布 #################
    ax1.set_box_aspect(1)  #强制地图轴的形状为正方，使其直径撑满格子高度
    ax1.axis('off')
    ### 创建地图
    m = Basemap(ax=ax1,
                projection='npstere',   # 北极投影
                boundinglat=30,         # 最低显示纬度
                lon_0=0,                # 中心经度，180:太平洋居中；90：亚洲居中
                resolution='l')

    # 设置经纬度刻度
    m.drawparallels(np.arange(30, 90, 30),
                    labels=[0, 0, 0, 0],
                    linewidth=0.5,
                    # fontsize=8,
                    color="black")

    m.drawmeridians(np.arange(0, 360, 60),
                    latmax=90,  #使经度线在北极交汇
                    labels=[0, 0, 0, 1],  #labels=[left, right, top, bottom] 控制经度显示与否
                    linewidth=0.5,
                    # fontsize=8,
                    color="gray")

    # # 填充大陆
    # m.fillcontinents(color='white', lake_color='white', zorder=1)

    # map_boundary = m.drawmapboundary(linewidth=0)  # 不显示边界线


    ### 绘制数据
    # 颜色映射
    if data_type == 'pheno slope':
        color_list = ['#762a83', '#9970ab', '#c2a5cf', '#e7d4e8',
                      '#d9f0d3', '#a6dba0', '#5aae61', '#1b7837']
        cmap = mpl.colors.ListedColormap(color_list)
    elif data_type == 'pheno cv':
        orange_cmap = mpl.colormaps['Oranges']
        colors = orange_cmap(np.linspace(0.2, 0.9, 8))
        cmap = mpl.colors.ListedColormap(colors)

    bins = np.linspace(colorbarmin, colorbarmax, 9)
    norm = mpl.colors.BoundaryNorm(bins, cmap.N)

    # 绘制
    # plot_data_slope = np.where(lats >= 30, plot_data_slope, np.nan)
    # plot_data_slope = np.hstack([plot_data_slope, plot_data_slope[:, 0:1]])
    # # plot_data_pvalue = np.hstack([plot_data_pvalue, plot_data_pvalue[:, 0:1]])
    # from mpl_toolkits.basemap import addcyclic
    # plot_data_slope, lons1 = addcyclic(plot_data_slope, lons[0, :])
    # plot_data_pvalue, _ = addcyclic(plot_data_pvalue, lons[0, :])
    #

    # lons, lats = np.meshgrid(lons1, lats[:, 0])

    # 生成网格坐标
    lons = np.linspace(-180, 180, cols, endpoint=False)
    lats = np.linspace(lat_max, lat_min, rows)
    lons, lats = np.meshgrid(lons, lats)

    if data_type == 'pheno slope':
        plot = m.pcolormesh(lons, lats, plot_data_slope, cmap=cmap, norm=norm, shading='nearest', latlon=True,
                            zorder=1)  # 避免极区撕裂
    if data_type == 'pheno cv':
        plot = m.pcolormesh(lons, lats, plot_data_cv, cmap=cmap, norm=norm, shading='nearest', latlon=True,
                            zorder=1)  # 避免极区撕裂

    plots.append(plot)  # 保存plot对象

    if data_type == 'pheno slope':
        # 添加显著性标记
        significant_mask = (plot_data_pvalue < 0.05) & np.isfinite(plot_data_pvalue) & np.isfinite(plot_data_slope)

        if np.any(significant_mask):
            sig_rows, sig_cols = np.where(significant_mask)
            sig_lons = lons[sig_rows, sig_cols]
            sig_lats = lats[sig_rows, sig_cols]

            # 将地理坐标转为投影坐标
            sig_x, sig_y = m(sig_lons, sig_lats)
            if scale == 55:
                s_size= 0.5
            elif scale == 11:
                s_size = 0.1
            ax1.scatter(sig_x, sig_y, marker='.', color='black', s=s_size,
                        linewidth=0.1, zorder=2)
    ax1.set_frame_on(False)

    ### 绘制大陆边界
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
        edgecolor='black',
        linewidth=0.8,
        clip_on=False,  # 这个对象是否要被当前 Axes 的矩形边界裁剪（clip）
        zorder=4  # 放最上层
    )

    ax1.add_patch(boundary_circle)

    # m.drawmapboundary(
    #     fill_color='none',
    #     color='red',
    #     linewidth=5
    # )

    # ax1.axis('off')


    # if name == 'All':
    #     if data_type == 'pheno slope':
    #         ax1.set_title(f'POS trend (days/yr)', pad=10, fontweight='bold')
    #     elif data_type == 'pheno cv':
    #         ax1.set_title(f'POS CV', pad=10, fontweight='bold')
    # else:
    #     # if name == 'Forest' or name == 'Arid':
    #     #     word = 'a'
    #     # elif name == 'Shrub' or name == 'Semi-arid':
    #     #     word = 'b'
    #     # elif name == 'Savanna' or name == 'Dry sub-humid':
    #     #     word = 'c'
    #     # elif name == 'Grass' or name == 'Humid':
    #     #     word = 'd'
    #
    #     ax1.set_title(f'{name}', pad=10, fontweight='bold')


    ## 统计 ##
    if data_type == 'pheno cv':
        h = 0.33
        v = 0.9
        ax1.text(h, v,
                 f'Mean = {np.nanmean(plot_data_cv):.2f}',
                 transform=ax1.transAxes,  # 使用相对坐标，方便定位
                 multialignment='center',  # 垂直居中
                 fontsize = 6)
    elif data_type == 'pheno slope':
        data_gte0 = plot_data_slope[(plot_data_slope >= 0) & np.isfinite(plot_data_slope)]
        data_lt0 = plot_data_slope[(plot_data_slope < 0) & np.isfinite(plot_data_slope)]
        sum_count = np.sum(np.isfinite(plot_data_slope))

        data_gte0_count = np.sum(np.isfinite(data_gte0))
        data_lt0_count = np.sum(np.isfinite(data_lt0))

        data_gte0_ratio = data_gte0_count / sum_count * 100
        data_lt0_ratio = data_lt0_count / sum_count * 100

        h = 0.24
        v = 0.82
        ax1.text(h, v,
                 f'Mean = {np.nanmean(plot_data_slope):.2f}\n'
                 f'Advance = {np.nanmean(data_lt0):.2f} ({data_lt0_ratio:.1f}%)\n'
                 f'Delay = {np.nanmean(data_gte0):.2f} ({data_gte0_ratio:.1f}%)',
                 transform=ax1.transAxes,  # 使用相对坐标，方便定位
                 multialignment='center',  # 垂直居中
                 fontsize = 6)


    ########### 子图2：逐纬度变化趋势

    # 使用实际纬度值作为y轴
    lat_centers = lats[:, 0]

    if data_type == 'pheno slope':
        plot_data_lat = np.nanmean(plot_data_slope, axis=1)

        ax2.axvline(x=0, color='gray', linestyle='--', linewidth=1)

    elif data_type == 'pheno cv':
        plot_data_lat = np.nanmean(plot_data_cv, axis=1)

        ax2.axvline(x=0.1, color='gray', linestyle='--', linewidth=1)

    ax2.plot(plot_data_lat, lat_centers, color='red', linewidth=1, alpha=0.8)

    if data_type == 'pheno slope':
        ax2.set_xlim(-0.45, 0.45)
        ax2.set_xticks(np.arange(-0.3, 0.301, 0.3))
        ax2.set_xticklabels(['-3  ', '0', '3'])  # 手动设置标签

        tick_size = plt.rcParams['xtick.labelsize']

        ax2.text(
            0.45,  # x = 刻度位置（数据坐标）
            -0.023,  # y = 稍微往下（轴坐标）
            r'$×10^{-1}$',  # 你想要的内容
            transform=ax2.get_xaxis_transform(),
            ha='left',  # 向右展开（避免压缩）
            va='top',
            fontsize = 8,
            clip_on=False
        )

    elif data_type == 'pheno cv' :
        ax2.set_xlim(0, 0.15)
        ax2.set_xticks(np.arange(0, 0.15, 0.1))
        ax2.set_xticklabels(['0', '1'])  # 手动设置标签

        tick_size = plt.rcParams['xtick.labelsize']

        ax2.text(
            0.14,  # x = 刻度位置（数据坐标）
            -0.023,  # y = 稍微往下（轴坐标）
            r'$×10^{-1}$',  # 你想要的内容
            transform=ax2.get_xaxis_transform(),
            ha='left',  # 向右展开（避免压缩）
            va='top',
            fontsize=8,
            clip_on=False
        )

    ax2.set_ylim(30, 90)
    ax2.set_yticks(np.arange(30, 91, 10))
    ax2.set_yticklabels(f'{x}°' for x in np.arange(30, 91, 10))

    ax2.tick_params(axis='both', which='major', length = 2, pad=3)

    ########### 子图3：Colorbar
    ### 生成Colorbar（使用最后一个位置）
    cbar = fig.colorbar(plots[0], cax=ax3, orientation='horizontal')

    cbar.set_ticks(bins)
    if data_type == 'pheno slope':
        cbar.set_ticklabels(['0' if x == 0 else
             f'{int(x * 10)}' if x * 10 == int(x * 10) else
             f'{x * 10}'
             for x in bins
             ])

        # tick_size = plt.rcParams['xtick.labelsize']

        x_pos = 0.95
        y_pos = -1.75
        ax3.text(
            x_pos,  # x = 刻度位置（数据坐标）
            y_pos,  # y = 稍微往下（轴坐标）
            r'$×10^{-1}$',  # 你想要的内容
            transform=ax3.transAxes,
            ha='left',  # 向右展开（避免压缩）
            va='top',
            fontsize=8,
            clip_on=False
        )
    elif data_type == 'pheno cv':
        cbar.set_ticklabels([f'0' if x == 0 else f'{int(x * 100)}' for x in bins ])

        tick_size = plt.rcParams['xtick.labelsize']

        x_pos = 0.95
        y_pos = -1.75
        ax3.text(
            x_pos,  # x = 刻度位置（数据坐标）
            y_pos,  # y = 稍微往下（轴坐标）
            r'$×10^{-2}$',  # 你想要的内容
            transform=ax3.transAxes,
            ha='left',  # 向右展开（避免压缩）
            va='top',
            fontsize=8,
            clip_on=False
        )
    if data_type == 'pheno slope':
        cbar.set_label('PPT trend (days/yr)', labelpad = 12)
    if data_type == 'pheno cv':
        cbar.set_label('PPT CV', labelpad = 12)

    ax3.tick_params(axis='both', which='major', length=2, pad=3)
    # cbar.set_label('')

    plt.tight_layout()

    # 当前 ax1 左下角
    pos1 = ax1.get_position()

    pos2 = ax2.get_position()

    pos3 = ax3.get_position()

    if name == 'All':
        # 重新设置 ax1
        ax1.set_position([
            pos1.x0 -0.04,  # 左边不变
            pos2.y0,  # 和 ax2 对齐底部
            pos2.height,
            pos2.height
        ])  # [left, bottom, width, height]
    else:
        ax1.set_position([
            pos1.x0 - 0.14,  # 左边不变
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
def plot_pheno_slope_lt0_sig_and_lt0_nosig_and_gt0_forAllvegType(data_slope, data_pvalue, data_cv, data_type, colorbarmin, colorbarmax, name, ax):

    ### 数据划分
    #三个分类的划分
    #三个分类的划分
    data_slope_lt0_mask = data_slope<0
    data_slope_gte0_mask = data_slope>=0

    pvalue_sig_mask = data_pvalue<=0.05

    # data_slope_lt0_sig = np.where(data_slope_lt0_mask & pvalue_sig_mask, data_slope, np.nan)
    # data_slope_lt0_all = np.where(data_slope_lt0_mask , data_slope, np.nan)
    # data_slope_gte0_sig = np.where(data_slope_gte0_mask & pvalue_sig_mask, data_slope, np.nan)
    # data_slope_gte0_all = np.where(data_slope_gte0_mask , data_slope, np.nan)

    data_slope_sig = np.where(pvalue_sig_mask, data_slope, np.nan)

    #每个分类里POS变化趋势的划分
    # max_trend = np.nanmax(data_slope)
    # min_trend = np.nanmin(data_slope)
    # print(f'max_trend={max_trend}, min_trend={min_trend}')

    if data_type == 'pheno slope':
        bins = np.arange(colorbarmin, colorbarmax+0.25, 0.25)
        # count_lt0_sig, _ = np.histogram(data_slope_lt0_sig, bins=bins)
        # count_lt0_all, _ = np.histogram(data_slope_lt0_all, bins=bins)
        # count_gte0_sig, _ = np.histogram(data_slope_gte0_sig, bins=bins)
        # count_gte0_all, _ = np.histogram(data_slope_gte0_all, bins=bins)

        count_sig, _ = np.histogram(data_slope_sig, bins=bins)
        count_all, _ = np.histogram(data_slope, bins=bins)
        print(f'bins:{bins}')
    elif data_type == 'pheno cv':
        bins = np.arange(colorbarmin, colorbarmax+0.03, 0.03)
        count_cv, _ = np.histogram(data_cv, bins=bins)

    ### plot
    fig = ax.get_figure()
    gs_inner = ax.get_subplotspec().subgridspec(2, 1,
                                                height_ratios=[5, 0.3],
                                                hspace=0.15)

    # 隐藏父级 ax，因为它只是个占位符
    ax.axis('off')


    #设置截断y轴
    if data_type == 'pheno slope':
        if name == 'All':
            if scale == 55:
                y_broken_low = 5000
                y_broken_high = 11000
                y_broken_max = 12000
                height_ratios = [1, 5]

            elif scale == 11:
                y_broken_low = 30000
                y_broken_high = 60000
                y_broken_max = 100000
                height_ratios = [1, 2]

            bax = brokenaxes(
                ylims=((0, y_broken_low), (y_broken_high, y_broken_max)),
                hspace=0.1,
                height_ratios=height_ratios,  # 上图：下图大小为1：5
                subplot_spec=gs_inner[0],
                d=0.005
            )

            if scale == 55:
                bax.axs[0].set_yticks([y_broken_high, y_broken_max])
                bax.axs[1].set_yticks([0, 1000, 2000, 3000, 4000])
            elif scale == 11:
                bax.axs[0].set_yticks([60000, 80000, 100000])
                bax.axs[1].set_yticks([0, 10000, 20000, 30000])

            for ax in bax.axs:
                # 获取当前y轴刻度值
                y_ticks = ax.get_yticks()
                # 根据y轴刻度值生成标签
                if scale == 55:
                    zoom_factor =0.001
                elif scale == 11:
                    zoom_factor =0.0001
                ax.set_yticklabels([
                    f'{int(y * zoom_factor)}'
                    for y in y_ticks
                ])

        elif name in ['Forests', 'Shrublands', 'Savannas', 'Grasslands']:
            if name == 'All':
                if scale == 55:
                    y_broken_low = 1800
                    y_broken_high = 2800
                    y_broken_max = 4000

                elif scale == 11:
                    y_broken_low = 18000
                    y_broken_high = 28000
                    y_broken_max = 40000

            ax.axis('off')
            bax = brokenaxes(
                ylims=((0, y_broken_low), (y_broken_high, y_broken_max)),
                hspace=0.1,
                height_ratios=[1, 3],  # 上图占1份，下图占2份。数值越大，对应的部分占地越广
                subplot_spec=gs_inner[0],
                d=0.003)
            # for ax in bax.axs:
            #     ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(abs(x))}"))
            if scale == 55:
                bax.axs[0].set_yticks([3000, 3500, 4000])
                bax.axs[1].set_yticks([0, 500, 1000, 1500])
            elif scale == 11:
                bax.axs[0].set_yticks([30000, 35000, 40000])
                bax.axs[1].set_yticks([0, 5000, 10000, 15000])
            for ax in bax.axs:
                # 获取当前y轴刻度值
                y_ticks = ax.get_yticks()
                # 根据y轴刻度值生成标签
                if scale == 55:
                    zoom_factor = 0.001
                elif scale == 11:
                    zoom_factor = 0.0001
                ax.set_yticklabels([
                    f'{int(y * zoom_factor)}'
                    for y in y_ticks
                ])

        elif name in ['Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']:
            ax.axis('off')
            if scale == 55:
                y_broken_low = 2200
                y_broken_high = 6000
                y_broken_max = 7000

            elif scale == 11:
                y_broken_low = 22000
                y_broken_high = 60000
                y_broken_max = 70000

            bax = brokenaxes(
                ylims=((0, y_broken_low), (y_broken_high, y_broken_max)),
                hspace=0.1,
                height_ratios=[1, 4],  # 上图占1份，下图占2份。数值越大，对应的部分占地越广
                subplot_spec=gs_inner[0],
                d=0.003)

            bax.axs[0].set_yticks([y_broken_high, y_broken_max])
            if scale == 55:
                bax.axs[1].set_yticks([0, 1000, 2000])
            elif scale == 11:
                bax.axs[1].set_yticks([0, 10000, 20000])
            for ax in bax.axs:
                # 获取当前y轴刻度值
                y_ticks = ax.get_yticks()
                # 根据y轴刻度值生成标签
                if scale == 55:
                    zoom_factor = 0.001
                elif scale == 11:
                    zoom_factor = 0.0001
                ax.set_yticklabels([
                    f'{int(y * zoom_factor)}'
                    for y in y_ticks
                ])

        # 设置bar的位置
        bin_centers = (bins[:-1] + bins[1:]) / 2
        print(f'bin_centers:{bin_centers}')

        color_list = ['#762a83', '#9970ab', '#c2a5cf', '#e7d4e8',
                      '#d9f0d3', '#a6dba0', '#5aae61', '#1b7837']

        cmap = mpl.colors.ListedColormap(color_list)

        norm = mpl.colors.BoundaryNorm(bins, cmap.N)

        bin_colors = cmap(norm(bin_centers))

        total_width = 0.2  # 一个刻度位内柱子的总占用宽度
        # n = 2  # 类别数量
        # width = total_width / n  # 单个柱子的宽度

        bax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

        # bax.bar(bin_centers, count_lt0_sig, width=total_width, linewidth = 0.4, hatch='///', facecolor='none', edgecolor='#f46d43', label='Sig-advance', zorder= 2)#, edgecolor='black', linewidth=0.5)
        # bax.bar(bin_centers, count_lt0_all, width=total_width, linewidth = 0.5, color='#fee090', label='Nonsig-advance', zorder= 1)
        # bax.bar(bin_centers, count_gte0_all, width=total_width, linewidth = 0.5, color='#e0f3f8', label='Nonsig-delay', zorder= 1)
        # bax.bar(bin_centers, count_gte0_sig, width=total_width, linewidth = 0.4, hatch='///', facecolor='none',edgecolor='#74add1', label='Sig-delay', zorder= 2)

        # 逐个柱子绘制，以确保颜色严格对应
        for j in range(len(count_all)):
            bax.bar(
                bin_centers[j],
                count_all[j],
                width=total_width,
                color=bin_colors[j],
                linewidth=0.5,
                zorder=1,
                edgecolor='none'
            )
        bax.bar(
            bin_centers,
            count_sig,
            width=total_width,
            hatch='/////',
            facecolor='none',
            edgecolor='black',
            linewidth= 0.4,
            zorder=2
        )

        bax.tick_params(axis='both', length=2, pad=3)

        bax.set_xlim(-1, 1)
        xticks = np.arange(-1, 1.01, 0.25)

        bax.axs[1].set_xticks(
            xticks,
            ['0' if x == 0 else
             f'{int(x * 10)}' if x * 10 == int(x * 10) else
             f'{x * 10}'
             for x in xticks
             ]
        )

        bax.axs[1].set_xticklabels(
            bax.axs[1].get_xticklabels(),
            rotation=45
        )

        ## 控制y轴 10的n次方
        if name in ['All', 'AI']:
            if scale == 55:
                ypos = 1.75
            elif scale == 11:
                ypos = 1.55
        else:
            ypos = 1.43
        if scale == 55:
            text = r'$×10^{3}$'
        elif scale == 11:
            text = r'$×10^{4}$'
        bax.axs[0].text(
            -0.1,  # x = 刻度位置（数据坐标）
            ypos,  # y = 稍微往下（轴坐标）
            text,  # 你想要的内容
            transform=bax.axs[0].transAxes,
            ha='left',  # 向右展开（避免压缩）
            va='top',
            rotation=0,
            fontsize=8,
            clip_on=False
        )

        if name == 'All':
            x_pos = 0.9  # x = 刻度位置（数据坐标）
            y_pos = -0.2  # y = 稍微往下（轴坐标）
        else:
            x_pos = 0.81  # x = 刻度位置（数据坐标）
            y_pos = -0.2  # y = 稍微往下（轴坐标）

        ## 控制x轴 10的n次方
        bax.axs[1].text(
            x_pos,  # x = 刻度位置（数据坐标）
            y_pos,  # y = 稍微往下（轴坐标）
            r'$×10^{-1}$',  # 你想要的内容
            transform=bax.axs[1].transAxes,
            ha='left',  # 向右展开（避免压缩）
            va='top',
            rotation=0,
            fontsize=8,
            clip_on=False
        )

        # # 2. 应用绝对值格式化
        # for ax in bax.axs:
        #     ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(abs(x))}"))

        if name == 'All':
            x_position = 0.5
            y_position = -0.52
        else:
            x_position = 0.5
            y_position = -0.52

        # Colorbar
        sig_patch = mpatches.Patch(
            facecolor='white',
            edgecolor='black',
            hatch='/////',
            label='Significant'
        )
        bax.axs[1].legend(
            handles=[sig_patch],
            loc='lower center',
            bbox_to_anchor=(x_position, y_position),
            ncol=1,
            frameon=False,  # 控制 legend（图例）外框是否显示
            handlelength=1,
            handleheight=1,
            columnspacing=0.5
        )

        if name == 'All':
            y_labelpad = 15
        else:
            y_labelpad = 15
        bax.set_ylabel('Frequency', labelpad=y_labelpad)  # 控制标签与刻度距离

    elif data_type == 'pheno cv':
        if name == 'All':
            if scale == 55:
                y_broken_low = 4500
                y_broken_high = 14000
                y_broken_max = 15000
                height_ratios = [1, 4]

            elif scale == 11:
                y_broken_low = 50000
                y_broken_high = 120000
                y_broken_max = 140000
                height_ratios = [1, 2]

            bax = brokenaxes(
                ylims=((0, y_broken_low), (y_broken_high, y_broken_max)),
                hspace=0.1,
                height_ratios=height_ratios,  # 上图占1份，下图占4份，数值越大，对应的部分占地越广
                subplot_spec=gs_inner[0],
                d=0.005
            )
            if scale == 55:
                bax.axs[0].set_yticks([14000, 15000])
                bax.axs[1].set_yticks([0, 1000, 2000, 3000, 4000])
            elif scale == 11:
                bax.axs[0].set_yticks([120000, 140000])
                bax.axs[1].set_yticks([0, 10000, 20000, 30000, 40000, 50000])

            bin_centers = (bins[:-1] + bins[1:]) / 2

            orange_cmap = mpl.colormaps['Oranges']
            colors = orange_cmap(np.linspace(0.2, 0.9, 8))

            bax.bar(bin_centers, count_cv, width=0.02, color=colors, label='POS cv')

            bax.tick_params(axis='both', length=2, pad=3)

            bax.set_xlim(0, 0.24)
            xticks = np.arange(0, 0.241, 0.03)

            # for ax_ in bax.axs:
            #     ax_.set_xticklabels(ax_.get_xticklabels(), rotation=90)

            bax.axs[1].set_xticks(
                xticks,
                ['0' if x == 0 else f'{int(x * 100)}' for x in xticks]
            )

            tick_size = plt.rcParams['xtick.labelsize']

            ## 控制y轴 10的n次方
            if scale == 55:
                text = r'$×10^{3}$'
            elif scale == 11:
                text = r'$×10^{4}$'

            bax.axs[0].text(
                -0.11,  # x = 刻度位置（数据坐标）
                1.65,  # y = 稍微往下（轴坐标）
                text,  # 你想要的内容
                transform=bax.axs[0].transAxes,
                ha='left',  # 向右展开（避免压缩）
                va='top',
                rotation=0,
                fontsize=8,
                clip_on=False
            )

            ## 控制x轴 10的n次方
            bax.axs[1].text(
                0.9,  # x = 刻度位置（数据坐标）
                -0.16,  # y = 稍微往下（轴坐标）
                r'$×10^{-2}$',  # 你想要的内容
                transform=bax.axs[1].transAxes,
                ha='left',  # 向右展开（避免压缩）
                va='top',
                rotation=0,
                fontsize=8,
                clip_on=False
            )

            # bax.set_xlabel('POS CV', labelpad=33) #控制标签与刻度距离

            for ax in bax.axs:
                # 获取当前y轴刻度值
                y_ticks = ax.get_yticks()
                # 根据y轴刻度值生成标签
                if scale == 55:
                    zoom_factor = 0.001
                elif scale == 11:
                    zoom_factor = 0.0001
                ax.set_yticklabels([
                    f'{int(y * zoom_factor)}'
                    for y in y_ticks
                ])


            bax.set_ylabel('Frequency', labelpad=15)  # 控制标签与刻度距离

        elif name in ['Forests', 'Shrublands', 'Savannas', 'Grasslands']:
            # bax = brokenaxes(
            #     ylims=((0, 3500), (4000, 5000)),
            #     hspace=0.1,
            #     height_ratios=[1, 1],  # 上图占1份，下图占2份。数值越大，对应的部分占地越广
            #     subplot_spec=sub_gs
            # )
            #
            # bax.axs[0].set_yticks([15000, 16000, 17000])
            # bax.axs[1].set_yticks([1000, 3500])

            ax1 = fig.add_subplot(gs_inner[0])

            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)

            bin_centers = (bins[:-1] + bins[1:]) / 2
            print(f'bin_centers:{bin_centers}')

            orange_cmap = mpl.colormaps['Oranges']
            colors = orange_cmap(np.linspace(0.2, 0.9, 8))

            ax1.bar(bin_centers, count_cv, width=0.02, color=colors, label='POS cv')

            if scale == 55:
                ymax = 5000
                interval = 1000
            elif scale == 11:
                ymax = 500000
                interval = 10000

            ax1.set_ylim(0, ymax)
            ax1.set_yticks(np.arange(0, ymax+0.1, interval))

            # 根据y轴刻度值生成标签
            y_ticks = ax1.get_yticks()
            if scale == 55:
                zoom_factor = 0.001
                text ='$×10^{-3}$'
            elif scale == 11:
                zoom_factor = 0.0001
                text = '$×10^{-4}$'
            ax.set_yticklabels([
                f'{int(y * zoom_factor)}'
                for y in y_ticks
            ])

            ax1.text(0, 1.07, text,
                     transform=ax.transAxes,
                     ha='center',
                     va='center',
                     rotation=0)

            ax1.set_xlim(0, 0.24)
            ax1.set_xticks(np.arange(0, 0.241, 0.03))
            ax1.tick_params(axis='x')
            xticks = ax1.get_xticks()
            ax1.set_xticks(
                xticks,
                ['0' if x == 0 else f'{int(x * 100)}' for x in xticks], fontsize = 8
            )

            tick_size = plt.rcParams['xtick.labelsize']

            ## 控制x轴 10的n次方
            ax1.text(
                0.85,  # x = 刻度位置（数据坐标）
                -0.125,  # y = 稍微往下（轴坐标）
                r'$×10^{-2}$',  # 你想要的内容
                transform=ax1.transAxes,
                ha='left',  # 向右展开（避免压缩）
                va='top',
                rotation=0,
                fontsize=8,
                clip_on=False
            )


            # ax1.set_xlabel('POS CV', labelpad=10)  # 控制标签与刻度距离

            ax1.set_ylabel('Frequency', labelpad=5)  # 控制标签与刻度距离

        elif name in ['Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']:

            if scale == 55:
                y_broken_low = 3000
                y_broken_high = 7500
                y_broken_max = 8500

            elif scale == 11:
                y_broken_low = 30000
                y_broken_high = 75000
                y_broken_max = 85000

            bax = brokenaxes(
                ylims=((0, y_broken_low), (y_broken_high, y_broken_max)),
                hspace=0.1,
                height_ratios=[1, 4],  # 上图占1份，下图占2份。数值越大，对应的部分占地越广
                subplot_spec=gs_inner[0],
                d=0.003
            )

            bax.axs[0].set_yticks([y_broken_high, y_broken_max])
            if scale == 55:
                bax.axs[1].set_yticks([0, 1000, 2000, 3000])
            elif scale == 11:
                bax.axs[1].set_yticks([0, 10000, 20000, 30000])

            ### y轴
            for ax in bax.axs:
                # 获取当前y轴刻度值
                y_ticks = ax.get_yticks()
                ax.set_yticks(y_ticks)
                # 根据y轴刻度值生成标签

                if scale == 55:
                    zoom_factor = 0.001
                    text = '$×10^{-3}$'
                elif scale == 11:
                    zoom_factor = 0.00001
                    text = '$×10^{-4}$'

                ax.set_yticklabels([
                    '0' if y == 0 else
                    f'{int(y * zoom_factor)}' if y == int(y) else
                    f'{y * zoom_factor:.1f}'
                    for y in y_ticks
                ])

            tick_size = plt.rcParams['xtick.labelsize']

            bax.axs[0].text(
                -0.03,  # x = 刻度位置（数据坐标）
                1.46,  # y = 稍微往下（轴坐标）
                text,  # 你想要的内容
                transform=bax.axs[0].get_xaxis_transform(),
                ha='left',  # 向右展开（避免压缩）
                va='top',
                fontsize=8,
                clip_on=False
            )

            bin_centers = (bins[:-1] + bins[1:]) / 2

            orange_cmap = mpl.colormaps['Oranges']
            colors = orange_cmap(np.linspace(0.2, 0.9, 8))

            bax.bar(bin_centers, count_cv, width=0.02, color=colors, label='POS cv')

            bax.set_xlim(0, 0.24)
            xticks = np.arange(0, 0.241, 0.03)

            bax.axs[1].set_xticks(
                xticks,
                ['0' if x == 0 else f'{int(x * 100)}' for x in xticks]
            )
            bax.axs[1].set_xticklabels(
                bax.axs[1].get_xticklabels(),
                fontsize=8
            )

            bax.axs[1].text(
                0.85,  # x = 刻度位置（数据坐标）
                -0.125,  # y = 稍微往下（轴坐标）
                r'$×10^{-2}$',  # 你想要的内容
                transform=bax.axs[1].transAxes,
                ha='left',  # 向右展开（避免压缩）
                va='top',
                rotation=0,
                fontsize=8,
                clip_on=False
            )


            # bax.set_xlabel('POS CV', labelpad=35) #控制标签与刻度距离

            bax.set_ylabel('Frequency', labelpad=20)  # 控制标签与刻度距离

    # if name == 'All':
    #     if data_type == 'pheno slope':
    #         bax.set_title(f'Frequency of POS trend', pad=10, fontweight='bold')
    #     elif data_type == 'pheno cv':
    #         bax.set_title(f'Frequency of POS CV ', pad=10, fontweight='bold')

    # plt.show()

## b
def plot_pheno_slope_forDiffvegType_and_AI(data_slope, data_cv, veg_data, ai_data, data_type, ax):

    if data_type == 'pheno slope':
        data = data_slope
    elif data_type == 'pheno cv':
        data = data_cv

    # 1. 数据准备
    veg_list = [data[(veg_data == i) & np.isfinite(data)] for i in [1, 2, 3, 4]]
    veg_labels = ['Forests', 'Shrublands', 'Savannas', 'Grasslands']

    ai_list = [
        data[(ai_data == 2) & np.isfinite(data)],  # Arid
        data[((ai_data == 3) | (ai_data == 4)) & np.isfinite(data)],  # Semi-Arid
        data[(ai_data == 5) & np.isfinite(data)],  # Dry sub-humid
        data[(ai_data == 6) & np.isfinite(data)]  # Humid
    ]
    ai_labels = ['Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']

    full_data = veg_list + ai_list
    x_positions = [1, 2, 3, 4, 5, 6, 7, 8]  # X轴物理位置

    # 修正：安全计算均值
    full_data_mean = [np.mean(d) if len(d) > 0 else 0 for d in full_data]

    ### plot
    # fig, ax = plt.subplots(figsize=(4, 4))  # 稍微调宽一点，文字不拥挤
    # plt.subplots_adjust(bottom=0.2)  # 底部留白给旋转后的标签

    ### plot
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
    # print(v_colors)

    # # 或者直接使用
    # colors = plt.cm.Paired(np.arange(6) / 6)

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
    ax1.tick_params(axis='both', which='major', length=2, pad=3)

    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(veg_labels + ai_labels, rotation=90)

    if data_type == 'pheno slope':
        ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax1.axvline(4.5, color='black', linestyle='-', linewidth=1)  # 垂直线分界

    if data_type == 'pheno slope':
        ax1.set_ylim(-0.75, 0.75)
        ax1.set_yticks(np.arange(-0.75, 0.751, 0.25))
        ax1.set_yticklabels(f'{int(x*10)}' for x in np.arange(-0.75, 0.751, 0.25))
        ax.text(-0.01, 1.08, r'$×10^{-1}$',
                transform=ax.transAxes,
                ha='center',
                va='center',
                # rotation=90,
                fontsize=8)
        ax1.set_ylabel('PPT trend (days/yr)', labelpad=2)
    elif data_type == 'pheno cv':
        ax1.set_ylim(0, 0.15)
        ax1.set_yticks(np.arange(0, 0.16, 0.05))
        ax1.set_yticklabels(f'{int(x * 100)}' for x in np.arange(0, 0.16, 0.05))
        ax.text(-0.01, 1.08, r'$×10^{-2}$',
                transform=ax.transAxes,
                ha='center',
                va='center',
                # rotation=90,
                fontsize=8)
        ax1.set_ylabel('PPT CV', labelpad=2)

    # 修正：设置均值文本标注
    for i, m in enumerate(full_data_mean):

        color = v_colors[i]

        if data_type == 'pheno slope':

            if i == 0:  # Forests
                yheight = 0.40
            elif i == 1:  # Shrublands
                yheight = 0.40
            elif i == 2:  # Savannas
                yheight = 0.50
            elif i == 3:  # Grasslands
                yheight = 0.60
            elif i == 4:  # Arid
                yheight = 0.60
            elif i == 5:  # Semi-arid
                yheight = 0.50
            elif i == 6:  # Dry sub-humid
                yheight = 0.40
            elif i == 7:  # Humid
                yheight = 0.35

        elif data_type == 'pheno cv':

            if i in [0, 1, 2, 6]:
                yheight = 0.09
            elif i == 7:
                yheight = 0.1
            elif i in [3, 4]:
                yheight = 0.14
            elif i == 5:
                yheight = 0.12

        ax1.text(
            x_positions[i],
            yheight,
            f'{m:.2f}',
            color=color,
            ha='center',
            va='bottom',
            fontsize=6
        )

    # if data_type == 'pheno slope':
    #     ax1.set_title(f'Mean of POS trend', pad=10, fontweight='bold')
    # elif data_type == 'pheno cv':
    #     ax1.set_title(f'Mean of POS CV', pad=10, fontweight='bold')

    # plt.show()

def plot_fig1(data_slope, data_pvalue, data_cv):

    # 统一设置所有字体大小
    plt.rcParams.update({
        'font.family': 'Arial',

        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Arial',  # 正常
        'mathtext.it': 'Arial:italic',  # 斜体
        'mathtext.bf': 'Arial:bold',  # 粗体

        'mathtext.default': 'regular',  # 避免自动变斜体

        'font.size': 10,
        'axes.titlesize': 10,
        'axes.labelsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })

    # 创建画布
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
    plot_pheno_slope_and_pvalue_forAllvegType(data_slope, data_pvalue, data_cv, -1, 1, 'pheno slope', 'All', ax = ax1)

    ### fig1 a right
    plot_pheno_slope_lt0_sig_and_lt0_nosig_and_gt0_forAllvegType(data_slope, data_pvalue, data_cv, 'pheno slope', -1, 1, 'All', ax = ax2)

    ### fig1 b
    plot_pheno_slope_forDiffvegType_and_AI(data_slope, data_cv, veg_type_data, ai_type_data, 'pheno slope', ax = ax3)

    ### fig1 c left
    plot_pheno_slope_and_pvalue_forAllvegType(data_slope, data_pvalue, data_cv,0, 0.24,'pheno cv', 'All', ax = ax4)

    ### fig1 c right
    plot_pheno_slope_lt0_sig_and_lt0_nosig_and_gt0_forAllvegType(data_slope, data_pvalue, data_cv, 'pheno cv', 0, 0.24, 'All', ax = ax5)

    ### fig1 d
    plot_pheno_slope_forDiffvegType_and_AI(data_slope, data_cv, veg_type_data, ai_type_data, 'pheno cv', ax = ax6)

    plt.savefig(rf'{fig_output}\All\POS_Slope_CV_{scale}km.png',
                dpi=600, bbox_inches='tight')

    # plt.show()


### S1-4
def plot_S1_2_forSlope(data_slope, data_pvalue, data_cv):

    # 1. 数据准备
    veg_slope_list = [np.where(veg_type_data == i, data_slope, np.nan) for i in [1, 2, 3, 4]]
    veg_pvalue_list = [np.where(veg_type_data == i, data_pvalue, np.nan) for i in [1, 2, 3, 4]]
    veg_labels = ['Forests', 'Shrublands', 'Savannas', 'Grasslands']

    ai_slope_list = [
        np.where(ai_type_data == 2, data_slope, np.nan),  # Arid
        np.where((ai_type_data == 3) | (ai_type_data == 4), data_slope, np.nan),  # Semi-Arid (合并 3 和 4)
        np.where(ai_type_data == 5, data_slope, np.nan),  # Dry sub-humid
        np.where(ai_type_data == 6, data_slope, np.nan)  # Humid
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

    for veg_slope, veg_pvalue, veg_name, (ax_l, ax_r) in zip(veg_slope_list, veg_pvalue_list, veg_labels, ax_pairs):
        # 绘制左侧地图组 (ax_l)
        plot_pheno_slope_and_pvalue_forAllvegType(veg_slope, veg_pvalue, data_cv, -1, 1, 'pheno slope', veg_name, ax=ax_l)

        # 绘制右侧柱状图组 (ax_r)
        plot_pheno_slope_lt0_sig_and_lt0_nosig_and_gt0_forAllvegType(veg_slope, veg_pvalue, data_cv, 'pheno slope', -1, 1, veg_name, ax=ax_r)

    plt.savefig(rf'{fig_output}\Veg\POS_Slope_{scale}km_Vegtype.png', dpi=600, bbox_inches='tight')

    # plt.show()

    ############################################## AI类型 ##################################################

    # 创建5个子图：4个地图 + 1个colorbar
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

    for ai_slope, ai_pvalue, ai_name, (ax_l, ax_r) in zip(ai_slope_list, ai_pvalue_list, ai_labels, ax_pairs):
        # 绘制左侧地图组 (ax_l)
        # 注意：我假设你修改了函数签名，加入了 veg_name
        plot_pheno_slope_and_pvalue_forAllvegType(ai_slope, ai_pvalue, data_cv, -1, 1, 'pheno slope', ai_name, ax=ax_l)

        # 绘制右侧柱状图组 (ax_r)
        plot_pheno_slope_lt0_sig_and_lt0_nosig_and_gt0_forAllvegType(ai_slope, ai_pvalue, data_cv, 'pheno slope', -1, 1, ai_name, ax=ax_r)

    plt.savefig(rf'{fig_output}\AI\POS_Slope_{scale}km_AItype.png', dpi=600, bbox_inches='tight')

    # # plt.show()


def plot_S3_4_forCV(data_slope, data_pvalue, data_cv):

    # 1. 数据准备
    veg_cv_list = [np.where(veg_type_data == i, data_cv, np.nan) for i in [1, 2, 3, 4]]
    veg_labels = ['Forests', 'Shrublands', 'Savannas', 'Grasslands']

    ai_cv_list = [
        np.where(ai_type_data == 2, data_cv, np.nan),  # Arid
        np.where((ai_type_data == 3) | (ai_type_data == 4), data_cv, np.nan),  # Semi-Arid (合并 3 和 4)
        np.where(ai_type_data == 5, data_cv, np.nan),  # Dry sub-humid
        np.where(ai_type_data == 6, data_cv, np.nan)  # Humid
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

    # 创建5个子图：4个地图 + 1个colorbar
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

    for veg_cv, veg_name, (ax_l, ax_r) in zip(veg_cv_list, veg_labels, ax_pairs):
        # 绘制左侧地图组 (ax_l)
        plot_pheno_slope_and_pvalue_forAllvegType(data_slope, data_pvalue, veg_cv, 0, 0.24, 'pheno cv', veg_name, ax=ax_l)

        # 绘制右侧柱状图组 (ax_r)
        plot_pheno_slope_lt0_sig_and_lt0_nosig_and_gt0_forAllvegType(data_slope, data_pvalue, veg_cv, 'pheno cv', 0, 0.24, veg_name, ax=ax_r)

    plt.savefig(rf'{fig_output}\Veg\POS_CV_{scale}km_Vegtype.png', dpi=600, bbox_inches='tight')

    # plt.show()

    ############################################## AI类型 ##################################################

    # 创建5个子图：4个地图 + 1个colorbar
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

    for ai_cv, ai_name, (ax_l, ax_r) in zip(ai_cv_list, ai_labels, ax_pairs):
        # 绘制左侧地图组 (ax_l)
        # 注意：我假设你修改了函数签名，加入了 veg_name
        plot_pheno_slope_and_pvalue_forAllvegType(data_slope, data_pvalue, ai_cv, 0, 0.24, 'pheno cv', ai_name, ax=ax_l)

        # 绘制右侧柱状图组 (ax_r)
        plot_pheno_slope_lt0_sig_and_lt0_nosig_and_gt0_forAllvegType(data_slope, data_pvalue, ai_cv, 'pheno cv', 0, 0.24, ai_name, ax=ax_r)

    plt.savefig(
        rf'{fig_output}\AI\POS_CV_{scale}km_AItype.png',
        dpi=600, bbox_inches='tight')
    #
    # # plt.show()


# ### All
plot_fig1(pheno_slope, pheno_slope_pvalue, pheno_cv)
print('fig1 plot done!')
# ## S1-4
# plot_S1_2_forSlope(pheno_slope, pheno_slope_pvalue, pheno_cv)
# print('S1-2 plot done!')
# plot_S3_4_forCV(pheno_slope, pheno_slope_pvalue, pheno_cv)
# print('S3-4 plot done!')


### 计算每年POS在不同植被类型的均值、25th、75th的值

## 区分欧亚大陆、北美大陆

###### 面划分 #######
north_america = rf'{input_same_path}\Continental boundaries\NorthAmerica_tif_vision.tif'
europe = rf'{north_america}\Continental boundaries\Europe_tif_vision.tif'
asia = rf'{north_america}\Continental boundaries\Asia_tif_vision.tif'

north_america_tif = gdal.Open(north_america)
europe_type_tif = gdal.Open(europe)
asia_type_tif = gdal.Open(asia)

def aggregate(tif_data):

    ds = gdal.Warp(
        "",
        tif_data,
        format="MEM",
        width=cols,
        height=rows,
        outputBounds=(lon_min, lat_min, lon_max, lat_max),
        dstSRS="EPSG:4326",
        resampleAlg=gdal.GRA_Max,
        srcNodata=1,  #输入数据中，值为 1 的像元表示 NoData（背景），不要参与重采样计算。
        dstNodata=1   #如果输出像元没有任何有效值，则输出值写成 1。
    )

    ds_array = ds.ReadAsArray()

    mask = (ds_array == 0)

    return mask

north_america_mask = aggregate(north_america_tif)
europe_mask = aggregate(europe_type_tif)
asia_mask = aggregate(asia_type_tif)


outlier_pheno_stack_na = np.where(
    north_america_mask,
    outlier_pheno_stack,
    np.nan
)

outlier_pheno_stack_europe = np.where(
    europe_mask,
    outlier_pheno_stack,
    np.nan
)

outlier_pheno_stack_asia = np.where(
    asia_mask,
    outlier_pheno_stack,
    np.nan
)



def cal_each_year_mean_25th_75th(stack1, stack2, stack3, stack4, ax, scope):
    rows = []
    for k, year in enumerate(years):

        if scope == 'Global':
            array_data_ofyear = stack1[k, :, :]
        elif scope == 'North America':
            array_data_ofyear = stack2[k, :, :]
        elif scope == 'Europe':
            array_data_ofyear = stack3[k, :, :]
        elif scope == 'Asia':
            array_data_ofyear = stack4[k, :, :]

        row = {'year': year}

        # =========================
        # Vegetation types
        # =========================
        veg_data_list = [
            np.where(veg_type_data == 1, array_data_ofyear, np.nan),  # Forest
            np.where(veg_type_data == 2, array_data_ofyear, np.nan),  # Shrublands
            np.where(veg_type_data == 3, array_data_ofyear, np.nan),  # Savanna
            np.where(veg_type_data == 4, array_data_ofyear, np.nan)   # Grasslands
        ]

        veg_labels = ['Forests', 'Shrublands', 'Savannas', 'Grasslands']

        for veg_data, veg_name in zip(veg_data_list, veg_labels):

            row[veg_name] = np.nanmean(veg_data)
            row[f'{veg_name}_25th_error'] = np.nanmean(veg_data) - np.nanpercentile(veg_data, 25)
            row[f'{veg_name}_75th_error'] = np.nanpercentile(veg_data, 75) - np.nanmean(veg_data)

        # =========================
        # Aridity classes
        # =========================
        ai_data_list = [
            np.where(ai_type_data == 2, array_data_ofyear, np.nan),                              # Arid
            np.where((ai_type_data == 3) | (ai_type_data == 4), array_data_ofyear, np.nan),      # Semi-arid
            np.where(ai_type_data == 5, array_data_ofyear, np.nan),                              # Dry sub-humid
            np.where(ai_type_data == 6, array_data_ofyear, np.nan)                               # Humid
        ]

        ai_labels = ['Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']

        for ai_data, ai_name in zip(ai_data_list, ai_labels):

            row[ai_name] = np.nanmean(ai_data)
            row[f'{ai_name}_25th_error'] = np.nanmean(ai_data) - np.nanpercentile(ai_data, 25)
            row[f'{ai_name}_75th_error'] = np.nanpercentile(ai_data, 75) - np.nanmean(ai_data)

        rows.append(row)

    # 转换为DataFrame
    df_year_mean_25th_75th = pd.DataFrame(rows)

    # 调整列顺序
    df_year_mean_25th_75th = df_year_mean_25th_75th[
        ['year',
         'Forests', 'Forests_25th_error', 'Forests_75th_error',
         'Shrublands', 'Shrublands_25th_error', 'Shrublands_75th_error',
         'Savannas', 'Savannas_25th_error', 'Savannas_75th_error',
         'Grasslands', 'Grasslands_25th_error', 'Grasslands_75th_error',
         'Arid', 'Arid_25th_error', 'Arid_75th_error',
         'Semi-arid', 'Semi-arid_25th_error', 'Semi-arid_75th_error',
         'Dry sub-humid', 'Dry sub-humid_25th_error', 'Dry sub-humid_75th_error',
         'Humid', 'Humid_25th_error', 'Humid_75th_error']
    ]

    print(df_year_mean_25th_75th.head())

    df_year_mean_25th_75th.to_csv(f'D:\CAU\phenology_swc_vpd\Global_test5\Fig\Fig 1 POS trend and CV\Scatter Plot in Site or pixel\POS_VegType_AiType_year_mean_25th_75th_{scope}.csv',
                              index=False,
                              encoding='utf-8-sig'
                              )


    ######### Plot #################

    fig = ax.get_figure()

    ax.axis('off')

    gs_inner = ax.get_subplotspec().subgridspec(
        1, 2,
        width_ratios=[1, 1],
        wspace=0.3
    )

    ax1 = fig.add_subplot(gs_inner[0])
    ax2 = fig.add_subplot(gs_inner[1])

    ## Veg
    x = df_year_mean_25th_75th['year']
    y = df_year_mean_25th_75th
    ax1.plot(x, y['Forests'], color = '#086A10', marker='o', markersize=3, linewidth=1.5, label = 'Forests')
    ax1.plot(x, y['Shrublands'], color = '#965724', marker='o', markersize=3, linewidth=1.5, label = 'Shrublands')
    ax1.plot(x, y['Savannas'], color = '#fdbd10', marker='o', markersize=3, linewidth=1.5, label = 'Savannas')
    ax1.plot(x, y['Grasslands'], color = '#B6FF05', marker='o', markersize=3, linewidth=1.5, label = 'Grasslands')

    ax1.set_ylim(140, 220)
    ax1.set_yticklabels(np.arange(140, 240, 20))

    # ax1.set_xlim(2001, 2024)

    ax1.set_ylabel('PPT (days)')

    if scope == 'Asia':
        ax1.set_xlabel('Year')
        ax1.legend(loc='lower center',
                  bbox_to_anchor=(0.5, -0.6),
                  ncol=2,
                  # fontsize=7,
                  frameon=False)

    ## AI
    x = df_year_mean_25th_75th['year']
    y = df_year_mean_25th_75th
    ax2.plot(x, y['Arid'], color='#e5505a', marker='o', markersize=3, linewidth=1.5, label='Arid')
    ax2.plot(x, y['Semi-arid'], color='#ff9f1c', marker='o', markersize=3, linewidth=1.5, label='Semi-arid')
    ax2.plot(x, y['Dry sub-humid'], color='#01b287', marker='o', markersize=3, linewidth=1.5, label='Dry sub-humid')
    ax2.plot(x, y['Humid'], color='#854c9e', marker='o', markersize=3, linewidth=1.5, label='Humid')

    ax2.set_ylim(140, 220)
    ax2.set_yticklabels(np.arange(140, 240, 20))

    # ax2.set_xlim(2001, 2024)

    if scope == 'Asia':
        ax2.set_xlabel('Year')
        ax2.legend(loc='lower center',
                   bbox_to_anchor=(0.5, -0.6),
                   ncol=2,
                   # fontsize=7,
                   frameon=False)


def cal_continent_yearly_ppt_and_plot(stack1, stack2, stack3, stack4):

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


    fig = plt.figure(figsize=(8.2, 11), dpi=100)
    gs = gridspec.GridSpec(4, 1,
                           # width_ratios=[1, 1],  # 三列的宽度比
                           height_ratios=[1, 1, 1, 1],  # 最后一个给colorbar
                           hspace=0.5)#, wspace=0.2)


    continent_list = ['Global', 'North America', 'Europe', 'Asia']

    for i, continent in enumerate(continent_list):
        if continent == 'Global':
            ax = plt.subplot(gs[0, 0])
        elif continent == 'North America':
            ax = plt.subplot(gs[1, 0])
        elif continent == 'Europe':
            ax = plt.subplot(gs[2, 0])
        elif continent == 'Asia':
            ax = plt.subplot(gs[3, 0])

        cal_each_year_mean_25th_75th(stack1, stack2, stack3, stack4, ax, continent)

    plt.savefig(rf'{fig_output}\Scatter Plot in Site or pixel\Continent_PPT_yearly_{scale}km.png', dpi = 300, bbox_inches='tight')

    # plt.tight_layout()
    # plt.show()


# cal_continent_yearly_ppt_and_plot(outlier_pheno_stack, outlier_pheno_stack_na, outlier_pheno_stack_europe, outlier_pheno_stack_asia)