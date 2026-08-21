
import os
import glob
import datetime
import sys

# import matplotlib.colors as colors
import matplotlib.pyplot as plt
import pandas as pd
from duckdb.experimental.spark.sql.functions import isnan
from mpl_toolkits.basemap import Basemap
from osgeo import gdal
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
import pingouin as pg
from scipy.stats import pearsonr
from joblib import Parallel, delayed, parallel_backend
import matplotlib as mpl
from matplotlib import colormaps
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.model_selection import train_test_split
import pymannkendall as mk
from xgboost import XGBRegressor
import shap
import gc
from sklearn.preprocessing import StandardScaler
from joblib import parallel_backend
from matplotlib.patches import Circle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import math
import random



######### Function #########
### Read SM and VPD bands
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
            gdal.GDT_Float32  # Default data type (can be modified as needed)
        )
        if not output_ds:
            raise RuntimeError(f"Unable to create output file: {output_path}")

        output_band = output_ds.GetRasterBand(1)  # Single band index is 1
        output_band.WriteArray(data, 0, 0)  # Write data (0,0 indicates top-left starting point)

        output_ds.SetProjection(crs)

        output_ds.SetGeoTransform(transform)

        output_ds = None  # Close dataset (Required! Otherwise the file may be corrupted)
        return True


# def cal_partial(i, j, vars, y, pos_matrix, cor_matrix, sm_matrix, vpd_matrix, ta_matrix, pre_matrix, srad_matrix):
def cal_partial(i, j, vars, y, pos_matrix, sos_matrix, cor_matrix, sm_matrix, vpd_matrix, ta_matrix, pre_matrix, srad_matrix):
# def cal_partial(i, j, vars, y, pos_stack, cor_matrix, sm_matrix, vpd_matrix):

    # mask = (~np.isnan(pos_stack[:, i, j]) &
    #         ~np.isnan(cor_matrix[:, i, j]) &
    #         ~np.isnan(sm_matrix[:, i, j]) &
    #         ~np.isnan(vpd_matrix[:, i, j]) &
    #         ~np.isnan(ta_matrix[:, i, j]) &
    #         ~np.isnan(pre_matrix[:, i, j]) &
    #         ~np.isnan(srad_matrix[:, i, j]))\

    all_vars = np.stack([
        pos_matrix[:, i, j],
        sos_matrix[:, i, j],
        cor_matrix[:, i, j],
        sm_matrix[:, i, j],
        vpd_matrix[:, i, j],
        ta_matrix[:, i, j],
        pre_matrix[:, i, j],
        srad_matrix[:, i, j]
    ], axis=0)
    # mask: time points where all variables are not NaN

    all_year = drought_year_stack[analyzed_start:analyzed_end, i, j]

    mask = np.all((np.isfinite(all_vars)) & (all_year != 2), axis=0)   # Adding ~ before np returns invalid values

    if mask.sum() > 5: # Count number of valid values

        # pos = np.array(pos_matrix[:, i, j][mask])
        #
        # sos_time_series = np.array(sos_matrix[:, i, j][mask])
        # cor_time_series = np.array(cor_matrix[:, i, j][mask])
        # sm_time_series = np.array(sm_matrix[:, i, j][mask])
        # vpd_time_series = np.array(vpd_matrix[:, i, j][mask])
        # ta_time_series = np.array(ta_matrix[:, i, j][mask])
        # pre_time_series = np.array(pre_matrix[:, i, j][mask])
        # srad_time_series = np.array(srad_matrix[:, i, j][mask])

        pos = np.where(mask, pos_matrix[:, i, j], np.nan)

        sos_time_series = np.where(mask, sos_matrix[:, i, j], np.nan)
        cor_time_series = np.where(mask, cor_matrix[:, i, j], np.nan)
        sm_time_series = np.where(mask, sm_matrix[:, i, j], np.nan)
        vpd_time_series = np.where(mask, vpd_matrix[:, i, j], np.nan)
        ta_time_series = np.where(mask, ta_matrix[:, i, j], np.nan)
        pre_time_series = np.where(mask, pre_matrix[:, i, j], np.nan)
        srad_time_series = np.where(mask, srad_matrix[:, i, j], np.nan)

        if variable_type == 'sos and climate':
            pixel_data = pd.DataFrame({
                'POS': pos,
                'SOS': sos_time_series,
                'Coupling': cor_time_series,   # cor
                'SM': sm_time_series,     # sm
                'VPD': vpd_time_series,    # vpd
                'Ta': ta_time_series,   # ta
                'Pre': pre_time_series, # pre
                'Srad': srad_time_series # srad
                })

        elif variable_type == 'only climate':
            pixel_data = pd.DataFrame({
                'POS': pos,
                'Coupling': cor_time_series,  # cor
                'SM': sm_time_series,  # sm
                'VPD': vpd_time_series,  # vpd
                'Ta': ta_time_series,  # ta
                'Pre': pre_time_series,  # pre
                'Srad': srad_time_series  # srad
            })

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

        # Sort and deduplicate: sort by absolute value of correlation coefficient in descending order
        # top_vars = result_df.reindex(result_df['r'].abs().sort_values(ascending=False).index)
        top_vars = result_df.sort_values(by='r', key=lambda x: x.abs(), ascending=False)
        # print('top_vars:', top_vars)

        # Get the 1st, 2nd, and 3rd variables (use drop_duplicates to ensure uniqueness)
        top_vars_unique = top_vars.drop_duplicates(subset='var').head(3)
        # second_vars_unique = top_vars.drop_duplicates(subset='var').head(2)

        # Display top 3 variables
        # print("Top 3 strongest partial correlations:")
        # print(top_vars_unique[['var', 'r', 'p-val']])

        first_varname = top_vars_unique['var'][0]
        second_varname = top_vars_unique['var'][1]
        third_varname = top_vars_unique['var'][2]
        # print(f'top var:{first_varname}\n'
        #       f'secong var:{second_varname}')

        first_r = top_vars_unique['r'][0]
        second_r = top_vars_unique['r'][1]
        third_r = top_vars_unique['r'][2]

        first_p = top_vars_unique['p-val'][0]
        second_p = top_vars_unique['p-val'][1]
        third_p = top_vars_unique['p-val'][2]

        if variable_type == 'sos and climate':
            sos_pcor = result_df.loc[result_df['var'] == 'SOS', 'r'].iloc[0]
        elif variable_type == 'only climate':
            sos_pcor = np.nan
        sm_pcor = result_df.loc[result_df['var'] == 'SM', 'r'].iloc[0]
        vpd_pcor = result_df.loc[result_df['var'] == 'VPD', 'r'].iloc[0]
        cor_pcor = result_df.loc[result_df['var'] == 'Coupling', 'r'].iloc[0]
        ta_pcor = result_df.loc[result_df['var'] == 'Ta', 'r'].iloc[0]
        pre_pcor = result_df.loc[result_df['var'] == 'Pre', 'r'].iloc[0]
        srad_pcor = result_df.loc[result_df['var'] == 'Srad', 'r'].iloc[0]


    else:
        first_varname = 'nan'
        second_varname = 'nan'
        third_varname = 'nan'

        first_r = np.nan
        second_r = np.nan
        third_r = np.nan

        first_p = np.nan
        second_p = np.nan
        third_p = np.nan

        sos_pcor = np.nan
        sm_pcor = np.nan
        vpd_pcor = np.nan
        cor_pcor = np.nan
        ta_pcor = np.nan
        pre_pcor = np.nan
        srad_pcor = np.nan

    return i, j, first_varname, second_varname, third_varname, first_r, second_r, third_r, first_p, second_p, third_p, sos_pcor, sm_pcor, vpd_pcor, cor_pcor, ta_pcor, pre_pcor, srad_pcor
    # return i, j, first_varname, second_varname, first_r, second_r, first_p, second_p, sm_pcor, vpd_pcor, cor_pcor, ta_pcor, pre_pcor, srad_pcor


def cal_ML(i, j, pos_matrix, sos_matrix, cor_matrix, sm_matrix, vpd_matrix, ta_matrix, pre_matrix, srad_matrix):

    if np.sum(np.isfinite(pos_matrix[:, i, j])) > 0:

        ### 1 Get 3x3 grid
        if grid_size == '3*3':
            r_start, r_end = max(0, i - 1), min(rows, i + 2)
            c_start, c_end = max(0, j - 1), min(cols, j + 2)
        elif grid_size == '5*5':
            r_start, r_end = max(0, i - 2), min(rows, i + 3)
            c_start, c_end = max(0, j - 2), min(cols, j + 3)
        elif grid_size == '10*10':
            r_start, r_end = max(0, i - 5), min(rows, i + 5)
            c_start, c_end = max(0, j - 5), min(cols, j + 5)

        if variable_type == 'sos and climate':
            all_vars = np.stack([
                pos_matrix[:, r_start:r_end, c_start:c_end],
                sos_matrix[:, r_start:r_end, c_start:c_end],
                cor_matrix[:, r_start:r_end, c_start:c_end],
                sm_matrix[:, r_start:r_end, c_start:c_end],
                vpd_matrix[:, r_start:r_end, c_start:c_end],
                ta_matrix[:, r_start:r_end, c_start:c_end],
                pre_matrix[:, r_start:r_end, c_start:c_end],
                srad_matrix[:, r_start:r_end, c_start:c_end]
            ], axis=0)
        elif variable_type == 'only climate':
            all_vars = np.stack([
                pos_matrix[:, r_start:r_end, c_start:c_end],
                cor_matrix[:, r_start:r_end, c_start:c_end],
                sm_matrix[:, r_start:r_end, c_start:c_end],
                vpd_matrix[:, r_start:r_end, c_start:c_end],
                ta_matrix[:, r_start:r_end, c_start:c_end],
                pre_matrix[:, r_start:r_end, c_start:c_end],
                srad_matrix[:, r_start:r_end, c_start:c_end]
            ], axis=0)

        ### 2 Standardize in-grid data
        normal_year = drought_year_stack[analyzed_start:analyzed_end, r_start:r_end, c_start:c_end]

        mask = np.all(np.isfinite(all_vars), axis=0) & (normal_year != 2)  # Adding ~ before np returns invalid values

        all_vars_std = np.full_like(all_vars, np.nan, dtype=np.float32)

        for v in range(all_vars.shape[0]):

            var_data = all_vars[v]

            mean_val = np.nanmean(var_data)
            std_val = np.nanstd(var_data)

            if np.isfinite(std_val) and std_val > 0:
                all_vars_std[v] = (var_data - mean_val) / std_val


        ### Get central pixel
        rel_i = i - r_start
        rel_j = j - c_start

        # val_mask = mask[:, rel_i, rel_j]
        # val_data = all_vars_std[:, val_mask, rel_i, rel_j]

        actual_rows = all_vars_std.shape[2]
        actual_cols = all_vars_std.shape[3]

        all_samples_list = []  # All 3x3 samples
        center_index = None
        # neighbor_samples_list = []  # 3x3 grid excluding central pixel

        # if analyze_years_length <= 12:
        #     valid_years = 8
        # elif analyze_years_length > 13:
        #     valid_years = analyze_years_length / 2
        valid_years = 12

        for r_idx in range(actual_rows):
            for c_idx in range(actual_cols):
                m = mask[:, r_idx, c_idx]
                if np.sum(m) > valid_years:
                    pixel_data = all_vars_std[:, m, r_idx, c_idx]
                    all_samples_list.append(pixel_data)
                    if r_idx == rel_i and c_idx == rel_j:
                        center_index = len(all_samples_list) - 1
        if center_index is None:
            return (i, j,
                    'nan', 'nan', 'nan',
                    np.nan, np.nan, np.nan,
                    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

        ### =========== Build ML Model ============= ###
        if variable_type == 'sos and climate':
            feature_names = ['SOS', 'Coupling', 'SM', 'VPD', 'Ta', 'Pre', 'Srad']
        elif variable_type == 'only climate':
            feature_names = ['Coupling', 'SM', 'VPD', 'Ta', 'Pre', 'Srad']

        importances_df = None
        first_varname = second_varname = third_varname = 'nan'
        first_imp = second_imp = third_imp = np.nan
        sos_imp = cor_imp = sm_imp = vpd_imp = ta_imp = pre_imp = srad_imp = np.nan

        ### Split train and test sets
        # val_mask = mask[:, rel_i, rel_j]
        if grid_size == '3*3':
            min_pixels = 6
        elif grid_size == '5*5':
            min_pixels = 16
        else:
            min_pixels = 66

        if len(all_samples_list) >= min_pixels:

            if the_most_important_var_method == 'RF':

                ## Train using all data, then compute internal OOB R2
                train_data = np.concatenate(all_samples_list, axis=1)

            elif the_most_important_var_method == 'XGBoost':

                ## Training set: Take 90% of data from all, ensuring central pixel is included
                ## Test set: Take 10% of data from all, ensuring central pixel is excluded

                n_samples = len(all_samples_list)

                # All non-central pixel indices
                candidate_idx = [k for k in range(n_samples) if k != center_index]

                random.seed(42)
                random.shuffle(candidate_idx)  # Randomly shuffle indices

                n_test = max(1, int(0.1 * n_samples))  #

                test_idx = candidate_idx[:n_test]

                train_idx = [k for k in range(n_samples) if k not in test_idx]

                train_data = np.concatenate(
                    [all_samples_list[k] for k in train_idx],
                    axis=1
                )

                test_data = np.concatenate(
                    [all_samples_list[k] for k in test_idx],
                    axis=1
                )
                X_test = test_data[1:, :].T
                pos_test = test_data[0, :]

            X_train, pos_train = train_data[1:, :].T, train_data[0, :]

            ### Explain feature importances of model
            # def calculate_shap(model, x_train, y_train, x_test, y_test):
            def calculate_shap(model, x_train, y_train):
                # # Step 1: Calculate MSE and R2 for train and test data
                # mse_train = mean_squared_error(y_train, model.predict(x_train))
                # r2_train = model.score(x_train, y_train)
                # print(f'Train R2: {r2_train}, MSE: {mse_train}')
                #
                # y_train_pred = model.predict(x_test)
                # r2_test = r2_score(y_test, y_train_pred)
                # mse_test = mean_squared_error(y_test, y_train_pred)
                # print(f'Test R2: {r2_test}, MSE: {mse_test}')

                # Step 2: Calculate SHAP values
                explainer = shap.TreeExplainer(model)
                # print("SHAP explainer done")
                shap_values = explainer(x_train)
                # print("SHAP values calculation done")

                # Step 3: Consolidate feature importance and metrics
                df_model = pd.DataFrame(model.feature_importances_, index=feature_names,
                                        columns=["importance"]).reset_index().rename(
                    columns={'index': 'var'})
                # print("Feature importances calculated")
                # print(df_model)

                # Step 4: Calculate mean absolute SHAP values
                mean_abs_shap_values = abs(shap_values.values).mean(0)
                # print(f'Mean absolute SHAP values: {mean_abs_shap_values}')

                df_shap = pd.DataFrame([feature_names, mean_abs_shap_values], index=["var", "shapvalue"]).T
                # print("SHAP values calculated")

                pos_range = y_train.max() - y_train.min()
                shap_sum = shap_values.values.sum(axis=1)  # sum of SHAP per pixel

                # print("POS_mean spatial range (days):", pos_range)
                # print("SHAP sum range (days):", shap_sum.max() - shap_sum.min())

                # Step 5: Merge SHAP values and feature importance
                df_merge = pd.merge(df_shap, df_model, on="var")
                # df_merge = df_merge.rename(columns={'var': 'feature'})

                # print('df_merge:\n', df_merge)

                return df_merge

            ##### RF ####
            if the_most_important_var_method == 'RF':
                # val_mask = mask[:, rel_i, rel_j]
                # if len(neighbor_samples_list) >= 5 and np.sum(val_mask) > 0:
                #     # train_data_all = np.concatenate(all_samples_list, axis=1)
                #     train_data_all = np.concatenate(neighbor_samples_list, axis=1)
                #     X_train, pos_train = train_data_all[1:, :].T, train_data_all[0, :]
                #
                #     val_data = all_vars_std[:, val_mask, rel_i, rel_j]
                #     X_val, pos_val = val_data[1:, :].T, val_data[0, :]

                rf_model = RandomForestRegressor(n_estimators=100, max_features=0.3,
                                                 bootstrap=True, oob_score=True, random_state=42)
                rf_model.fit(X_train, pos_train)

                # test_r2 = rf_model.score(X_test, pos_test)

                if rf_model.oob_score_ >= 0.2:  ### Xtrain comes from all_samples_list
                # if test_r2 >= 0.2:
                    # ### Model-predicted feature importance
                    # importances_df = pd.DataFrame({'var': feature_names, 'importance': rf_model.feature_importances_})

                    # ### Model-explained feature importance
                    importances_df = calculate_shap(rf_model, X_train, pos_train)
            ##### XGBoost ####
            elif the_most_important_var_method == 'XGBoost':
                # val_mask = mask[:, rel_i, rel_j]
                # # Must satisfy: Enough neighbors for training AND central pixel has data for validation
                # if len(neighbor_samples_list) >= 5 and np.sum(val_mask) > 0:
                #
                #     train_data_all = np.concatenate(neighbor_samples_list, axis=1)
                #     X_train, pos_train = train_data_all[1:, :].T, train_data_all[0, :]
                #
                #     val_data = all_vars_std[:, val_mask, rel_i, rel_j]
                #     X_val, pos_val = val_data[1:, :].T, val_data[0, :]

                xgb_model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
                xgb_model.fit(X_train, pos_train)

                test_r2 = xgb_model.score(X_test, pos_test)
                if test_r2 >= 0.2:
                    # ### Model-predicted feature importance
                    # importances_df = pd.DataFrame({'var': feature_names, 'importance': xgb_model.feature_importances_})

                    # ### Model-explained feature importance
                    importances_df = calculate_shap(xgb_model, X_train, pos_train)

            if importances_df is not None:
                # Sort and deduplicate: sort by absolute value of correlation coefficient in descending order
                # top_vars = importances_df.sort_values(by='importance', ascending=False)
                top_vars = importances_df.sort_values(by='shapvalue', ascending=False)
                # print('top_vars:', top_vars)

                first_varname = top_vars.iloc[0]['var']
                second_varname = top_vars.iloc[1]['var']
                third_varname = top_vars.iloc[2]['var']
                # print(f'top var:{first_varname}\n'
                #       f'secong var:{second_varname}')

                first_imp = top_vars.iloc[0]['shapvalue']
                second_imp = top_vars.iloc[1]['shapvalue']
                third_imp = top_vars.iloc[2]['shapvalue']

                if variable_type == 'sos and climate':
                    sos_imp = importances_df.loc[importances_df['var'] == 'SOS', 'shapvalue'].iloc[0]
                elif variable_type == 'only climate':
                    sos_imp = np.nan
                cor_imp = importances_df.loc[importances_df['var'] == 'Coupling', 'shapvalue'].iloc[0]
                sm_imp = importances_df.loc[importances_df['var'] == 'SM', 'shapvalue'].iloc[0]
                vpd_imp = importances_df.loc[importances_df['var'] == 'VPD', 'shapvalue'].iloc[0]
                ta_imp = importances_df.loc[importances_df['var'] == 'Ta', 'shapvalue'].iloc[0]
                pre_imp = importances_df.loc[importances_df['var'] == 'Pre', 'shapvalue'].iloc[0]
                srad_imp = importances_df.loc[importances_df['var'] == 'Srad', 'shapvalue'].iloc[0]

            # else:
            #     first_varname = second_varname = third_varname = 'nan'
            #     first_imp = second_imp = third_imp = np.nan
            #     sos_imp = cor_imp = sm_imp = vpd_imp = ta_imp = pre_imp = srad_imp = np.nan

        else:
            first_varname = second_varname = third_varname = 'nan'
            first_imp = second_imp = third_imp = np.nan
            sos_imp = cor_imp = sm_imp = vpd_imp = ta_imp = pre_imp = srad_imp = np.nan

    else:
        first_varname = second_varname = third_varname = 'nan'
        first_imp = second_imp = third_imp = np.nan
        sos_imp = cor_imp = sm_imp = vpd_imp = ta_imp = pre_imp = srad_imp = np.nan


    return (i, j, first_varname, second_varname, third_varname,
            first_imp, second_imp, third_imp,
            sos_imp, cor_imp, sm_imp, vpd_imp, ta_imp, pre_imp, srad_imp)


###################################### 1 Input and Output Settings ################################################
###########################  ==== 1.1 Input Settings ==== #################################
start_year = 2001
end_year = 2024
years_length = end_year - start_year + 1
print('years_length:', years_length)

pheno = 'pos'  ##### If POS then 'pos', if Mat-Sen then 'length of pos'

Outlier = 'Yes'  # Yes / No
OutnosigCor = 'No'  # Yes / No

Basedon = 'Based_on_detrendPheno'  ### Based_on_detrendPheno means using detrended SOS and POS for partial correlation
                                   ### Based_on_OriginPheno means using original SOS and POS for partial correlation, etc.

the_most_important_var_method = 'Partial'  ### RF / XGBoost / Partial

## ! Note: If using XGBoost or RF, select accordingly
grid_size = '3*3' # '3*3' / '5*5' / '10*10'

variable_type = 'sos and climate'  ### sos and climate / only climate

input_same_path = rf'D:\FigShare_data\55km'

folder_cor = fr'{input_same_path}\Climate_data\climate_in_best_preseason\Correlation(SM_VPD_pearson)'  #(POS-30) - POS

folder_cor_pvalue = fr'{input_same_path}\Climate_data\climate_in_best_preseason\Correlation(SM_VPD_pearson)\Pvalue'  #(POS-30) - POS

folder_sm = fr'{input_same_path}\Climate_data\climate_in_best_preseason\SM_preseason_mean'  #(POS-30) - POS

folder_vpd = fr'{input_same_path}\Climate_data\climate_in_best_preseason\VPD_preseason_mean'  #(POS-30) - POS

folder_ta = fr'{input_same_path}\Climate_data\climate_in_best_preseason\Ta_preseason_mean'

folder_pre = fr'{input_same_path}\Climate_data\climate_in_best_preseason\Pre_preseason_sum'

folder_srad= fr'{input_same_path}\Climate_data\climate_in_best_preseason\Srad_preseason_sum'

#### input POS tif path
if Basedon == 'Based_on_detrendPheno':
    pos_folder = fr'{input_same_path}\POSdetrend_55km'  #start
    sos_folder = fr'{input_same_path}\SOSdetrend_55km'  #start
elif Basedon == 'Based_on_OriginPheno':
    pos_folder = fr'{input_same_path}\POS_55km'  # start
    sos_folder = fr'{input_same_path}\SOS_55km'  # start

#### input SPEI identify drought event tif path
drought_path = rf'{input_same_path}\drought_event(POS_SPEI3_threshold10%_way3)'

#### input AI tif path
ai_tif_file = rf'{input_same_path}\AI\NH30_84_AI(graident)_{scale}km.tif'

#### input vegetation type tif path
veg_type_file = rf'{input_same_path}\Veg_type\NH_veg_type_{scale}km(Python).tif'

#### input coupling gradient tif path
cor_mean_file = fr'{input_same_path}\mean\SM_VPD_Cor17_8_0\Cor_mean_{scale}km_All.tif'  #SOS - POS


###########################  ==== 1.2 Setting of output ==== #################################
output_partial_max_var_png_path = fr'D:\Result'


####################################### 2 Read data  #################################################
tif_files_cor = sorted(glob.glob(os.path.join(folder_cor, '*.tif')))

tif_files_cor_pvalue = sorted(glob.glob(os.path.join(folder_cor_pvalue, '*.tif')))

tif_files_sm = sorted(glob.glob(os.path.join(folder_sm, '*.tif')))

tif_files_vpd = sorted(glob.glob(os.path.join(folder_vpd, '*.tif')))

tif_files_ta = sorted(glob.glob(os.path.join(folder_ta, '*.tif')))

tif_files_pre = sorted(glob.glob(os.path.join(folder_pre, '*.tif')))

tif_files_srad = sorted(glob.glob(os.path.join(folder_srad, '*.tif')))

pos_tif_files = sorted(glob.glob(os.path.join(pos_folder, '*.tif')))
sos_tif_files = sorted(glob.glob(os.path.join(sos_folder, '*.tif')))

drought_year_tif_files = sorted(glob.glob(os.path.join(drought_path, '*.tif')))

if not tif_files_sm_1:
    raise FileNotFoundError("No tif_files_sm_1 TIF files found!")
if not tif_files_sm_2:
    raise FileNotFoundError("No tif_files2 TIF files found!")
if not tif_files_sm_3:
    raise FileNotFoundError("No tif_files3 TIF files found!")
if not pos_tif_files:
    raise FileNotFoundError("No POS TIF files found!")


###################################################
first_tif = tif_files_sm[0]
sample_tif = gdal.Open(first_tif)

if sample_tif is None:
    raise RuntimeError(f"Failed to open TIF file: {sample_tif} (unsupported driver or corrupted file)")

# Get geotransform parameters: projection, pixel size
# Coordinates and projection | Coordinate Reference System: Spatial reference framework for the data
crs = sample_tif.GetProjectionRef()          # Automatically retrieve input CRS
# print('crs:', crs)
gt = sample_tif.GetGeoTransform()  # Geographic coordinates: Longitude/Latitude. Mathematical transformation parameters to map pixel coordinates to real-world geographic coordinates.
proj = sample_tif.GetProjection()  # Projected coordinates: XY (unit: meters)

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

# Calculate longitude and latitude range (correcting for negative pixel_height)
lon_min = top_left_x
lon_max = top_left_x + cols * pixel_width  # Right boundary longitude
lat_min = top_left_y + rows * pixel_height  # Bottom boundary latitude (southernmost point, potentially smaller)
lat_max = top_left_y  # Top boundary latitude (northernmost point, potentially larger)
print(f"Longitude range: {lon_min:.6f} -> {lon_max:.6f}")
print(f"Latitude range: {lat_min:.6f} -> {lat_max:.6f}")


############################################ 3 Stacking ###################################################
## Data stacking
cor_stack = []

cor_pvalue_stack = []

sm_stack = []

vpd_stack = []

ta_stack = []

pre_stack = []

srad_stack = []

pos_stack = []
sos_stack = []

drought_year_stack = []

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

for tif_file in tif_files_srad_1:
    get_band(tif_file, srad_stack_1)

for tif_file in pos_tif_files:
    get_band(tif_file, pos_stack)
for tif_file in sos_tif_files:
    get_band(tif_file, sos_stack)

for tif_file in drought_year_tif_files:
    get_band(tif_file, drought_year_stack)

ai_tif = gdal.Open(ai_tif_file)
ai_data = ai_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
print('AI shape:', ai_data.shape)


veg_type_tif = gdal.Open(veg_type_file)
veg_type_data = veg_type_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
# veg_type_data = veg_type_data[1:-1, 1:-1]
print('veg_type_data shape:', veg_type_data.shape)

cor_mean_tif = gdal.Open(cor_mean_file)
cor_mean_data = cor_mean_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)

print('Stack start')
cor_pre_stack = np.stack(cor_stack, axis=0)#[:, 505:510, 505:510]

cor_pvalue_pre_stack = np.stack(cor_pvalue_stack, axis=0)#[:, 505:510, 505:510]

sm_pre_stack = np.stack(sm_stack, axis=0)

vpd_pre_stack = np.stack(vpd_stack, axis=0)

ta_pre_stack = np.stack(ta_stack, axis=0)

pre_pre_stack = np.stack(pre_stack, axis=0)

srad_pre_stack = np.stack(srad_stack, axis=0)

pos_stack = np.stack(pos_stack, axis=0)

sos_stack = np.stack(sos_stack, axis=0)

drought_year_stack = np.stack(drought_year_stack, axis=0)

print('Stack end!')
print('Preseason match done!')

########################### 5 Outlier Removal (/ Remove Non-Significant Cor Pixels) ################################
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
        if len(x_flatten) < analyze_years_length/2:
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

            if (len(np.isfinite(x_masked)) > (analyze_years_length/2)) & (len(np.isfinite(x_masked)) <= analyze_years_length):
                return x_masked, i, j  # IQR, lower_range, upper_range,  minv, maxv
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

    ### IQR Outlier
    results = Parallel(n_jobs=15, verbose=10)(
        delayed(Outlier_array_IQR)(
            pos_stack[:years_length, i, j], i, j, 25, 75
        ) for i, j in zip(row_indices, col_indices)
    )

    for data_mask, i, j in results:
        outlier_pos_stack[:, i, j] = data_mask

    results = Parallel(n_jobs=15, verbose=10)(
        delayed(Outlier_array_IQR)(
            sos_stack[:years_length, i, j], i, j, 25, 75
        ) for i, j in zip(row_indices, col_indices)
    )

    for data_mask, i, j in results:
        outlier_sos_stack[:, i, j] = data_mask

    results = Parallel(n_jobs=15, verbose=10)(
        delayed(Outlier_array_IQR)(
            cor_stack[:years_length, i, j], i, j, 25, 75
        ) for i, j in zip(row_indices, col_indices)
    )

    for data_mask, i, j in results:
        outlier_cor_stack[:, i, j] = data_mask

    results = Parallel(n_jobs=15, verbose=10)(
        delayed(Outlier_array_IQR)(
            sm_stack[:years_length, i, j], i, j, 25, 75
        ) for i, j in zip(row_indices, col_indices)
    )

    for data_mask, i, j in results:
        outlier_sm_stack[:, i, j] = data_mask

    results = Parallel(n_jobs=15, verbose=10)(
        delayed(Outlier_array_IQR)(
            vpd_stack[:years_length, i, j], i, j, 25, 75
        ) for i, j in zip(row_indices, col_indices)
    )

    for data_mask, i, j in results:
        outlier_vpd_stack[:, i, j] = data_mask

    results = Parallel(n_jobs=15, verbose=10)(
        delayed(Outlier_array_IQR)(
            ta_stack[:years_length, i, j], i, j, 25, 75
        ) for i, j in zip(row_indices, col_indices)
    )

    for data_mask, i, j in results:
        outlier_ta_stack[:, i, j] = data_mask

    results = Parallel(n_jobs=15, verbose=10)(
        delayed(Outlier_array_IQR)(
            pre_stack[:years_length, i, j], i, j, 25, 75
        ) for i, j in zip(row_indices, col_indices)
    )

    for data_mask, i, j in results:
        outlier_pre_stack[:, i, j] = data_mask

    results = Parallel(n_jobs=15, verbose=10)(
        delayed(Outlier_array_IQR)(
            srad_stack[:years_length, i, j], i, j, 25, 75
        ) for i, j in zip(row_indices, col_indices)
    )

    for data_mask, i, j in results:
        outlier_srad_stack[:, i, j] = data_mask

    mask = (np.isfinite(outlier_pos_stack) & np.isfinite(outlier_sos_stack) &
            np.isfinite(outlier_cor_stack) & np.isfinite(outlier_sm_stack) & np.isfinite(outlier_vpd_stack)
            & np.isfinite(outlier_ta_stack) & np.isfinite(outlier_pre_stack) & np.isfinite(outlier_srad_stack))

elif Outlier == 'No':

    mask = (np.isfinite(pos_stack) &
            np.isfinite(sos_stack) &
            np.isfinite(cor_pre_stack) &
            np.isfinite(sm_pre_stack) &
            np.isfinite(vpd_pre_stack) &
            np.isfinite(ta_pre_stack) &
            np.isfinite(pre_pre_stack) &
            np.isfinite(srad_pre_stack))


if OutnosigCor == 'Yes':
    mask = (cor_pvalue_pre_stack <= 0.1) & mask
vaild_year_length = np.sum(mask, axis=0)

space_mask = vaild_year_length > (years_length/2)  # shape：(rows, cols)

print(f'Original valid pixel count: {np.count_nonzero(pos_stack[0, :, :])}\n'
      f'Valid pixel count after outlier removal: {np.count_nonzero(space_mask)}')

space_mask_3d = space_mask[np.newaxis, :, :]  # (1, rows, cols)
space_mask_3d = np.repeat(space_mask_3d, years_length, axis=0)  # (years_length, rows, cols)



########################### 6 Preseason Pearson+Partial ################

def standardize_data(data_stack, mask_3d):
    """Time standardize"""

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

    standardized_data = np.where(mask_3d, standardized_data, np.nan)

    return standardized_data


dtype = [('var', '<U10')]
varname_first = np.empty((rows, cols), dtype=dtype)
varname_second = np.empty((rows, cols), dtype=dtype)
varname_third = np.empty((rows, cols), dtype=dtype)

if the_most_important_var_method == 'Partial':

    ########################### 6 Temporal Pixel-Wise Normalization of Explanatory Variables ###################
    if Outlier == 'Yes':
        pos_standardized = standardize_data(outlier_pos_stack, space_mask_3d)
        sos_standardized = standardize_data(outlier_sos_stack, space_mask_3d)
        cor_standardized = standardize_data(outlier_cor_stack, space_mask_3d)
        sm_standardized = standardize_data(outlier_sm_stack, space_mask_3d)
        vpd_standardized = standardize_data(outlier_vpd_stack, space_mask_3d)
        ta_standardized = standardize_data(outlier_ta_stack, space_mask_3d)
        pre_standardized = standardize_data(outlier_pre_stack, space_mask_3d)
        srad_standardized = standardize_data(outlier_srad_stack, space_mask_3d)

    elif Outlier == 'No':
        pos_standardized = standardize_data(pos_stack, space_mask_3d)
        sos_standardized = standardize_data(sos_stack, space_mask_3d)
        cor_standardized = standardize_data(cor_pre_stack, space_mask_3d)
        sm_standardized = standardize_data(sm_pre_stack, space_mask_3d)
        vpd_standardized = standardize_data(vpd_pre_stack, space_mask_3d)
        ta_standardized = standardize_data(ta_pre_stack, space_mask_3d)
        pre_standardized = standardize_data(pre_pre_stack, space_mask_3d)
        srad_standardized = standardize_data(srad_pre_stack, space_mask_3d)

    del sos_stack, cor_stack, sm_stack, vpd_stack, ta_stack, pre_stack, srad_stack
    gc.collect()

    print('Partial correlation calculation start!')

    if variable_type == 'sos and climate':
        var_list = ['POS', 'SOS', 'Ta', 'Pre', 'Srad', 'SM', 'VPD', 'Coupling']
    elif variable_type == 'only climate':
        var_list = ['POS', 'Ta', 'Pre', 'Srad', 'SM', 'VPD', 'Coupling']

    p_coefficient_r_first = np.full((rows, cols), np.nan)
    p_coefficient_r_second = np.full((rows, cols), np.nan)
    p_coefficient_r_third = np.full((rows, cols), np.nan)

    p_coefficient_p_first = np.full((rows, cols), np.nan)
    p_coefficient_p_second = np.full((rows, cols), np.nan)
    p_coefficient_p_third = np.full((rows, cols), np.nan)

    sos_pcor_matrix = np.full((rows, cols), np.nan)
    sm_pcor_matrix = np.full((rows, cols), np.nan)
    vpd_pcor_matrix = np.full((rows, cols), np.nan)
    cor_pcor_matrix = np.full((rows, cols), np.nan)
    ta_pcor_matrix = np.full((rows, cols), np.nan)
    pre_pcor_matrix = np.full((rows, cols), np.nan)
    srad_pcor_matrix = np.full((rows, cols), np.nan)

    # with parallel_backend('threading', n_jobs=15):
    results = Parallel(n_jobs=15, verbose=10)(
            delayed(cal_partial)(
                i, j, var_list[1:], var_list[0], pos_standardized,  ## Unstandardized POS: outlier_pos_stack  |  Standardized POS: pos_standardized
                                                sos_standardized,
                                                cor_standardized,
                                                sm_standardized,
                                                vpd_standardized,
                                                ta_standardized,
                                                pre_standardized,
                                                srad_standardized
            )
            for i, j in zip(row_indices, col_indices)
        )

    for (i, j, first_varname, second_varname, third_varname,
         first_r, second_r, third_r,
         first_p, second_p, third_p,
         sos_pcor, sm_pcor, vpd_pcor, cor_pcor, ta_pcor, pre_pcor, srad_pcor) in results:
        varname_first[i, j] = first_varname
        varname_second[i, j] = second_varname
        varname_third[i, j] = third_varname

        p_coefficient_r_first[i, j] = first_r
        p_coefficient_r_second[i, j] = second_r
        p_coefficient_r_third[i, j] = third_r

        p_coefficient_p_first[i, j] = first_p
        p_coefficient_p_second[i, j] = second_p
        p_coefficient_p_third[i, j] = third_p

        sos_pcor_matrix[i, j] = sos_pcor
        sm_pcor_matrix[i, j] = sm_pcor
        vpd_pcor_matrix[i, j] = vpd_pcor
        cor_pcor_matrix[i, j] = cor_pcor
        ta_pcor_matrix[i, j] = ta_pcor
        pre_pcor_matrix[i, j] = pre_pcor
        srad_pcor_matrix[i, j] = srad_pcor


elif the_most_important_var_method in ['RF', 'XGBoost']:

    print(f'{the_most_important_var_method} calculation start!')

    ML_imp_first = np.full((rows, cols), np.nan)
    ML_imp_second = np.full((rows, cols), np.nan)
    ML_imp_third = np.full((rows, cols), np.nan)

    sos_imp_matrix = np.full((rows, cols), np.nan)
    sm_imp_matrix = np.full((rows, cols), np.nan)
    vpd_imp_matrix = np.full((rows, cols), np.nan)
    cor_imp_matrix = np.full((rows, cols), np.nan)
    ta_imp_matrix = np.full((rows, cols), np.nan)
    pre_imp_matrix = np.full((rows, cols), np.nan)
    srad_imp_matrix = np.full((rows, cols), np.nan)

    if Outlier == 'Yes':
        results = Parallel(n_jobs=15, verbose=10)(
                delayed(cal_ML)(
                    i, j, outlier_pos_stack,  ##POS不标准化：outlier_pos_stack  POS标准化：pos_standardized
                        outlier_sos_stack,
                        outlier_cor_stack,
                        outlier_sm_stack,
                        outlier_vpd_stack,
                        outlier_ta_stack,
                        outlier_pre_stack,
                        outlier_srad_stack
                )
                for i, j in zip(row_indices, col_indices)
            )
    elif Outlier == 'No':
        results = Parallel(n_jobs=15, verbose=10)(
            delayed(cal_ML)(
                i, j, pos_stack,  ##POS不标准化：outlier_pos_stack  POS标准化：pos_standardized
                sos_stack,
                cor_pre_stack,
                sm_pre_stack,
                vpd_pre_stack,
                ta_pre_stack,
                pre_pre_stack,
                srad_pre_stack
            )
            for i, j in zip(row_indices, col_indices)
        )

    for (i, j, first_varname, second_varname, third_varname,
         first_imp, second_imp, third_imp,
         sos_imp, cor_imp, sm_imp, vpd_imp, ta_imp, pre_imp, srad_imp) in results:

        varname_first[i, j] = first_varname
        varname_second[i, j] = second_varname
        varname_third[i, j] = third_varname

        ML_imp_first[i, j] = first_imp
        ML_imp_second[i, j] = second_imp
        ML_imp_third[i, j] = third_imp

        sos_imp_matrix[i, j] = sos_imp
        cor_imp_matrix[i, j] = cor_imp
        sm_imp_matrix[i, j] = sm_imp
        vpd_imp_matrix[i, j] = vpd_imp
        ta_imp_matrix[i, j] = ta_imp
        pre_imp_matrix[i, j] = pre_imp
        srad_imp_matrix[i, j] = srad_imp

print('calculation end!')



############################# Coupling Effect Changes Plot ##############################
######## ======= Unclassified / Combined Plot =============== ###########
# def plot_NH_All(varname_data, r_data, grade_by, veg_type, plotdata_type, importance_grape):
def plot_NH_All(varname_data, importance_grape, grade_by, ax):

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from mpl_toolkits.basemap import Basemap
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib.gridspec as gridspec
    import numpy as np
    import os
    from brokenaxes import brokenaxes
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # Create subplot layout
    fig = ax.get_figure()
    gs_inner = ax.get_subplotspec().subgridspec(2, 1, height_ratios=[5, 0.3], hspace=0.15)
    ax1 = fig.add_subplot(gs_inner[0])  # Map
    ax2 = fig.add_subplot(gs_inner[1])  # Colorbar

    ax.axis('off')

    plots = []  # Store plot objects for each subplot

    ########### 1 Spatial Distribution #################
    ax1.set_box_aspect(1)  # Force map axis aspect ratio to square so its diameter fills the cell height

    ### 1.1 Create Map
    m = Basemap(ax=ax1,
                projection='npstere',  # North Polar Stereographic projection
                boundinglat=30,  # Minimum displayed latitude (currently 30°N)
                lon_0=0,  # Center longitude (modifiable) 180: Pacific-centered; 90: Asia-centered
                resolution='l')

    # Generate grid coordinates
    lons = np.linspace(-180, 180, cols, endpoint=False)
    lats = np.linspace(lat_max, lat_min, rows)
    lons, lats = np.meshgrid(lons, lats)

    # Set latitude and longitude parallels/meridians
    m.drawparallels(np.arange(30, 90, 30),
                    labels=[0, 0, 0, 0],
                    linewidth=0.5,
                    # fontsize=8,
                    color="black")

    m.drawmeridians(np.arange(0, 360, 60),
                    latmax=90,  # Ensure longitude lines converge at the North Pole
                    labels=[0, 0, 0, 0],  # labels=[left, right, top, bottom] controls longitude label visibility
                    linewidth=0.5,
                    # fontsize=8,
                    color="gray")

    # # Fill landmasses
    # m.fillcontinents(color='white', lake_color='white', zorder=1)

    # map_boundary = m.drawmapboundary(linewidth=0)  # Hide boundary line

    ### 1.2 Plot Data
    # Color mapping
    # category_colors = {
    #     'SOS': '#ff7a00',
    #     'Ta': '#f2b701',
    #     'Srad': '#a5aa99',
    #     'Pre': '#e73f74',
    #     'SM': '#11a579',
    #     'VPD': '#3969ac',
    #     'Cor': '#7f3c8d'
    # }
    # category_colors = {
    #     'SOS': '#33a02c',  # Green
    #     'Ta': '#ff7f00',   # Orange
    #     'Srad': '#a5aa99', # Gray
    #     'Pre': '#1f78b4',  # Blue
    #     'SM': '#6a3d9a',   # Purple
    #     'VPD': '#ffff99',  # Yellow
    #     'Cor': '#e31a1c'   # Red
    # }
    category_colors = {
        'SOS': '#b2df8a',  # Green
        'Ta': '#fdbf6f',   # Orange
        'Srad': '#a5aa99', # Gray
        'Pre': '#a6cee3',  # Blue
        'SM': '#cab2d6',   # Purple
        'VPD': '#ffff99',  # Yellow
        'Coupling': '#e31a1c'   # Red
    }

    # Create mapping dictionary
    keys = list(category_colors.keys())
    str_to_int = {k: i for i, k in enumerate(keys)}

    # Vectorized conversion (handling NaNs)
    def safe_map(x):
        if isinstance(x, np.ma.core.MaskedConstant) or str(x) == 'nan':
            return -1
        return str_to_int.get(x, -1)  # Return -1 if key does not exist

    # Convert to integer matrix
    int_data = np.vectorize(safe_map)(varname_data)
    int_data = np.ma.masked_where(int_data == -1, int_data)  # Mask invalid values

    # Create colormap
    cmap = ListedColormap(list(category_colors.values()))
    norm = BoundaryNorm(np.arange(len(category_colors) + 1) - 0.5,
                        len(category_colors))

    # Plot spatial distribution map
    plot = m.pcolormesh(lons, lats, int_data, cmap=cmap, norm=norm, shading='nearest', latlon=True,
                        zorder=1)  # Avoid polar tear
    plots.append(plot)  # Save plot object

    # Draw boundaries
    terrence = r'D:\CAU\Border\NH_terrence\NH_Terrence_30_90'
    # m.readshapefile(terrence, 'NH_Terrence', drawbounds=True, linewidth=0.2, color='black', zorder=3)
    m.readshapefile(terrence, 'NH_Terrence', drawbounds=False)
    for shape in m.NH_Terrence:
        # Convert list to numpy array for easier calculation
        points = np.array(shape)
        x, y = points[:, 0], points[:, 1]

        # Core logic: Calculate projection distance between adjacent points
        # If the distance between two adjacent points in the projection plane suddenly becomes very large,
        # it indicates a loop line crossing the map center.
        dist = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)

        # Set a threshold (projection coordinates are typically large, e.g., ~100,000 scale)
        # If adjacent point distance exceeds 1/10 of the map diameter, identify as an abnormal jump
        threshold = (ax1.get_xlim()[1] - ax1.get_xlim()[0]) * 0.1

        # Find indices of jump points
        break_indices = np.where(dist > threshold)[0]

        if len(break_indices) == 0:
            # No jumps; draw the entire line directly
            ax1.plot(x, y, color='black', linewidth=0.3, zorder=3)
        else:
            # Jumps detected; split line into segments and draw individually
            # This removes artifact lines crossing the map center while keeping valid boundaries intact
            start_idx = 0
            for break_idx in break_indices:
                ax1.plot(x[start_idx:break_idx + 1], y[start_idx:break_idx + 1],
                         color='black', linewidth=0.3, zorder=3)
                start_idx = break_idx + 1
            # Draw the final segment
            ax1.plot(x[start_idx:], y[start_idx:], color='black', linewidth=0.3, zorder=3)

    # # North Pole point
    # x_pole, y_pole = m(0, 90)
    # ax1.scatter(x_pole, y_pole, color='black', s=5, zorder=3)

    ### 1.4 Outer Boundary Clipping
    from matplotlib.patches import Circle

    x0, x1 = ax1.get_xlim()
    y0, y1 = ax1.get_ylim()

    center = [(x0 + x1) / 2, (y0 + y1) / 2]
    # radius = (x1 - x0) / 2
    radius = min(x1 - x0, y1 - y0) / 2

    clip_circle = Circle(center, radius, transform=ax1.transData)

    for artist in ax1.collections + ax1.lines + ax1.patches:
        artist.set_clip_path(clip_circle)

    boundary_circle = Circle(
        center,
        radius,
        transform=ax1.transData,
        facecolor='none',
        edgecolor='black',  # Color
        linewidth=0.8,
        clip_on=False,  # Controls whether this object is clipped by the current Axes rectangular boundary
        zorder=4  # Place on topmost layer
    )

    ax1.add_patch(boundary_circle)

    ax1.axis('off')


    ##################### 2 Statistical Plot #############################
    ### 2.1 Statistics
    counts = [np.sum(varname_data == k) for k in keys]
    total = np.sum([np.sum(varname_data == k) for k in keys if str(k) != 'nan'])
    print(f'counts:\n'
          f'{counts}')
    print(f'total={total}')

    fractions = [c / total * 100 for c in counts]

    ### 2.2 Plotting
    bin_centers = ['SOS', 'Ta', 'Srad', 'Pre', 'SM', 'VPD', 'Coupling']

    if importance_grape == 'first':
        # Create inset container on ax1
        inset_pos = [0.01, 0, 0.3, 0.3]
        axins = ax1.inset_axes(inset_pos)

        axins.set_facecolor('white')
        axins.patch.set_alpha(1)

        # Hide ticks while retaining background
        axins.set_xticks([])
        axins.set_yticks([])

        # Hide borders
        for spine in axins.spines.values():
            spine.set_visible(False)

        # Create upper and lower sub-axes (simulating broken axis)
        axins_top = axins.inset_axes([0, 0.65, 1, 0.3])  # Top portion
        axins_bottom = axins.inset_axes([0, 0, 1, 0.6])  # Bottom portion

        # Draw bar chart
        x_indexes = np.arange(len(bin_centers))  # Generate [0, 1, 2, 3, 4, 5, 6]
        plot_colors = [category_colors[k] for k in bin_centers]

        axins_top.bar(x_indexes, fractions, color=plot_colors, width=0.7)
        axins_bottom.bar(x_indexes, fractions, color=plot_colors, width=0.7)

        # Set broken axis limits
        total_max = max(arr.max() for arr in fractions)
        print(f"Absolute maximum value is: {total_max}")
        if OutnosigCor == 'No':
            if the_most_important_var_method == 'Partial':
                axins_top.set_ylim(40, 45)
            elif the_most_important_var_method == 'RF':
                axins_top.set_ylim(60, 65)
        if OutnosigCor == 'Yes':
            axins_top.set_ylim(45, 50)

        if the_most_important_var_method == 'Partial':
            axins_bottom.set_ylim(0, 25.001)
            axins_bottom.set_yticks(np.arange(0, 25, 10))
        elif the_most_important_var_method == 'RF':
            axins_bottom.set_ylim(0, 30.001)
            axins_bottom.set_yticks(np.arange(0, 30, 10))

        # Beautification (hide inner borders)
        axins_top.spines['bottom'].set_visible(False)
        axins_top.spines['top'].set_visible(False)
        axins_top.spines['right'].set_visible(False)

        axins_bottom.spines['top'].set_visible(False)
        axins_bottom.spines['right'].set_visible(False)

        axins_top.tick_params(axis='x', bottom=False, labelbottom=False)  # Top chart hides x-axis
        axins_top.tick_params(axis='y')
        axins_bottom.tick_params(axis='x', bottom=False, labelbottom=False)
        axins_bottom.tick_params(axis='y')

        # Broken axis symbols
        d = 0.02

        # kwargs = dict(transform=axins_top.transAxes, color='k', clip_on=False)
        # axins_top.plot((-d, +d), (-d, +d), **kwargs)
        # # axins_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        #
        # kwargs.update(transform=axins_bottom.transAxes)
        # axins_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        # # axins_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

        bbox_top = axins_top.get_position()
        bbox_bottom = axins_bottom.get_position()

        ratio_top = bbox_top.height / bbox_top.width
        ratio_bottom = bbox_bottom.height / bbox_bottom.width

        axins_top.plot((-d, +d), (-d * ratio_top, +d * ratio_top),
                       transform=axins_top.transAxes, color='k', clip_on=False)

        axins_bottom.plot((-d, +d), (1 - d * ratio_bottom, 1 + d * ratio_bottom),
                          transform=axins_bottom.transAxes, color='k', clip_on=False)

        axins_top.set_ylabel("Frequency (%)")
        axins_top.yaxis.set_label_coords(-0.35, -0.35)

    if importance_grape in ['second', 'third']:
        inset_pos = [0.01, 0, 0.3, 0.3]
        axins = ax1.inset_axes(inset_pos)
        # axins.set_axis_off()  # Ensure container axis itself is invisible

        # Draw bar chart
        x_indexes = np.arange(len(bin_centers))  # Generate [0, 1, 2, 3, 4, 5, 6]
        plot_colors = [category_colors[k] for k in bin_centers]

        axins.bar(x_indexes, fractions, color=plot_colors, width=0.7)

        if importance_grape == 'second':
            if the_most_important_var_method == 'Partial':

                axins.set_ylim(0, 25)

                axins.set_yticks(np.arange(0, 25.01, 5))
            elif the_most_important_var_method == 'RF':
                axins.set_ylim(0, 40)

                axins.set_yticks(np.arange(0, 40.01, 10))

        elif importance_grape == 'third':
            axins.set_ylim(0, 25)

            axins.set_yticks(np.arange(0, 25.01, 5))

        # Beautification (hide inner borders)
        axins.spines['top'].set_visible(False)
        axins.spines['right'].set_visible(False)

        axins.tick_params(axis='x', bottom=False, labelbottom=False)
        axins.tick_params(axis='y')

        axins.set_ylabel("Frequency (%)")

    # # Set background transparency
    # axins.set_facecolor('white')
    # axins.patch.set_alpha(1.0)  # Ensure complete opacity
    # # axins.set_zorder(5)  # Ensure placement above map layer (pcolormesh)


    ################### 3 Colorbar ######################
    if importance_grape == 'third':
        # Create colorbar
        cbar_ax = ax2

        cbar = fig.colorbar(plots[0], cax=cbar_ax, orientation='horizontal')

        # Set colorbar labels and ticks
        cbar.set_ticks(np.arange(len(category_colors)))
        cbar.set_ticklabels(list(category_colors.keys()), fontsize=9)
        cbar.ax.tick_params(axis='x', which='major')
        cbar.ax.minorticks_off()
    else:
        ax2.set_visible(False)

        # # Manually adjust colorbar position to center it
        # cbar.ax.set_position([0.21, 0.3, 0.6, 0.02])


    # # Display figure
    # plt.tight_layout()

    # output_path1 = os.path.join(output_partial_max_var_png_path, rf"SM_VPD_Cor17\In temporal\Climate{climate_test_number}\{grade_by}\{Basedon}\Preseason\SM_VPD_Cor{cor_test_number}_{analyzed_start_year}-{analyzed_end_year}_Outlier({Outlier})_Outnosig({OutnosigCor})_{the_most_important_var_method}({importance_grape})_varname.png")  # SM_VPD_pearson_
    # plt.savefig(output_path1, dpi=600)
    # print(f'PNG saved to: {output_path1}')
    # plt.show()

    return fractions, bin_centers, fig


def plot_NH_single(cor_first_data, cor_second_data, cor_third_data, varname, ax):
    """
    Displays only the color mapping corresponding to varname.
    varname: Variable name, such as 'SM', 'VPD', 'Pre', etc.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from mpl_toolkits.basemap import Basemap
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import numpy as np
    import matplotlib.colors as mcolors


    # Conversion function
    def safe_map_with_weight(x, weight):
        if isinstance(x, np.ma.core.MaskedConstant) or str(x) == 'nan' or x is None:
            return np.nan
        elif x == varname:
            return weight
        else:
            return 0

    ### Convert three datasets
    int_data1 = np.vectorize(lambda x: safe_map_with_weight(x, 1))(cor_first_data)  # First weight: 1
    int_data2 = np.vectorize(lambda x: safe_map_with_weight(x, 2))(cor_second_data)  # Second weight: 2
    int_data3 = np.vectorize(lambda x: safe_map_with_weight(x, 3))(cor_third_data)  # Third weight: 3

    int_data_origin = int_data1 + int_data2 + int_data3  # Results in values ranging from 1 to 3

    int_data = np.ma.masked_where((int_data_origin == 0) | np.isnan(int_data_origin), int_data_origin)

    ### Color
    # Define all possible color mappings
    all_category_colors = {
        'SOS': '#b2df8a',  # Green
        'Ta': '#fdbf6f',   # Orange
        'Srad': '#a5aa99', # Gray
        'Pre': '#a6cee3',  # Blue
        'SM': '#cab2d6',   # Purple
        'VPD': '#ffff99',  # Yellow
        'Coupling': '#e31a1c' # Red
    }

    # Get RGB values for base color
    base_rgb = np.array(mcolors.to_rgb(all_category_colors[varname]))


    def lighten(color, factor):
        return color + (1 - color) * factor

    def darken(color, factor):
        return color * (1 - factor)

    # colors = [
    #     base_rgb,  # Dark (Original)
    #     lighten(base_rgb, 0.3),  # Medium
    #     lighten(base_rgb, 0.6)   # Light
    # ]
    colors = [
        darken(base_rgb, 0.4),  # Dark
        base_rgb,  # Original
        lighten(base_rgb, 0.4)  # Light
    ]

    # Clip values within 0-1
    colors = [np.clip(c, 0, 1) for c in colors]

    cmap = ListedColormap(colors)
    cmap.set_bad(alpha=0)

    vmin = int_data.min()
    vmax = int_data.max()
    norm = BoundaryNorm(
        np.arange(vmin - 0.5, vmax + 1, 1),
        int(vmax - vmin + 1)
    )

    ### Plot
    # Create subplot layout
    fig = ax.get_figure()
    gs_inner = ax.get_subplotspec().subgridspec(2, 1, height_ratios=[5, 0.3], hspace=0.15)
    ax1 = fig.add_subplot(gs_inner[0])  # Map
    ax2 = fig.add_subplot(gs_inner[1])  # Colorbar

    ax.axis('off')
    ax1.axis('off')

    plots = []

    # Create Map
    m = Basemap(ax=ax1,
                projection='npstere',
                boundinglat=30,
                lon_0=0,
                resolution='l')

    # Generate grid coordinates
    lons = np.linspace(-180, 180, cols, endpoint=False)
    lats = np.linspace(lat_max, lat_min, rows)
    lons, lats = np.meshgrid(lons, lats)

    # Draw parallels and meridians
    m.drawparallels(np.arange(30, 90, 30), labels=[0, 0, 0, 0],
                    linewidth=0.5, color="black")
    m.drawmeridians(np.arange(0, 360, 60), latmax=90,
                    labels=[0, 0, 0, 0], linewidth=0.5, color="gray")

    # Plot three layers of data (transparency overlay)
    plot = m.pcolormesh(lons, lats, int_data, cmap=cmap, norm=norm,
                        alpha=1.0, shading='nearest', latlon=True, zorder=1)

    plots.append(plot)

    # Draw boundaries
    terrence = r'D:\CAU\Border\NH_terrence\NH_Terrence_30_90'
    m.readshapefile(terrence, 'NH_Terrence', drawbounds=False)
    for shape in m.NH_Terrence:
        points = np.array(shape)
        x, y = points[:, 0], points[:, 1]
        dist = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
        threshold = (ax1.get_xlim()[1] - ax1.get_xlim()[0]) * 0.1
        break_indices = np.where(dist > threshold)[0]

        start_idx = 0
        for break_idx in break_indices:
            ax1.plot(x[start_idx:break_idx + 1], y[start_idx:break_idx + 1],
                     color='black', linewidth=0.3, zorder=3)
            start_idx = break_idx + 1
        ax1.plot(x[start_idx:], y[start_idx:], color='black', linewidth=0.3, zorder=3)

    # North Pole point
    # x_pole, y_pole = m(0, 90)
    # ax1.scatter(x_pole, y_pole, color='black', s=5, zorder=3)

    # Circular clipping
    x0, x1 = ax1.get_xlim()
    y0, y1 = ax1.get_ylim()
    center = [(x0 + x1) / 2, (y0 + y1) / 2]
    radius = (x1 - x0) / 2

    clip_circle = Circle(center, radius, transform=ax1.transData)
    for artist in ax1.collections + ax1.lines + ax1.patches:
        artist.set_clip_path(clip_circle)

    boundary_circle = Circle(center, radius,
                             transform=ax1.transData,
                             facecolor='none',
                             edgecolor='black',
                             linewidth=0.8,
                             clip_on=False,  # Controls whether this object is clipped by the current Axes rectangular boundary
                             zorder=4)
    ax1.add_patch(boundary_circle)


    ##################### 2 Statistical Plot #############################
    print(f'np.sum(np.isfinite(int_data_origin)):{np.sum(np.isfinite(int_data_origin))}')
    ### 2.1 Statistics
    counts = [np.sum(int_data == k) for k in range(1,4)]
    total = np.sum(np.isfinite(int_data_origin))
    fractions = [c / total * 100 for c in counts]
    print(f'fractions:{fractions}')

    ### 2.2 Plotting
    bin_centers = ['First', 'Second', 'Third']

    # Create inset container on ax1
    inset_pos = [0.01, 0, 0.3, 0.3]
    axins = ax1.inset_axes(inset_pos)

    axins.set_facecolor('white')
    axins.patch.set_alpha(1)

    # Hide ticks while retaining background
    axins.set_xticks([])
    axins.set_yticks([])

    # # Hide borders
    # for spine in axins.spines.values():
    #     spine.set_visible(False)

    # # Create upper and lower sub-axes (simulating broken axis)
    # axins_top = axins.inset_axes([0, 0.65, 1, 0.2])  # Top portion
    # axins_bottom = axins.inset_axes([0, 0, 1, 0.6])  # Bottom portion

    # Draw bar chart
    x_indexes = np.arange(len(bin_centers))  # Generate [0, 1, 2, 3, 4, 5, 6]
    # bar_colors = [colors[0], colors[1], colors[2]]

    axins.bar(x_indexes, fractions, color=colors, width=0.7)

    # Set broken axis limits
    total_max = max(arr.max() for arr in fractions)
    print(f"Absolute maximum value is: {total_max}")

    # frac_max = max(fractions)
    # y_max = (math.ceil(frac_max / 5) + 1) * 5
    if the_most_important_var_method == 'Partial':
        y_max = 20
    elif the_most_important_var_method == 'RF':
        y_max = 15

    axins.set_ylim(0, y_max + 0.001)
    axins.set_yticks(np.arange(0, y_max + 0.001, 5))
    tick_size = plt.rcParams['xtick.labelsize']
    axins.tick_params(axis='y', labelsize=tick_size)

    # Beautification (hide inner borders)
    axins.spines['top'].set_visible(False)
    axins.spines['right'].set_visible(False)

    axins.tick_params(axis='x', bottom=False, labelbottom=False)
    axins.tick_params(axis='y')

    axins.set_ylabel("Frequency (%)")


    ### Colorbar
    cbar = fig.colorbar(plots[0], cax=ax2, orientation='horizontal')

    ticks = np.arange(vmin, vmax + 1)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(['First', 'Second', 'Third'][:len(ticks)], fontsize=9)

    cbar.ax.minorticks_off()

    # Optional: Add title
    # ax1.set_title(f'{varname} Distribution', fontsize=10, fontweight='bold')

    return ax


def plot_Fig2_S5(first_plot_data, second_plot_data, third_plot_data):

    plt.rcParams.update({
        'font.family': 'Arial',

        'mathtext.fontset': 'custom',

        'mathtext.rm': 'Arial',  # Regular
        'mathtext.it': 'Arial:italic',  # Italic
        'mathtext.bf': 'Arial:bold',  # Bold

        # Optional (Recommended)
        'mathtext.default': 'regular',  # Avoid auto-italicizing

        'font.size': 10,
        'axes.titlesize': 10,
        'axes.labelsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        # 'text.usetex': False,  # Do not use external LaTeX
    })

    ################  ========= Plot ============ ######################
    fig = plt.figure(figsize=(8.2, 8.2), dpi=300)
    gs = gridspec.GridSpec(2, 2,
                           width_ratios=[1, 1],  # Column width ratios
                           height_ratios=[5, 5],  # Bottom row reserved for colorbar
                           hspace=0.03, wspace=0.15)
    # plt.tight_layout()
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, 0])
    ax4 = plt.subplot(gs[1, 1])


    ######################## Distribution #####################
    fraction_first, bin_center, ax1 = plot_NH_All(first_plot_data, 'first', 'All', ax1)
    fraction_second, bin_center, ax2 = plot_NH_All(second_plot_data, 'second', 'All', ax2)
    fraction_third, bin_center, ax3 = plot_NH_All(third_plot_data, 'third', 'All', ax3)

    ax4 = plot_NH_single(first_plot_data, second_plot_data, third_plot_data, 'Coupling', ax4)

    print(f'Condition: Outlier({Outlier}) CornoSig({OutnosigCor})\n'
          f'    First:   Top2 Total:   Top3 Total:\n')
    for i, label in enumerate(bin_center):
        print(f"{label} - {fraction_first[i]:.1f}%     {(fraction_first[i]+fraction_second[i]):.1f}%    {(fraction_first[i]+fraction_second[i]+fraction_third[i]):.1f}%\n")
        # f"Second: {label} - {fraction_second[i]:.1f}%\n"
        # f"Third: {label} - {fraction_third[i]:.1f}%\n"
        # f"Top2 Total: {label} -{(fraction_first[i]+fraction_second[i]):.1f}%\n",
        # f"Top3 Total: {label} -{(fraction_first[i]+fraction_second[i]+fraction_third[i]):.1f}%")
    for i, label in enumerate(bin_center):
        print(f"Top3 Total: {label} -{(fraction_first[i]+fraction_second[i]+fraction_third[i]):.1f}%")

    # ######################## Colorbar ######################
    # cbar_ax = plt.subplot(gs[-1, 0])  # First variable spatial distribution
    #
    # cbar = fig.colorbar(plots[0], cax=cbar_ax, orientation='horizontal')
    #
    # category_colors = {
    #     'SOS': '#ff7a00',
    #     'Ta': '#f2b701',
    #     'Srad': '#a5aa99',
    #     'Pre': '#e73f74',
    #     'SM': '#11a579',
    #     'VPD': '#3969ac',
    #     'Cor': '#7f3c8d'
    # }
    #
    # # Set colorbar labels and ticks
    # cbar.set_ticks(np.arange(len(category_colors)))
    # cbar.set_ticklabels(list(category_colors.keys()))
    # cbar.ax.tick_params(axis='x', which='major')
    #
    # # Manually adjust colorbar position to center it
    # cbar.ax.set_position([0.21, 0.3, 0.6, 0.02])
    #
    #
    # output_path1 = os.path.join(output_partial_max_var_png_path, rf"SM_VPD_Cor17\In temporal\Climate{climate_test_number}\All\{Basedon}\Preseason\SM_VPD_Cor{cor_test_number}_{analyzed_start_year}-{analyzed_end_year}_Outlier({Outlier})_Outnosig({OutnosigCor})_{the_most_important_var_method}_varname.png")  # SM_VPD_pearson_
    if variable_type == 'sos and climate':
        output_path1 = os.path.join(output_partial_max_var_png_path,
                                    rf"D:\Fig\Fig 2 Driver of POS dynamic\All\SM_VPD_Cor{cor_test_number}_{analyzed_start_year}-{analyzed_end_year}_Outlier({Outlier})_Outnosig({OutnosigCor})_{the_most_important_var_method}_varname_2.png")  # SM_VPD_pearson_
    elif variable_type == 'only climate':
        output_path1 = os.path.join(output_partial_max_var_png_path,
                                    rf"D:\Fig\Fig 2 Driver of POS dynamic\No SOS\All\SM_VPD_Cor{cor_test_number}_{analyzed_start_year}-{analyzed_end_year}_Outlier({Outlier})_Outnosig({OutnosigCor})_{the_most_important_var_method}_varname.png")  # SM_VPD_pearson_
    plt.savefig(output_path1, dpi=300, bbox_inches='tight')
    print(f'PNG saved to: {output_path1}')
    # plt.show()






######## ======= Distinguish Vegetation Types, AI Gradients, and Cor Gradients Plot =============== ###########
def plot_S6_8(varname_data_first, varname_data_second, varname_data_third, grade_by):

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from mpl_toolkits.basemap import Basemap
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib.gridspec as gridspec
    import numpy as np
    import os
    from brokenaxes import brokenaxes
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # Globally update font size settings
    plt.rcParams.update({
        'font.family': 'Arial',

        'mathtext.fontset': 'custom',

        'mathtext.rm': 'Arial',  # Regular
        'mathtext.it': 'Arial:italic',  # Italic
        'mathtext.bf': 'Arial:bold',  # Bold

        # Optional (Recommended)
        'mathtext.default': 'regular',  # Avoid automatic italicization

        'font.size': 10,
        'axes.titlesize': 10,
        'axes.labelsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        # 'text.usetex': False,  # Do not use external LaTeX
    })

    if grade_by == 'Veg':
        fig = plt.figure(figsize=(8.6, 13.5))
        gs = gridspec.GridSpec(5, 3,
                               width_ratios=[1, 1, 1],  # Width ratio of the columns
                               height_ratios=[5, 5, 5, 5, 1],  # The last row is allocated for the colorbar
                               hspace=0.001, wspace=0.2)
        # Define 4 vegetation types
        types = ['Forest', 'Shrub', 'Savanna', 'Grass']#, 'Wet']
        titles = ['Forest', 'Shrub', 'Savanna', 'Grass']# 'Wet']

    elif grade_by == 'AI':
        fig = plt.figure(figsize=(8.6, 13.5))
        gs = gridspec.GridSpec(5, 3,
                               width_ratios=[1, 1, 1],  # Width ratio of the columns
                               height_ratios=[5, 5, 5, 5, 1],  # The last row is allocated for the colorbar
                               hspace=0.001, wspace=0.2)
        types = ['Arid', 'Semi-Arid', 'Dry sub-humid', 'Humid']
        titles = ['Arid', 'Semi-Arid', 'Dry sub-humid', 'Humid']
    elif grade_by == 'Cor mean':
        fig = plt.figure(figsize=(8.6, 15.5))
        gs = gridspec.GridSpec(6, 3,
                               width_ratios=[1, 1, 1],  # Width ratio of the columns
                               height_ratios=[5, 5, 5, 5, 5, 1],  # The last row is allocated for the colorbar
                               hspace=0.3, wspace=0.2)
        types = [#'Cor(<-0.5)',
                 'Cor(<-0.4)', 'Cor(-0.4~-0.3)', 'Cor(-0.3~-0.2)',
                 'Cor(-0.2~-0.1)', 'Cor(-0.1~0)']
        titles = [#'Cor(<-0.5)',
                  'Cor(<-0.4)', 'Cor(-0.4~-0.3)', 'Cor(-0.3~-0.2)',
                  'Cor(-0.2~-0.1)', 'Cor(-0.1~0)']




    for i, (type, title) in enumerate(zip(types, titles)):

        # Set subplot title letters
        subplot_letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        word = subplot_letters[i] if i < len(subplot_letters) else ''

        if grade_by == 'Veg':
            # Create vegetation type mask
            if type == 'All':
                mask = np.isfinite(veg_type_data)
            elif type == 'Forest':
                ax1 = plt.subplot(gs[0, 0])  # Spatial distribution of the 1st variable
                ax2 = plt.subplot(gs[0, 1])  # Spatial distribution of the 1st variable
                ax3 = plt.subplot(gs[0, 2])  # Spatial distribution of the 1st variable
                mask = (veg_type_data == 1)
            elif type == 'Shrub':
                ax1 = plt.subplot(gs[1, 0])  # Spatial distribution of the 1st variable
                ax2 = plt.subplot(gs[1, 1])  # Spatial distribution of the 1st variable
                ax3 = plt.subplot(gs[1, 2])  # Spatial distribution of the 1st variable
                mask = (veg_type_data == 2)
            elif type == 'Savanna':
                ax1 = plt.subplot(gs[2, 0])  # Spatial distribution of the 1st variable
                ax2 = plt.subplot(gs[2, 1])  # Spatial distribution of the 1st variable
                ax3 = plt.subplot(gs[2, 2])  # Spatial distribution of the 1st variable
                mask = (veg_type_data == 3)
            elif type == 'Grass':
                ax1 = plt.subplot(gs[3, 0])  # Spatial distribution of the 1st variable
                ax2 = plt.subplot(gs[3, 1])  # Spatial distribution of the 1st variable
                ax3 = plt.subplot(gs[3, 2])  # Spatial distribution of the 1st variable
                mask = (veg_type_data == 4)
            elif type == 'Wet':
                mask = (veg_type_data == 5)
        elif grade_by == 'AI':
            # Create vegetation type mask
            if type == 'All':
                mask = np.isfinite(ai_data)
            elif type == 'Hyper Arid':
                mask = (ai_data == 1)  # AI 0-0.03
            elif type == 'Arid':
                ax1 = plt.subplot(gs[0, 0])  # Spatial distribution of the 1st variable
                ax2 = plt.subplot(gs[0, 1])  # Spatial distribution of the 1st variable
                ax3 = plt.subplot(gs[0, 2])  # Spatial distribution of the 1st variable
                mask = (ai_data == 2)  # AI 0.03-0.2
            elif type == 'Semi-Arid':
                ax1 = plt.subplot(gs[1, 0])  # Spatial distribution of the 1st variable
                ax2 = plt.subplot(gs[1, 1])  # Spatial distribution of the 1st variable
                ax3 = plt.subplot(gs[1, 2])  # Spatial distribution of the 1st variable
                mask = (ai_data == 3) | (ai_data == 4)  # AI 0.2-0.35
            elif type == 'Dry sub-humid':
                ax1 = plt.subplot(gs[2, 0])  # Spatial distribution of the 1st variable
                ax2 = plt.subplot(gs[2, 1])  # Spatial distribution of the 1st variable
                ax3 = plt.subplot(gs[2, 2])  # Spatial distribution of the 1st variable
                mask = (ai_data == 5)  # AI 0.5-0.65
            elif type == 'Humid':
                ax1 = plt.subplot(gs[3, 0])  # Spatial distribution of the 1st variable
                ax2 = plt.subplot(gs[3, 1])  # Spatial distribution of the 1st variable
                ax3 = plt.subplot(gs[3, 2])  # Spatial distribution of the 1st variable
                mask = (ai_data == 6)  # AI > 0.65

        elif grade_by == 'Cor mean':

            if type == 'All':
                mask = np.isfinite(cor_mean_data)
            elif type == 'Cor(-0.1~0)':
                ax1 = plt.subplot(gs[0, 0])  # Spatial distribution of the 1st variable
                ax2 = plt.subplot(gs[0, 1])  # Spatial distribution of the 1st variable
                ax3 = plt.subplot(gs[0, 2])  # Spatial distribution of the 1st variable
                mask = (-0.1 <= cor_mean_data) & (cor_mean_data < 0)
            elif type == 'Cor(-0.2~-0.1)':
                ax1 = plt.subplot(gs[1, 0])  # Spatial distribution of the 1st variable
                ax2 = plt.subplot(gs[1, 1])  # Spatial distribution of the 1st variable
                ax3 = plt.subplot(gs[1, 2])  # Spatial distribution of the 1st variable
                mask = (-0.2 <= cor_mean_data) & (cor_mean_data < -0.1)
            elif type == 'Cor(-0.3~-0.2)':
                ax1 = plt.subplot(gs[2, 0])  # Spatial distribution of the 1st variable
                ax2 = plt.subplot(gs[2, 1])  # Spatial distribution of the 1st variable
                ax3 = plt.subplot(gs[2, 2])  # Spatial distribution of the 1st variable
                mask = (-0.3 <= cor_mean_data) & (cor_mean_data < -0.2)
            elif type == 'Cor(-0.4~-0.3)':
                ax1 = plt.subplot(gs[3, 0])  # Spatial distribution of the 1st variable
                ax2 = plt.subplot(gs[3, 1])  # Spatial distribution of the 1st variable
                ax3 = plt.subplot(gs[3, 2])  # Spatial distribution of the 1st variable
                mask = (-0.4 <= cor_mean_data) & (cor_mean_data < -0.3)
            elif type == 'Cor(<-0.4)':
                ax1 = plt.subplot(gs[4, 0])  # Spatial distribution of the 1st variable
                ax2 = plt.subplot(gs[4, 1])  # Spatial distribution of the 1st variable
                ax3 = plt.subplot(gs[4, 2])  # Spatial distribution of the 1st variable
                mask = (cor_mean_data < -0.4) #& (-0.5 <= cor_mean_data)
            # elif type == 'Cor(<-0.5)':
            #     ax1 = plt.subplot(gs[5, 0])  # Spatial distribution of the 1st variable
            #     ax2 = plt.subplot(gs[5, 1])  # Spatial distribution of the 1st variable
            #     ax3 = plt.subplot(gs[5, 2])  # Spatial distribution of the 1st variable
            #     mask = (cor_mean_data < -0.5)

        # Apply mask
        plot_varname_data_first = np.where(mask, varname_data_first, np.nan)
        plot_varname_data_second = np.where(mask, varname_data_second, np.nan)
        plot_varname_data_third = np.where(mask, varname_data_third, np.nan)


        ################  ========= plot ============ ######################
        plots = []  # Store plot objects for each subplot

        ########### 1 Spatial distribution and frequency #################
        def plot_distribution_frequency(plot_data, input_ax, importance_grape):
            ########### 1 Spatial distribution #################
            input_ax.set_box_aspect(1)  # Force the map axes to be square so its diameter fills the grid height
            input_ax.axis('off')
            ### 1.1 Create map
            m = Basemap(ax=input_ax,
                        projection='npstere',  # North Polar Stereographic Projection
                        boundinglat=30,  # Lowest latitude to display (currently 30°N)
                        lon_0=0,  # Center longitude (can be modified); 180: Pacific centered; 90: Asia centered
                        resolution='l')

            # Generate meshgrid coordinates
            lons = np.linspace(-180, 180, cols, endpoint=False)
            lats = np.linspace(lat_max, lat_min, rows)
            lons, lats = np.meshgrid(lons, lats)

            # Set latitude and longitude ticks
            m.drawparallels(np.arange(30, 90, 30),
                            labels=[0, 0, 0, 0],
                            linewidth=0.5,
                            # fontsize=8,
                            color="black")

            m.drawmeridians(np.arange(0, 360, 60),
                            latmax=90,  # Make meridian lines intersect at the North Pole
                            labels=[0, 0, 0, 1],  # labels=[left, right, top, bottom] Controls longitude label visibility
                            linewidth=0.6,
                            # fontsize=8,
                            color="gray")

            # # Fill continents
            # m.fillcontinents(color='white', lake_color='white', zorder=1)

            # map_boundary = m.drawmapboundary(linewidth=0)  # Do not display boundary line

            ### 1.2 Plot data
            # Color mapping
            category_colors = {
                'SOS': '#b2df8a',  # Green
                'Ta': '#fdbf6f',  # Orange
                'Srad': '#a5aa99',  # Gray
                'Pre': '#a6cee3',  # Blue
                'SM': '#cab2d6',  # Purple
                'VPD': '#ffff99',  # Yellow
                'Coupling': '#e31a1c'  # Red
            }

            # Create mapping dictionary
            keys = list(category_colors.keys())
            str_to_int = {k: i for i, k in enumerate(keys)}

            # Vectorized transformation (handle NaNs)
            def safe_map(x):
                if isinstance(x, np.ma.core.MaskedConstant) or str(x) == 'nan':
                    return -1
                return str_to_int.get(x, -1)  # Return -1 if key doesn't exist

            # Convert to integer matrix
            int_data = np.vectorize(safe_map)(plot_data)
            int_data = np.ma.masked_where(int_data == -1, int_data)  # Mask invalid values

            # Create colormap
            cmap = ListedColormap(list(category_colors.values()))
            norm = BoundaryNorm(np.arange(len(category_colors) + 1) - 0.5,
                                len(category_colors))

            # Plot spatial distribution
            plot = m.pcolormesh(lons, lats, int_data, cmap=cmap, norm=norm, shading='nearest', latlon=True,
                                zorder=1)  # Avoid tearing in polar regions
            plots.append(plot)  # Save plot object


            ### 1.3 Draw boundaries
            ### Draw boundaries
            terrence = r'D:\CAU\Border\NH_terrence\NH_Terrence_30_90'
            # m.readshapefile(terrence, 'NH_Terrence', drawbounds=True, linewidth=0.2, color='black', zorder=3)
            m.readshapefile(terrence, 'NH_Terrence', drawbounds=False)
            for shape in m.NH_Terrence:
                # Convert list to numpy array for easy calculation
                points = np.array(shape)
                x, y = points[:, 0], points[:, 1]

                # Core logic: Calculate projection distance between adjacent points
                # If the projection distance between two adjacent points suddenly becomes very large, it indicates a "cross-center" loop line
                dist = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)

                # Set a threshold (projection coordinates are usually large, e.g. 100,000 magnitude)
                # If adjacent point distance exceeds 1/10th of map diameter, classify as abnormal jump
                threshold = (input_ax.get_xlim()[1] - input_ax.get_xlim()[0]) * 0.1

                # Find indices of jump points
                break_indices = np.where(dist > threshold)[0]

                if len(break_indices) == 0:
                    # No jumps; draw the entire line directly
                    input_ax.plot(x, y, color='black', linewidth=0.3, zorder=3)
                else:
                    # Jumps detected; split line segments and draw separately
                    # This removes straight lines crossing the origin while keeping normal boundaries
                    start_idx = 0
                    for break_idx in break_indices:
                        input_ax.plot(x[start_idx:break_idx + 1], y[start_idx:break_idx + 1],
                                 color='black', linewidth=0.3, zorder=3)
                        start_idx = break_idx + 1
                    # Draw the final segment
                    input_ax.plot(x[start_idx:], y[start_idx:], color='black', linewidth=0.3, zorder=3)

            # # North Pole
            # x_pole, y_pole = m(0, 90)
            # input_ax.scatter(x_pole, y_pole, color='black', s=5, zorder=3)

            ### 1.4 Clip outermost boundary
            from matplotlib.patches import Circle

            x0, x1 = input_ax.get_xlim()
            y0, y1 = input_ax.get_ylim()

            center = [(x0 + x1) / 2, (y0 + y1) / 2]
            radius = (x1 - x0) / 2

            clip_circle = Circle(center, radius, transform=input_ax.transData)

            for artist in input_ax.collections + input_ax.lines + input_ax.patches:
                artist.set_clip_path(clip_circle)

            boundary_circle = Circle(
                center,
                radius,
                transform=input_ax.transData,
                facecolor='none',
                edgecolor='black',  # Color
                linewidth=1,
                clip_on = False,
                zorder=4  # Place on top layer
            )

            input_ax.add_patch(boundary_circle)


            # input_ax.set_title(f'({word}) {type}')

            ##################### 2 Statistics plot #############################
            ### 2.1 Statistics
            counts = [np.sum(plot_data == k) for k in keys]
            total = np.sum([np.sum(plot_data == k) for k in keys if str(k) != 'nan'])
            fractions = [c / total * 100 for c in counts]

            ### 2.2 Plotting
            bin_centers = ['SOS', 'Ta', 'Srad', 'Pre', 'SM', 'VPD', 'Coupling']

            inset_pos = [0.01, 0, 0.3, 0.3]
            axins = input_ax.inset_axes(inset_pos)
            # axins.set_axis_off()  # Ensure container axis itself is invisible

            # Draw bar chart
            x_indexes = np.arange(len(bin_centers))  # Generates [0, 1, 2, 3, 4, 5, 6]
            plot_colors = [category_colors[k] for k in bin_centers]

            axins.bar(x_indexes, fractions, color=plot_colors, width=0.7)

            if importance_grape == 'first':
                axins.set_ylim(0, 55)

                axins.set_yticks(np.arange(0, 55.01, 10))

            if importance_grape in ['second', 'third']:
                axins.set_ylim(0, 25)

                axins.set_yticks(np.arange(0, 25.01, 5))

            # Formatting (hide inner spines)
            axins.spines['top'].set_visible(False)
            axins.spines['right'].set_visible(False)

            axins.tick_params(axis='x', bottom=False, labelbottom=False)
            axins.tick_params(axis='y', labelsize=7)

            axins.set_ylabel("Frequency (%)")
            # # Add background transparency
            # axins.set_facecolor('white')
            # axins.patch.set_alpha(1.0)  # Ensure fully opaque
            # # axins.set_zorder(5)  # Ensure above map layer (pcolormesh)

        plot_distribution_frequency(plot_varname_data_first, ax1, 'first')
        plot_distribution_frequency(plot_varname_data_second, ax2, 'second')
        plot_distribution_frequency(plot_varname_data_third, ax3, 'third')

    ################### 2 Colorbar ######################
    # Create colorbar
    cbar_ax = plt.subplot(gs[-1, 0])  # Spatial distribution of 1st variable

    cbar = fig.colorbar(plots[0], cax=cbar_ax, orientation='horizontal')

    # Set colorbar labels and ticks
    category_colors = {
        'SOS': '#b2df8a',  # Green
        'Ta': '#fdbf6f',  # Orange
        'Srad': '#a5aa99',  # Gray
        'Pre': '#a6cee3',  # Blue
        'SM': '#cab2d6',  # Purple
        'VPD': '#ffff99',  # Yellow
        'Coupling': '#e31a1c'  # Red
    }

    cbar.set_ticks(np.arange(len(category_colors)))
    cbar.set_ticklabels(list(category_colors.keys()))
    cbar.ax.tick_params(axis='x', which='major')

    # Manually adjust colorbar position to center it
    if grade_by in ['Veg', 'AI']:
        cbar.ax.set_position([0.21, 0.14, 0.6, 0.01])
    elif grade_by == 'Cor mean':
        cbar.ax.set_position([0.21, 0.14, 0.6, 0.01])

    # Display figure
    plt.tight_layout()

    # output_path1 = os.path.join(output_partial_max_var_png_path, rf"SM_VPD_Cor17\In temporal\Climate{climate_test_number}\{grade_by}\{Basedon}\Preseason\SM_VPD_Cor{cor_test_number}_{analyzed_start_year}-{analyzed_end_year}_Outlier({Outlier})_Outnosig({OutnosigCor})_{the_most_important_var_method}_varname.png")  # SM_VPD_pearson_
    if variable_type == 'sos and climate':
        output_path1 = os.path.join(output_partial_max_var_png_path, rf"D:\Fig\Fig 2 Driver of POS dynamic\{grade_by}\SM_VPD_Cor{cor_test_number}_{analyzed_start_year}-{analyzed_end_year}_Outlier({Outlier})_Outnosig({OutnosigCor})_{the_most_important_var_method}_varname_2.png")  # SM_VPD_pearson_
    elif variable_type == 'only climate':
        output_path1 = os.path.join(output_partial_max_var_png_path,
                                    rf"D:\Fig\Fig 2 Driver of POS dynamic\No SOS\{grade_by}\SM_VPD_Cor{cor_test_number}_{analyzed_start_year}-{analyzed_end_year}_Outlier({Outlier})_Outnosig({OutnosigCor})_{the_most_important_var_method}_varname.png")  # SM_VPD_pearson_
    plt.savefig(output_path1, dpi=300, bbox_inches='tight')
    print(f'PNG saved to: {output_path1}')

    # if grade_by == 'Cor mean':
    #     cor_mean_tif_path = 'D:\CAU\phenology_swc_vpd\Global_test5\Data\Classify\Cor_mean'
    #     out_put = os.path.join(cor_mean_tif_path, 'NH_cor_mean_55km(Python).tif')
    #     save_tif_gdal(out_put, cor_mean_data, rows, cols, crs, gt)
    # plt.show()



######## ======= Plot variable r values ============= ##########
def plot_cor_pos_slope_forAllvegType(plot_data_slope, plot_data_r2, plot_data_pvalue, colorbarmin, colorbarmax, data_sig, name, ax):

    plt.rcParams.update({
        'font.family': 'Arial',

        'mathtext.fontset': 'custom',

        'mathtext.rm': 'Arial',  # Regular
        'mathtext.it': 'Arial:italic',  # Italic
        'mathtext.bf': 'Arial:bold',  # Bold

        # Optional (Recommended)
        'mathtext.default': 'regular',  # Avoid automatic italicization

        'font.size': 9,
        'axes.titlesize': 9,
        'axes.labelsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        # 'text.usetex': False,  # Do not use external LaTeX
    })

    fig = ax.get_figure()
    gs_inner = ax.get_subplotspec().subgridspec(2, 2,
                                                width_ratios=[5, 0.8],
                                                height_ratios=[5, 0.3],
                                                hspace=0.17, wspace=0.01)

    # Hide parent ax as it serves only as a placeholder
    ax.axis('off')

    plots = []  # Store plot objects for each subplot

    # ax1 = plt.subplot(gs[0, 0])
    # ax2 = plt.subplot(gs[0, 1])
    # ax3 = plt.subplot(gs[0, 2])
    # Create the actual three inner sub-axes
    ax1 = fig.add_subplot(gs_inner[0, 0])  # Map
    ax2 = fig.add_subplot(gs_inner[0, 1])  # Latitudinal profile
    ax3 = fig.add_subplot(gs_inner[1, :])  # Colorbar spanning two columns



    ########### Subplot 1: Spatial distribution #################
    ax1.set_box_aspect(1)  # Force map axis to be square so its diameter fills grid height
    ### Create map
    m = Basemap(ax=ax1,
                projection='npstere',   # North Polar Stereographic Projection
                boundinglat=30,         # Lowest latitude to display (currently 30°N)
                lon_0=0,                # Center longitude (can be modified); 180: Pacific centered; 90: Asia centered
                resolution='l')

    # Generate meshgrid coordinates
    lons = np.linspace(-180, 180, cols, endpoint=False)
    lats = np.linspace(lat_max , lat_min , rows)
    lons, lats = np.meshgrid(lons, lats)

    # Set latitude and longitude ticks
    m.drawparallels(np.arange(30, 90, 30),
                    labels=[0, 0, 0, 0],
                    linewidth=0.5,
                    # fontsize=8,
                    color="black")

    m.drawmeridians(np.arange(0, 360, 60),
                    latmax=90,  # Make meridians intersect at North Pole
                    labels=[0, 0, 0, 1],  # labels=[left, right, top, bottom] Controls longitude label visibility
                    linewidth=0.6,
                    # fontsize=8,
                    color="gray")

    # # Fill continents
    # m.fillcontinents(color='white', lake_color='white', zorder=1)

    # map_boundary = m.drawmapboundary(linewidth=0)  # Do not display boundary line


    ### Plot data
    # Color mapping
    piyg_cmap = mpl.colormaps['PiYG']
    colors = piyg_cmap(np.linspace(0, 1, 8))
    cmap = mpl.colors.ListedColormap(colors)

    bins = np.linspace(colorbarmin, colorbarmax, 9)
    norm = mpl.colors.BoundaryNorm(bins, cmap.N)

    # Plot
    # plot_data_slope = np.where(lats >= 30, plot_data_slope, np.nan)
    # plot_data_slope = np.hstack([plot_data_slope, plot_data_slope[:, 0:1]])
    # # plot_data_pvalue = np.hstack([plot_data_pvalue, plot_data_pvalue[:, 0:1]])
    # from mpl_toolkits.basemap import addcyclic
    # plot_data_slope, lons1 = addcyclic(plot_data_slope, lons[0, :])
    # plot_data_pvalue, _ = addcyclic(plot_data_pvalue, lons[0, :])
    #

    # lons, lats = np.meshgrid(lons1, lats[:, 0])
    plot = m.pcolormesh(lons, lats, plot_data_slope, cmap=cmap, norm=norm, shading='nearest', latlon=True,
                        zorder=1)  # Avoid polar region tearing

    plots.append(plot)  # Save plot object

    if data_sig == 'Yes':
        # Add significance stippling/markers
        significant_mask = (plot_data_pvalue < 0.05) & np.isfinite(plot_data_pvalue) & np.isfinite(plot_data_slope)

        if np.any(significant_mask):
            sig_rows, sig_cols = np.where(significant_mask)
            sig_lons = lons[sig_rows, sig_cols]
            sig_lats = lats[sig_rows, sig_cols]

            # Convert geographical coordinates to projection coordinates
            sig_x, sig_y = m(sig_lons, sig_lats)

            ax1.scatter(sig_x, sig_y, marker='.', color='black', s=0.5,
                       linewidth=0.1, zorder=2)
    ax1.set_frame_on(False)

    ### Draw boundaries
    terrence = r'D:\CAU\Border\NH_terrence\NH_Terrence_30_90'
    # m.readshapefile(terrence, 'NH_Terrence', drawbounds=True, linewidth=0.2, color='black', zorder=3)
    m.readshapefile(terrence, 'NH_Terrence', drawbounds=False)
    for shape in m.NH_Terrence:
        # Convert list to numpy array for easy calculation
        points = np.array(shape)
        x, y = points[:, 0], points[:, 1]

        # Core logic: Calculate projection distance between adjacent points
        # If the projection distance between two adjacent points suddenly becomes very large, it indicates a "cross-center" loop line
        dist = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)

        # Set a threshold (projection coordinates are usually large, e.g. 100,000 magnitude)
        # If adjacent point distance exceeds 1/10th of map diameter, classify as abnormal jump
        threshold = (ax1.get_xlim()[1] - ax1.get_xlim()[0]) * 0.1

        # Find indices of jump points
        break_indices = np.where(dist > threshold)[0]

        if len(break_indices) == 0:
            # No jumps; draw the entire line directly
            ax1.plot(x, y, color='black', linewidth=0.3, zorder=3)
        else:
            # Jumps detected; split line segments and draw separately
            # This removes straight lines crossing the origin while keeping normal boundaries
            start_idx = 0
            for break_idx in break_indices:
                ax1.plot(x[start_idx:break_idx + 1], y[start_idx:break_idx + 1],
                         color='black', linewidth=0.3, zorder=3)
                start_idx = break_idx + 1
            # Draw the final segment
            ax1.plot(x[start_idx:], y[start_idx:], color='black', linewidth=0.3, zorder=3)

    x_pole, y_pole = m(0, 90)
    ax1.scatter(x_pole, y_pole, color='black', s=5, zorder=3)

    ### Clip outermost boundary
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
        zorder=4  # Place on top layer
    )

    ax1.add_patch(boundary_circle)

    data_gte0 = plot_data_slope[(plot_data_slope >= 0) & np.isfinite(plot_data_slope)]
    data_lt0 = plot_data_slope[(plot_data_slope < 0) & np.isfinite(plot_data_slope)]
    sum_count = np.sum(np.isfinite(plot_data_r2))

    data_gte0_count = np.sum(np.isfinite(data_gte0))
    data_lt0_count = np.sum(np.isfinite(data_lt0))

    data_gte0_ratio = data_gte0_count / sum_count * 100
    data_lt0_ratio = data_lt0_count / sum_count * 100

    h = 0.25
    v = 0.82
    ax1.text(h, v,
             f'Mean = {np.nanmean(plot_data_slope):.2f}\n'
             f'P = {np.nanmean(data_gte0):.2f} ({data_gte0_ratio:.1f}%)\n'
             f'N = {np.nanmean(data_lt0):.2f} ({data_lt0_ratio:.1f}%)',
             transform=ax1.transAxes,  # Use relative coordinates for easy positioning
             multialignment='center',  # Center vertically
             fontsize = 10)

    ### Significance statistics ###
    data_gt0_sig_005 = plot_data_slope[(plot_data_slope > 0) & (plot_data_pvalue <= 0.05) & np.isfinite(plot_data_slope) & np.isfinite(plot_data_pvalue)]
    data_gt0_sig_010 = plot_data_slope[(plot_data_slope > 0) & (plot_data_pvalue <= 0.1) & np.isfinite(plot_data_slope) & np.isfinite(plot_data_pvalue)]
    data_lte0_sig_005 = plot_data_slope[(plot_data_slope <= 0) & (plot_data_pvalue <= 0.05) & np.isfinite(plot_data_slope) & np.isfinite(plot_data_pvalue)]
    data_lte0_sig_010 = plot_data_slope[(plot_data_slope <= 0) & (plot_data_pvalue <= 0.1) & np.isfinite(plot_data_slope) & np.isfinite(plot_data_pvalue)]
    sum_count = np.sum(np.isfinite(plot_data_r2))

    data_gt0_sig_005_count = np.sum(np.isfinite(data_gt0_sig_005))
    data_gt0_sig_010_count = np.sum(np.isfinite(data_gt0_sig_010))
    data_lte0_sig_005_count = np.sum(np.isfinite(data_lte0_sig_005))
    data_lte0_sig_010_count = np.sum(np.isfinite(data_lte0_sig_010))

    data_gt0_sig_005ratio = data_gt0_sig_005_count / sum_count * 100
    data_gt0_sig_010ratio = data_gt0_sig_010_count / sum_count * 100
    data_lte0_sig_005ratio = data_lte0_sig_005_count / sum_count * 100
    data_lte0_sig_010ratio = data_lte0_sig_010_count / sum_count * 100
    print(f'data_gt0_sig_005ratio = {data_gt0_sig_005ratio}\n'
          f'data_gt0_sig_010ratio = {data_gt0_sig_010ratio}\n'
          f'data_lte0_sig_005ratio = {data_lte0_sig_005ratio}\n'
          f'data_lte0_sig_010ratio = {data_lte0_sig_010ratio}')


    ########### Subplot 2: Latitudinal profile / trend

    # Use actual latitude values as y-axis
    lat_centers = lats[:, 0]

    plot_data_lat = np.nanmean(plot_data_slope, axis=1)

    ax2.axvline(x=0, color='gray', linestyle='--', linewidth=1)

    ax2.plot(plot_data_lat, lat_centers, color='red', linewidth=1, alpha=0.8)

    ax2.set_xlim(-1, 1)
    ax2.set_xticks(np.arange(-1, 1.01, 1))
    ax2.set_xticklabels(['-1', '0', '1'])  # Manually set tick labels

    # tick_size = plt.rcParams['xtick.labelsize']

    # ax2.text(
    #     0.45,  # x = tick position (data coordinates)
    #     -0.005,  # y = slightly lower (axes coordinates)
    #     r'$×10^{-1}$',  # Custom text content
    #     transform=ax2.get_xaxis_transform(),
    #     ha='left',  # Align left to prevent compression
    #     va='top',
    #     fontsize = tick_size,
    #     clip_on=False
    # )

    ax2.set_ylim(30, 90)
    ax2.set_yticks(np.arange(30, 91, 10))
    ax2.set_yticklabels(f'{x}°' for x in np.arange(30, 91, 10))

    ax2.tick_params(axis='both', which='major', length = 2, pad=3)

    ########### Subplot 3: Colorbar
    ### Generate Colorbar (using the final slot)
    cbar = fig.colorbar(plots[0], cax=ax3, orientation='horizontal')

    cbar.set_ticks(bins)

    cbar.set_ticklabels(['0' if x == 0 else
         # f'{int(x * 10)}' if x * 10 == int(x * 10) else
         # f'{x * 10}'
         f'{x:.1f}'
         for x in bins
         ])

    #
    # ax3.text(
    #     1.05,  # x = tick position (data coordinates)
    #     -0.25,  # y = slightly lower (axes coordinates)
    #     r'$×10^{-1}$',  # Custom text content
    #     transform=ax3.transAxes,
    #     ha='left',  # Align left to prevent compression
    #     va='top',
    #     fontsize=tick_size,
    #     clip_on=False
    # )

    cbar.set_label('Slope between coupling and PPT')


    plt.tight_layout()

    # Current bottom-left of ax1
    pos1 = ax1.get_position()

    pos2 = ax2.get_position()

    pos3 = ax3.get_position()

    if name == 'All':
        # Re-adjust ax1 position
        ax1.set_position([
            pos1.x0 -0.06,  # Left position unchanged
            pos2.y0,  # Align bottom with ax2
            pos2.height,
            pos2.height
        ])  # [left, bottom, width, height]
    else:
        ax1.set_position([
            pos1.x0 - 0.11,  # Left position unchanged
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

    plt.tight_layout()

    plt.show()
    plt.close()





first_plot_data = varname_first.astype(str)
second_plot_data = varname_second.astype(str)
third_plot_data = varname_third.astype(str)

###### Fig2 ######
cor_lt0_maks = cor_mean_data <= 0
first_plot_data = np.where(cor_lt0_maks, first_plot_data, np.nan)
second_plot_data = np.where(cor_lt0_maks, second_plot_data, np.nan)
third_plot_data = np.where(cor_lt0_maks, third_plot_data, np.nan)

plot_Fig2_S5(first_plot_data, second_plot_data, third_plot_data)
print('Fig 2 plot done!')

# print(f'Condition: Outlier({Outlier}) CornoSig({OutnosigCor})\n'
#       f'    First:   Top2 Total:   Top3 Total:\n')
# for i, label in enumerate(bin_center):
#     print(f"{label} - {fraction_first[i]:.1f}%     {(fraction_first[i]+fraction_second[i]):.1f}%    {(fraction_first[i]+fraction_second[i]+fraction_third[i]):.1f}%\n")
#           # f"Second: {label} - {fraction_second[i]:.1f}%\n"
#           # f"Third: {label} - {fraction_third[i]:.1f}%\n"
#           f"Top2 Total: {label} -{(fraction_first[i]+fraction_second[i]):.1f}%")
#           # f"Top3 Total: {label} -{(fraction_first[i]+fraction_second[i]+fraction_third[i]):.1f}%")
# for i, label in enumerate(bin_center):
#     print(f"Top3 Total: {label} -{(fraction_first[i]+fraction_second[i]+fraction_third[i]):.1f}%")
#
# # # ###### SI ######
# plot_S6_8(first_plot_data, second_plot_data, third_plot_data, 'Veg')   # VegType / AI / Cor mean
# print('S6 plot done!')
# plot_S6_8(first_plot_data, second_plot_data, third_plot_data, 'AI')   # VegType / AI / Cor mean
# print('S7 plot done!')
# print('S7 plot done!')
# plot_S6_8(first_plot_data, second_plot_data, third_plot_data, 'Cor mean')   # VegType / AI / Cor mean
# print('S8 plot done!')


