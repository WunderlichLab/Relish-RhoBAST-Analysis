# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 08:12:43 2024

@author: emmar

Name: Emma Rits
Date: Friday, August 30, 2024
Description: Cell fate predictor SVM

"""

#%% file info

file_path_trace  = "/path/to/your/data/AllDatasets_TraceDescriptors"
file_path_clus   = "/path/to/your/data/AllDatasets_SubclusterTraces"
file_path_fft    = "/path/to/your/data/AllDatasets_FFT"
file_path_svm    = "/path/to/your/data/AllDatasets_SVM"
file_path_dict   = "/path/to/your/data/AllDatasets_IntensitiesDict"


#%% import packages, functions, and data

# packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
import pickle
import copy
import random
from matplotlib.lines import Line2D
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset, inset_axes
from scipy.integrate import trapz, simps, quad
from statistics import mean 
from sklearn.svm import SVC
from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from feature_engine.selection import RecursiveFeatureAddition
from scipy.stats import uniform, randint, chisquare
from IPython.display import display


# current date
import datetime
current_date   = datetime.date.today()
formatted_date = current_date.strftime("%Y-%m-%d")


# functions
from ratiotimetrace_clustering_compiled import save_data, import_data, fillna_custom, dataframe_treat, cull_nans, smooth
from trace_behavior_sorting_SVM import test_SVM_params, test_param_grid, test_param_dist
from cell_trace_classifying_interface import flatten_trace_df


# # UNCOMMENT DURING FIRST RUN
# # data (from lab drive)
# goodcomp_dict_intensities              = import_data(file_path_dict, "dict_intensities_alldatasets_goodcells_071624", file_type = "")

# # data (from local drive)
# # goodcomp_dict_intensities              = import_data(file_path_local, "dict_intensities_alldatasets_goodcells_071624", file_type = "")
# # goodcomp_SVM_results_dict              = import_data(file_path_SVM, "goodcomp_SVM_results_dict")
# # goodcomp_subcluster_traces_div_smooth  = import_data(file_path_local, "goodcomp_subcluster_traces_div_smooth")
goodcomp2_dict_intensities_area        = import_data(file_path_local, "dict_intensities_alldatasets_goodcells_areas_090924", file_type = "")
goodcomp2_SVM_results_dict_noIc        = import_data(file_path_SVMc, "goodcomp2_SVM_results_dict_noIc")
goodcomp2_subcluster_traces_div_smooth = import_data(file_path_local, "goodcomp2_subcluster_traces_div_smooth")
goodcomp3_dict_intensities_area        = import_data(file_path_local, "dict_intensities_alldatasets_goodcells_areas_101024", file_type = "")
goodcomp3_SVM_results_dict_noIc        = import_data(file_path_SVMc, "goodcomp3_SVM_results_dict_noIc")
goodcomp3_subcluster_traces_div_smooth = import_data(file_path_local, "goodcomp3_subcluster_traces_div_smooth")


#%% functions to prep data

def extract_data(dict_intensities, stim_time = 30, sg_order = 2, sg_factor = 5, area = True):
    """
    Parameters
    ----------
    curr_data : str
        Folder name for current dataset
    type_dict_intensities : str
        # filtered version of dict_intensities that gets imported
    normtype : str
        Type of normalization to be performed (subtractive or divisive)
    
    Returns
    -------
    df_dict : dict
        Dictionary of dfs containing ratio timecourse values for PGN, 1X, 10X, and 100X compiled for all dates
        Org:  Original (with time column)
        Copy: Revised (with time column removed)
        # Indexing: ratio_subsets[treatment][replicate][cell][time][value]
        #     Treatments: 0 = PGN
        #                 1 = 1X
        #                 2 = 10X
        #                 3 = 100X
        #     Replicates (3x)
        #     Cell (# varies)
        #     Time (0-34): Ordered according to timepoint
        #     Value:      0 = Time
        #                 1 = Ratio value
    Z_dict : dict
        Dictionary of arrays containing linkage matrices for PGN, 1X, 10X, and 100X
    """
    # retrieve dictionary from file
    dates             = list(dict_intensities.keys())
    treatments        = ["-PGN", "1X", "10X", "100X"]
    locations         = ["cyto", "nuclear", "ratio", "total", "cyto areas", "nuclear areas", "total areas"] if area else ["cyto", "nuclear", "ratio", "total"]
    locations_dict    = dict.fromkeys(locations)
    
    for location in locations:
        if "total" in location:
            continue
        
        dict_comp         = dict.fromkeys(dates)
        dict_comp_smooth  = dict.fromkeys(treatments)
        dict_comp_smooth_copy = dict.fromkeys(treatments)
        trunc_time        = None
        
        for date, date_data in dict_intensities.items():
            for viewframe, viewframe_data in date_data.items():
                for cell, cell_data in viewframe_data[location].items():
                    end_time   = cell_data[-1][0]
                    trunc_time = int(end_time) if trunc_time is None or end_time < trunc_time else trunc_time
        # print(trunc_time)
        
        for date in dates:
            date_str          = ''.join([c for c in date if c != "-"])
            dict_date         = dict_intensities[date]
            keys              = list(dict_date.keys())
            
            # define -PGN, 1X, 10X, and 100X subsets of dict_intensities
            dict_PGNsubset    = {k: dict_date[k] for k in keys if "-PGN" in k}
            keys_PGNsubset    = list(dict_PGNsubset.keys())    
            keys_PGNsubset.reverse()
            dict_1Xsubset     = {k: dict_date[k] for k in keys if "1X" in k}
            keys_1Xsubset     = list(dict_1Xsubset.keys())
            keys_1Xsubset.reverse()
            dict_10Xsubset    = {k: dict_date[k] for k in keys if "10X" in k}
            keys_10Xsubset    = list(dict_10Xsubset.keys())
            keys_10Xsubset.reverse()
            dict_100Xsubset   = {k: dict_date[k] for k in keys if "100X" in k}
            keys_100Xsubset   = list(dict_100Xsubset.keys())
            keys_100Xsubset.reverse()
            # print(date, "PGN: ", keys_PGNsubset)
            # print(date, "1X: ", keys_1Xsubset)
            # print(date, "10X: ", keys_10Xsubset)
            # print(date, "100X: ", keys_100Xsubset, "\n")
            
            # compile subsets defined above
            keys_subsets      = [keys_PGNsubset, keys_1Xsubset, keys_10Xsubset, keys_100Xsubset]
            dict_subsets      = [dict_PGNsubset, dict_1Xsubset, dict_10Xsubset, dict_100Xsubset]
            subsets_dict      = {}
        
            for i in range(4): # looping through all four treatments    
                treatment         = treatments[i]    
                key_subset        = keys_subsets[i]
                # print(key_subset)
                dict_subset       = dict_subsets[i]
                loc_subset        = []
                cell_name_subset  = []
                
                for j in range(len(key_subset)): # looping through all replicates
                    # retrieve field of view    
                    viewframes        = []
                    pattern           = r'(\d+_[-\w]+_\d+)'
                    for string in key_subset:
                        match = re.search(pattern, string)
                        if match:
                            viewframes.append(match.group(1))
                    viewframe         = viewframes[j]
                    viewframe_no      = viewframe[-1]
                    # print(viewframe, viewframe_no)
                    framenames        = [k for k in key_subset if viewframe in k]
                    
                    if len(framenames) != 0:
                        tif_key = framenames[0]
                        # print(tif_key)
                    else:
                        continue
                    
                    dct               = dict_subset[tif_key]
                    loc_reps        = []                        # list of values across all three replicates
                    cell_name_reps    = []                      # list of cell names across all three replicates
        
                    for key in list(dct[location].keys()): # looping through all cells
                        data          = []                      # list of values across current replicate
                        
                        value         = dct[location][key]       # list of values for current cell
                        # print(date, treatment, key, len(value))
                        data         += value
                        
                        cell_no       = int(key.strip("Cell ")) # cell number of current cell
                        cell_name     = "Cell " + str(date_str) + "-" + str(viewframe_no) + "-" + str(cell_no)
                        cell_name_reps.append(cell_name)
                        
                        # update
                        loc_reps   += [data]
                        
                    # add replicate data to list
                    loc_subset     += loc_reps
                    cell_name_subset += cell_name_reps
                
                # create dict with all values (across three replicates for each dates) for each treatment
                reps_dict        = dict(zip(cell_name_subset, loc_subset))
        
                # replace NaNs
                temp_df           = pd.DataFrame(reps_dict)              # convert to df
                temp_df           = fillna_custom(temp_df)                # replace NaNs
                dates_dict        = pd.DataFrame.to_dict(temp_df)         # convert back to dict
                
                # associate each dict with its corresponding treatment
                subsets_dict[treatment] = dates_dict
                    
            # create dataframe
            temp_df_ratio      = pd.DataFrame(subsets_dict)
            
            # make dataframe (by treatment); "copies" all have time column deleted
            df_loc_PGN       = dataframe_treat(temp_df_ratio, "-PGN")
            df_loc_PGN_org, df_loc_PGN_copy      = cull_nans(df_loc_PGN)
            df_loc_PGN_copy  = df_loc_PGN_copy.transpose()
            df_loc_1X        = dataframe_treat(temp_df_ratio, "1X")
            df_loc_1X_org, df_loc_1X_copy        = cull_nans(df_loc_1X)
            df_loc_1X_copy   = df_loc_1X_copy.transpose()
            df_loc_10X       = dataframe_treat(temp_df_ratio, "10X")
            df_loc_10X_org, df_loc_10X_copy      = cull_nans(df_loc_10X)
            df_loc_10X_copy  = df_loc_10X_copy.transpose()
            df_loc_100X      = dataframe_treat(temp_df_ratio, "100X")
            df_loc_100X_org, df_loc_100X_copy    = cull_nans(df_loc_100X)
            df_loc_100X_copy = df_loc_100X_copy.transpose()
            
            df_dict_org        = {"-PGN": df_loc_PGN_org, "1X": df_loc_1X_org, "10X": df_loc_10X_org, "100X": df_loc_100X_org}
            df_dict_copy       = {"-PGN": df_loc_PGN_copy, "1X": df_loc_1X_copy, "10X": df_loc_10X_copy, "100X": df_loc_100X_copy}
            
            times              = list(df_loc_PGN_org["Time"])
            # print(times)
            df_dict_smooth_org, df_dict_smooth_copy = smooth(df_dict_copy, times, sg_factor, sg_order, trunc_time = trunc_time)
            
            # update dictionaries
            dict_comp[date]        = df_dict_org
            
            for treat in df_dict_smooth_org:
                if treat in dict_comp_smooth and dict_comp_smooth[treat] is not None:
                    cols_to_use = df_dict_smooth_org[treat].columns.difference(dict_comp_smooth[treat].columns)
                    dict_comp_smooth[treat] = pd.concat([dict_comp_smooth[treat], df_dict_smooth_org[treat][cols_to_use]], axis=1)
                else: # if not, assign the df from the generated dictionary to dict_comp_smooth
                    dict_comp_smooth[treat]   = df_dict_smooth_org[treat]
            
            for treat in df_dict_smooth_copy:
                if treat in dict_comp_smooth_copy: # if the treatment key exists, concatenate the new df to the existing one
                    dict_comp_smooth_copy[treat] = pd.concat([dict_comp_smooth_copy[treat], df_dict_smooth_copy[treat]], axis=0)
                else: # if not, assign the df from the generated dictionary to dict_comp_smooth
                    dict_comp_smooth_copy[treat] = df_dict_smooth_copy[treat]
        
        # num_cells = [df.shape[0] for treatment, df in dict_comp_smooth_copy.items()]
        # print("Num cells: ", num_cells)
        
        # update locations_dict
        locations_dict[location] = dict_comp_smooth_copy
        
    # calculate total Relish
    total_dict = dict.fromkeys(treatments)
    
    for treatment in treatments:
        # extract cytoplasmic data
        treatment_cyto_df = locations_dict["cyto"][treatment]
        treatment_cyto_idxs = treatment_cyto_df.index
        
        # extract nuclear data
        treatment_nuc_df  = locations_dict["nuclear"][treatment]
        treatment_nuc_idxs  = treatment_nuc_df.index
        
        if not treatment_cyto_idxs.equals(treatment_nuc_idxs):
            raise ValueError("Cell indices must be the same for both nuclear and cytoplasmic data.")
        if treatment_cyto_df.equals(treatment_nuc_df):
            raise ValueError(f"{treatment}: Cytoplasmic and nuclear dataframes are equal.")
        
        # sum for total
        treatment_total_df = treatment_cyto_df.add(treatment_nuc_df, fill_value = 0)
        
        # update corresponding treatment entry in total_dict
        total_dict[treatment] = treatment_total_df
    
    locations_dict["total"] = total_dict
    
    # calculate total area
    total_area_dict = dict.fromkeys(treatments)
    
    for treatment in treatments:
        # extract cytoplasmic data
        treatment_cyto_df = locations_dict["cyto areas"][treatment]
        treatment_cyto_idxs = treatment_cyto_df.index
        
        # extract nuclear data
        treatment_nuc_df  = locations_dict["nuclear areas"][treatment]
        treatment_nuc_idxs  = treatment_nuc_df.index
        
        if not treatment_cyto_idxs.equals(treatment_nuc_idxs):
            raise ValueError("Cell indices must be the same for both nuclear and cytoplasmic data.")
        if treatment_cyto_df.equals(treatment_nuc_df):
            raise ValueError(f"{treatment}: Cytoplasmic and nuclear dataframes are equal.")
        
        # sum for total
        treatment_total_df = treatment_cyto_df.add(treatment_nuc_df, fill_value = 0)
        
        # update corresponding treatment entry in total_dict
        total_area_dict[treatment] = treatment_total_df
    
    locations_dict["total areas"] = total_area_dict
    
    return locations_dict


def process_data(locations_dict, SVM_results_dict, stim_time = 30, trunc = True):
    locations = list(locations_dict.keys())
    # print(locations)
    locations_dict_process = dict.fromkeys(locations)
    
    for location, location_data in locations_dict.items():
        location_df = pd.DataFrame()
        
        for treatment, treatment_df in location_data.items():
            if trunc:
                # extract pre-stimulus data
                treatment_df = treatment_df.copy().iloc[:, :stim_time]
            
            else:
                # extract all data
                treatment_df = treatment_df.copy()
                
            # add treatment to cell names
            treatment_df.index = treatment + " " + treatment_df.index
            # print(f"{location} ({treatment}): {treatment_df.index}")
            
            # update location_df
            if location_df.empty:
                location_df = treatment_df
                # print(f"{location}, {treatment}")
            else:
                location_df = pd.concat([location_df, treatment_df], axis = 0)
        
        locations_dict_process[location] = location_df
        
    # check to make sure cell names and order are consistent
    consistency, cell_names = check_cell_names(locations_dict_process)
    # print(list(cell_names))
    # print(len(cell_names))
    if not consistency:
        raise ValueError("Cell names are not consistent.")
        
    # calculate average and SD values for each cell
    col_names = []
    for location in ["cyto", "nuclear", "ratio", "total", "cyto areas", "nuclear areas", "total areas"]:
        col_names += [f"{location} (avg)", f"{location} (SD)", f"{location} (AUC)"]
                         
    averages_df = pd.DataFrame(index = cell_names, columns = ["Behavior"] + col_names)
    
    # update location columns with average values (t = 0 to t = stim_time)
    for location in locations:
        # retrieve location data
        cell_data = locations_dict_process[location]
        
        # calculate average and SD
        avg_data  = cell_data.mean(axis = 1)
        sd_data   = cell_data.std(axis = 1)
        # print(f"{location}: {avg_data}")
        # print(f"{location}: {type(avg_data)}")
        # print(f"{location}: {type(sd_data)}")
        # print(f"{location}: {cell_data.index}")
        # print(f"{location}: {averages_df.index}")
        # print(f"Indices equal {location}: {cell_data.index.equals(averages_df.index)}")
        averages_df.loc[:, f"{location} (avg)"] = avg_data
        averages_df.loc[:, f"{location} (SD)"]  = sd_data
        
        # calculate area under the curve
        times = cell_data.columns.astype(float)
        auc_data = cell_data.apply(lambda cell: trapz(cell, x = times), axis = 1)
        averages_df.loc[:, f"{location} (AUC)"] = auc_data
    
    # update behavior column with SVM results
    for cell in averages_df.index:
        cell_behavior = SVM_results_dict["All cells"].loc[cell, "Predicted"]
        # print(f"{cell}: {cell_behavior}")
        averages_df.loc[cell, "Behavior"] = cell_behavior
    
    return locations_dict_process, averages_df


def check_cell_names(locations_dict_process):
    # use the indices of the "cyto" df as a reference
    reference_idx = locations_dict_process[list(locations_dict_process.keys())[0]].index
    
    # check if the other indices are the same
    if all (location_df.index.equals(reference_idx) for location_df in locations_dict_process.values()):
        # print("Cell names are consistent.")
        return True, reference_idx
    else:
        # print("Cell names are not consistent.")
        return False
    

#%% functions to visualize features

def plot_feature_traces(locations_dict, averages_df, SVM_results_dict, predictor_SVM_results_df, num_categories = 5, stim_time = 30):
    colors          = ["#DC143C", "#FF6F61", "indigo", "dodgerblue", "grey"]
    color_palette   = sns.color_palette(colors)
    behavior_colors_dict = {5: {"I": color_palette[0], "Id": color_palette[1], "G": color_palette[2], "D": color_palette[3], "N": color_palette[4]},
                            3: {"I": color_palette[0], "L": color_palette[3], "N": color_palette[4]},
                            2: {"R": color_palette[0], "N": color_palette[4]},
                            "imm": {"I": color_palette[0], "NI": color_palette[4]}}
    behavior_keys_dict   = {5: {"I": "Immediate", "Id": "Immediate with decay", "G": "Gradual", "D": "Delayed", "N": "Nonresponsive"},
                            3: {"I": "Immediate", "L": "Long-term", "N": "Nonresponsive"},
                            2: {"R": "Responsive", "N": "Nonresponsive"},
                            "imm": {"I": "Immediate", "NI": "Non-immediate"}}
    
    behavior_colors        = behavior_colors_dict[num_categories]
    behavior_keys          = behavior_keys_dict[num_categories]
    behavior_rev           = {v: k for k, v in behavior_keys.items()}
    
    # extract data
    locations_dict_full, _ = process_data(locations_dict, SVM_results_dict, trunc = False)
    features               = list(locations_dict_full)
    
    # initialize figure
    fig, axs               = plt.subplots(4, len(features), figsize = (5 + (5 * len(features)), 15))
    col_names              = [feature.title() for feature in features]
    row_names              = ["Traces", "Average (pre-stimulus)", "SD (pre-stimulus)", "AUC (pre-stimulus)"]
    trace_ylabel_dict      = {"cyto": "Amount of Relish (fluorescence)",
                                 "nuclear": "Amount of Relish (fluorescence)",
                                 "ratio": "Nuclear/cytoplasmic Relish (ratio)",
                                 "total": "Amount of Relish (fluorescence)",
                                 "cyto areas": f"Area (μm$^2$)",
                                 "nuclear areas": f"Area (μm$^2$)",
                                 "total areas": f"Area (μm$^2$)"}
    
    # extract real behavior (predictions from classifer SVM)
    real_behaviors = SVM_results_dict["All cells"]["Predicted"]

    if num_categories == 2:
        real_behaviors = real_behaviors.replace({
            "N": "N",               # keep nonresponsive category unchanged
            "G": "R",               # label gradual cells as responsive
            "D": "R",               # label delayed cells as responsive
            "I": "R",               # label immediate cells as responsive
            "Id": "R"})             # label immediate with plateau cells as responsive
        
    elif num_categories == "imm":
        real_behaviors = real_behaviors.replace({
            "N": "NI",              # label nonresponsive cells as non-immediate
            "G": "NI",              # label gradual cells as non-immediate
            "D": "NI",              # label delayed cells as non-immediate
            "I": "I",               # label immediate cells as immediate
            "Id": "I"})             # label immediate with plateau cells as immediate
    
    elif num_categories == 3:
        real_behaviors = real_behaviors.replace({
            "N": "N",               # keep nonresponsive category unchanged
            "G": "L",               # label gradual cells as long-term
            "D": "L",               # label delayed cells as long-term
            "I": "I",               # label immediate cells as immediate
            "Id": "I"})             # label immediate with plateau cells as immediate
        
    # print(real_behaviors)
    
    # # retrieve histogram bin edges
    # bin_edges_dict = {}
    # for data_type in ["avg", "SD", "AUC"]:
    #     cols = [f"{feature} ({data_type})" for feature in features]
    #     data = averages_df[cols]
    #     # print(f"{data_type}: min = {data.min()}")
    #     # print(f"{data_type}: max = {data.max()}")
    #     min_val = min(data.min())
    #     max_val = max(data.max())
    #     bin_edges_dict[data_type] = [min_val, max_val]
    # # print(bin_edges)
    
    for n, (location, location_df) in enumerate(locations_dict_full.items()):
        times         = list(location_df.columns)
        
        # pull min/max prestim values
        prestim_traces = location_df.loc[:, : stim_time]
        # print(prestim_traces)
        max_prestim    = round(max(prestim_traces.max()) + 0.05, 1)
        min_prestim    = round(min(prestim_traces.min()) - 0.05, 1)
        # print(f"Pre-stim range ({location}): [{min_prestim}, {max_prestim}]")
        
        # define inset limits
        ins_x1, ins_x2, ins_y1, ins_y2 = 0, stim_time, min_prestim, max_prestim
        
        # plot zoomed inset axis
        axs_inset = inset_axes(axs[0, n], width = "30%", height = "15%", loc = "upper right", borderpad = 1.5)
        
        for behavior, behavior_name in behavior_keys.items():
            
            # ROW 1: FEATURE TRACES
            cell_idxs = real_behaviors.index[real_behaviors == behavior].tolist()
            # print(f"{behavior} ({len(cell_idxs)}): {cell_idxs}")
            
            # retrieve trace values and color
            traces = location_df.loc[cell_idxs]
            color  = behavior_colors[behavior]
            
            # plot feature traces
            axs[0, n].plot(times, traces.T, linewidth = 0.5, color = color, alpha = 0.6)
            
            # set axes labels
            axs[0, n].set_xlabel("Time (min)", fontsize = 14)
            trace_ylabel = trace_ylabel_dict[location]
            axs[0, n].set_ylabel(trace_ylabel, fontsize = 14)
            
            # plot traces in inset
            axs_inset.plot(times, traces.T, linewidth = 0.5, color = color, alpha = 0.6)
            axs_inset.set_xlim(ins_x1, ins_x2)
            axs_inset.set_ylim(ins_y1, ins_y2)
            axs_inset.set_xticks([ins_x1, ins_x2])
            axs_inset.set_yticks([ins_y1, ins_y2])
            
            
        # ROWS 2-4: HISTOGRAM OF AVERAGES, SD, AUC
        for i, data_type in enumerate(["avg", "SD", "AUC"]):
            colors = list(behavior_colors.values())
            
            # retrieve data
            column    = f"{location} ({data_type})"
            data      = averages_df[column]
            # print(f"{location}: {avg_data}")
            max_bin   = data.max()
            min_bin   = data.min()
            hist_bins = np.linspace(min_bin, max_bin, 51)
            
            array     = []
            
            for behavior, behavior_name in behavior_keys.items():
                cell_idxs = real_behaviors.index[real_behaviors == behavior].tolist()
                behavior_data = data.loc[cell_idxs]
                array.append(behavior_data)
            
            # plot histogram
            axs[i + 1, n].hist(array, bins = hist_bins, stacked = True, density = True, color = colors)
            
            # set axes ticks and labels
            axs[i + 1, n].set_xlabel(f"{trace_ylabel_dict[location]} ({data_type})", fontsize = 14)
            # axs[i + 1, n].set_ylim(0, 800)
            if n == 0:
                axs[i + 1, n].set_ylabel("Density (cells)", fontsize = 14)
            else:
                axs[i + 1, n].set_yticklabels([])
            
            
    # add row and column labels
    pad = 5
    for ax, col in zip(axs[0], col_names):
        ax.annotate(col, xy = (0.5, 1), xytext = (0, pad),
                    xycoords = 'axes fraction', textcoords = 'offset points',
                    fontsize = 20, ha = 'center', va = 'baseline')

    for ax, row in zip(axs[:,0], row_names):
        ax.annotate(row, xy = (0, 0.5), xytext = (-ax.yaxis.labelpad - pad, 0),
                    xycoords = ax.yaxis.label, textcoords = 'offset points',
                    fontsize = 20, ha = 'right', va = 'center', rotation = 90)
    
    # # create custom legend
    # leg_lines       = [Line2D([0], [0], color = line_color, lw = 0.5) for line_color in list(behavior_colors.values())]
    # leg_labels      = [behavior_name for behavior_name in list(behavior_keys.values())]
    # fig.legend(leg_lines, leg_labels, loc = "upper right", bbox_to_anchor = (1.05, 1), title = "Original classification", title_fontsize = 14, fontsize = 12)
            
    plt.tight_layout()
    plt.savefig(f"classified_traces_{formatted_date}", dpi = 700)
    plt.show
    
    
#%% functions to train and run SVM

def run_SVM_predictor(location_averages_df, grid = True, feat_select = None, num_categories = 5):
    # extract features (X) and targets (y)
    X_SVM_df    = location_averages_df.drop(columns = ["Behavior"])
    
    if feat_select != None:
        X_SVM_df = X_SVM_df[feat_select]
        
    if num_categories == 5:
        y_SVM_df    = location_averages_df["Behavior"]
    
    elif num_categories == 2:
        y_SVM_df = location_averages_df["Behavior"].replace({
            "N": "N",               # keep nonresponsive category unchanged
            "G": "R",               # label gradual cells as responsive
            "D": "R",               # label delayed cells as responsive
            "I": "R",               # label immediate cells as responsive
            "Id": "R"})             # label immediate with plateau cells as responsive
        
    elif num_categories == "imm":
        y_SVM_df = location_averages_df["Behavior"].replace({
            "N": "NI",               # label nonresponsive cells as non-immediate
            "G": "NI",               # label gradual cells as non-immediate
            "D": "NI",               # label delayed cells as non-immediate
            "I": "I",               # label immediate cells as immediate
            "Id": "I"})             # label immediate with plateau cells as immediate
        
    elif num_categories == 3:
        y_SVM_df = location_averages_df["Behavior"].replace({
            "N": "N",               # keep nonresponsive category unchanged
            "G": "L",               # label gradual cells as long-term
            "D": "L",               # label delayed cells as long-term
            "I": "I",               # label immediate cells as immediate
            "Id": "I"})             # label immediate with plateau cells as immediate
    
    print(y_SVM_df.value_counts())
    
    # scale data
    scaler      = MinMaxScaler()
    scaled_data = scaler.fit_transform(X_SVM_df)
    X_SVM_df    = pd.DataFrame(scaled_data, columns = X_SVM_df.columns, index = X_SVM_df.index)
    
    # train and optimize SVM
    if grid:
        svm_model, best_params, best_score = test_SVM_params(X_SVM_df, y_SVM_df, random = False, param_grid = test_param_grid)
    else:
        svm_model, best_params, best_score = test_SVM_params(X_SVM_df, y_SVM_df, random = True, param_dist = test_param_dist)
    # print(f"{best_params}: {best_score}")
    
    # run SVM
    y_pred = svm_model.predict(X_SVM_df)
    
    # update results
    results_df = pd.DataFrame(index = location_averages_df.index, columns = ["Actual", "Predicted"])
    results_df.loc[:, "Actual"] = y_SVM_df
    results_df.loc[:, "Predicted"] = y_pred
    
    return svm_model, best_params, results_df


def run_SVM_predictor_RFA(location_averages_df):
    """
    Perform feature selection using RecursiveFeatureAddition.

    Parameters:
    ----------
    location_averages_df : pd.DataFrame
        DataFrame containing features and target values. The target column should be named 'Behavior'.

    Returns:
    -------
    selected_features : Index
        The features selected by RecursiveFeatureAddition.
    """
    # extract features (X) and targets (y)
    X_SVM_df = location_averages_df.drop(columns=["Behavior"])
    y_SVM_df = location_averages_df["Behavior"]
    
    # split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_SVM_df, y_SVM_df)
    
    # scale features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_df = pd.DataFrame(X_train_scaled, columns = X_train.columns, index = X_train.index)
    X_test_df = pd.DataFrame(X_test_scaled, columns = X_test.columns, index = X_test.index)
    # print("X_train_scaled shape:", X_train_df.shape)
    # print("X_train_scaled columns:", X_train_df.columns)
    # display(X_train_df)
    
    # create recursive feature addition selector
    selector = RecursiveFeatureAddition(
        estimator = SVC(),
        scoring = 'accuracy',                                                   # use accuracy to evaluate the model
        cv = 5                                                                  # number of cross-validation folds
    )
    
    # fit feature selector
    try:
        selector.fit(X_train_df, y_train)
        selected_features = X_train_df.columns[selector.get_support()]
        print("Selected Features:", selected_features)
    except ValueError as e:
        print("ValueError during fitting:", e)
    except Exception as e:
        print("Error during fitting:", e)
        
    # selector.fit(X_train_df, y_train)
    
    # # get selected features
    # selected_features = X_train_df.columns[selector.get_support()]
    
    # evaluate selected features
    X_train_selected = selector.transform(X_train_df)
    X_test_selected = selector.transform(X_test_df)
    
    svm_model = SVC()                                                           # create new SVM model for evaluation
    svm_model.fit(X_train_selected, y_train)
    
    y_pred = svm_model.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Selected Features: {selected_features}")
    print(f"Model Accuracy with Selected Features: {accuracy}")

    return selected_features


def run_SVM_predictor_RFE(location_averages_df, SVM_params = None):
    # extract features (X) and targets (y)
    X_SVM_df = location_averages_df.drop(columns=["Behavior"])
    y_SVM_df = location_averages_df["Behavior"]
    
    # split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_SVM_df, y_SVM_df)
    
    # scale features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_df = pd.DataFrame(X_train_scaled, columns = X_train.columns, index = X_train.index)
    X_test_df = pd.DataFrame(X_test_scaled, columns = X_test.columns, index = X_test.index)
    # print("X_train_scaled shape:", X_train_df.shape)
    # print("X_train_scaled columns:", X_train_df.columns)
    # display(X_train_df)
    
    # linear kernel
    if SVM_params == None:
        # create RFE feature selector
        model = SVC(kernel = "linear")
        selector = RFE(estimator = model, n_features_to_select = None, step = 1)
        
        # fit feature selector
        selector.fit(X_train_df, y_train)
        
        # get selected features
        selected_features = list(X_SVM_df.columns[selector.support_])
        
        # transform datasets to selected features
        X_train_select = selector.transform(X_train_df)
        X_test_select  = selector.transform(X_test_df)
        
        # create and evaluate SVM model
        svm_model = SVC(kernel="linear")
        svm_model.fit(X_train_select, y_train)
        
        y_pred = svm_model.predict(X_test_select)
        accuracy = accuracy_score(y_test, y_pred)
    
    # nonlinear kernel
    else:
        selector = SelectKBest(score_func = chi2, k = 5)
        X_train_select = selector.fit_transform(X_train_df, y_train)
        X_test_select  = selector.transform(X_test_df)
        
        selected_features = list(X_SVM_df.columns[selector.get_support()])
    
        # create and evaluate SVM model
        svm_model = SVC(**SVM_params)
        svm_model.fit(X_train_select, y_train)
        
        y_pred = svm_model.predict(X_test_select)
        accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Selected Features: {selected_features}")
    print(f"Model Accuracy with Selected Features: {accuracy}")
    
    return selected_features


#%% functions to visualize results

def plot_prestim_traces(subcluster_traces_smooth, locations_dict, results_df, treatment = "all", stim_time = 30, num_categories = 5):
    """
    USE ONLY IF remove_Ic = False.
    Plots traces for comparison between original and predicted behaviors:
        One subplot for each behavior type.
        Each subplot contains traces for all cells predicted to display that behavior by the SVM.
        Each trace is colored according to the original behavior assigned by eye.

    Parameters
    ----------
    subcluster_traces_smooth : dict, optional
        Dictionary of interpolated/smoothed ratio time trace values for cells in each subcluster within clusters for all treatments.
    results_dict : dict
        Summary of actual and predicted results per cell for each kernel tested.
    treatment : str, optional
        Treatment of interest for plotting.  The default is "all".

    Returns
    -------
    all_traces_df : pd.DataFrame
        Timecourse values for all cells in all treatments, clusters, and subclusters in subcluster_traces_smooth flattened into one df.

    """
    
    colors          = ["#DC143C", "#FF6F61", "indigo", "dodgerblue", "grey"]
    color_palette   = sns.color_palette(colors)
    behavior_colors_dict = {5: {"I": color_palette[0], "Id": color_palette[1], "G": color_palette[2], "D": color_palette[3], "N": color_palette[4]},
                            3: {"I": color_palette[0], "L": color_palette[3], "N": color_palette[4]},
                            2: {"R": color_palette[0], "N": color_palette[4]},
                            "imm": {"I": color_palette[0], "NI": color_palette[4]}}
    behavior_keys_dict   = {5: {"I": "Immediate", "Id": "Immediate with decay", "G": "Gradual", "D": "Delayed", "N": "Nonresponsive"},
                            3: {"I": "Immediate", "L": "Long-term", "N": "Nonresponsive"},
                            2: {"R": "Responsive", "N": "Nonresponsive"},
                            "imm": {"I": "Immediate", "NI": "Non-immediate"}}
    
    behavior_colors = behavior_colors_dict[num_categories]
    behavior_keys   = behavior_keys_dict[num_categories]
    # print(behavior_colors)
    # print(behavior_keys)
    behavior_rev    = {v: k for k, v in behavior_keys.items()}
    
    # flatten trace data
    all_traces_df   = flatten_trace_df(subcluster_traces_smooth)
    times           = list(all_traces_df.columns)
    
    # initialize figure
    if num_categories != "imm":
        fig, axs        = plt.subplots(1, num_categories, figsize = (5 + (5 * num_categories), 10), sharex = True, sharey = True)
    else:
        fig, axs        = plt.subplots(1, 2, figsize = (5 + (5 * 2), 10), sharex = True, sharey = True)
    
    for n, (SVM_behavior, behavior_name) in enumerate(behavior_keys.items()):
        behavior_traces = pd.DataFrame()
        num_cells       = 0
        
        # define inset limits
        ins_x1, ins_x2, ins_y1, ins_y2 = 0, stim_time, 0.5, 1.5
        # print(f"{treatment} ({behavior_name}): Max y-axis value = {axs[n].get_ylim()[1]}")
        
        # plot zoomed inset axis
        # inset_width   = str(int(100 - ((stim_time / axs[n].get_xlim()[1] * 100) + 20))) + "%"
        # print(f"{treatment} ({behavior_name}: Inset width = {inset_width}")
        axs_inset = inset_axes(axs[n], width = "50%", height = "25%", loc = "upper right", borderpad = 1.5)
        
        for orig_behavior, orig_behavior_name in behavior_keys.items():
            mask    = (results_df["Predicted"] == SVM_behavior) & (results_df["Actual"] == orig_behavior)
            indices = results_df.index[mask].tolist()
            # print(f"Cells predicted to be {behavior_name}: {indices}")
            # print(f"Cells predicted to be {behavior_name}: {len(indices)}")
            if treatment != "all":
                indices     = [index for index in indices if treatment in index]
            
            traces          = all_traces_df.loc[indices]
            # display(traces)
            behavior_traces = pd.concat([behavior_traces, traces], axis = 0)
            
            # plot traces
            color           = behavior_colors[orig_behavior]
            # print(f"Plotting {orig_behavior} with color {color}")
            axs[n].plot(times, traces.T, linewidth = 0.5, color = color)
            
            # plot traces in inset
            axs_inset.plot(times, traces.T, linewidth = 0.5, color = color)
            axs_inset.set_xlim(ins_x1, ins_x2)
            axs_inset.set_ylim(ins_y1, ins_y2)
            axs_inset.set_xticks([ins_x1, ins_x2])
            axs_inset.set_yticks([ins_y1, ins_y2])
            axs_inset.tick_params(axis = "both", labelsize = 14)
        
        num_cells   = behavior_traces.shape[0]
        
        # plot averages
        axs[n].plot(times, behavior_traces.mean(axis = 0), color = "black", linewidth = 1.5)
        
        # set axes labels and title
        title_color = behavior_colors[SVM_behavior]
        axs[n].set_title(f"{behavior_name} (n = {num_cells})", fontsize = 20, color = title_color)
        # axs[n].set_xlabel("Time (min)", fontsize = 14)
        # axs[n].set_ylabel("Nuclear Relish fraction (fold change)", fontsize = 14)
        axs[n].tick_params(axis = "both", labelsize = 14)
        
        # set y-axis limits and ticks
        axs[n].set_ylim(0.6, 2.0)
        axs[n].set_yticks([0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
        
        # highlight pre- and post-stim times
        y_min, y_max = axs[n].get_ylim()
        prestim  = patches.Rectangle((0, 0.6), 30, 0.025, color = "black")
        hatch = patches.Rectangle((30, 0.6), times[-1] - 30, 0.025, edgecolor="black", facecolor="none", hatch='///')
        axs[n].add_patch(prestim)
        axs[n].add_patch(hatch)
        
        # zoom in on pre-stim times
        # axs[n].axvline(x = stim_time, color = "grey", linestyle = "dashed", linewidth = 0.5)
        # mark_inset(axs[n], axs_inset, loc1 = 2, loc2 = 4, fc = "none", ec = "0.5")
    
    fig.supxlabel("Time (min)", fontsize = 16)
    fig.text(0, 0.5, 'Nuclear Relish fraction (fold change)', va = 'center', ha = 'center', fontsize = 16, rotation = 'vertical')
    
    # # create custom legend
    # leg_lines       = [Line2D([0], [0], color = line_color, lw = 0.5) for line_color in list(behavior_colors.values())]
    # leg_labels      = [behavior_name for behavior_name in list(behavior_keys.values())]
    # # print(leg_lines)
    # fig.legend(leg_lines, leg_labels, loc = "upper right", bbox_to_anchor = (1.05, 1), title = "Original classification", title_fontsize = 16, fontsize = 14)
    
    treatment_cells = len([cell for cell in list(results_df.index) if treatment in cell])
    total_cells     = results_df.shape[0]
    # if treatment == "all":
    #     plt.suptitle(f"SVM behavior classification (n = {total_cells})", fontsize = 20)
    # else:
    #     plt.suptitle(f"SVM behavior classification ({treatment}, n = {treatment_cells})", fontsize = 20)
    
    plt.tight_layout()
    plt.savefig(f"prestim_traces_{num_categories}-cats_{formatted_date}", dpi = 700)
    plt.show()
        
    return all_traces_df


def percent_behaviors(subcluster_traces_smooth, predictor_SVM_results_df, num_categories = 5, plot = False):
    colors          = ["#DC143C", "#FF6F61", "indigo", "dodgerblue", "grey"]
    color_palette   = sns.color_palette(colors)
    
    tmts = [tmt for tmt in subcluster_traces_smooth]
    behavior_colors_dict = {5: {"I": color_palette[0], "Id": color_palette[1], "G": color_palette[2], "D": color_palette[3], "N": color_palette[4]},
                            3: {"I": color_palette[0], "L": color_palette[3], "N": color_palette[4]},
                            2: {"R": color_palette[0], "N": color_palette[4]},
                            "imm": {"I": color_palette[0], "NI": color_palette[4]}}
    behavior_keys_dict   = {5: {"I": "Immediate", "Id": "Immediate with decay", "G": "Gradual", "D": "Delayed", "N": "Nonresponsive"},
                            3: {"I": "Immediate", "L": "Long-term", "N": "Nonresponsive"},
                            2: {"R": "Responsive", "N": "Nonresponsive"},
                            "imm": {"I": "Immediate", "NI": "Non-immediate"}}
    
    behavior_colors = behavior_colors_dict[num_categories]
    behavior_keys   = behavior_keys_dict[num_categories]
    behavior_rev    = {v: k for k, v in behavior_keys.items()}
    
    col_names = ["# cells"] + [behavior for behavior in behavior_keys]
    
    # print(treatments)
    # print(behaviors)
    # print(col_names)
    
    act_df = pd.DataFrame(index = tmts, columns = col_names)
    pred_df = pd.DataFrame(index = tmts, columns = col_names)
    
    for tmt in tmts:
        tmt_cells = predictor_SVM_results_df.index[predictor_SVM_results_df.index.str.contains(tmt)].tolist()
        tot_tmt_cells = len(tmt_cells)
        act_df.loc[tmt, "# cells"] = tot_tmt_cells
        pred_df.loc[tmt, "# cells"] = tot_tmt_cells
        
        tmt_data = predictor_SVM_results_df.loc[tmt_cells]
        # print(tmt_data)
        
        for behavior in col_names:
            if behavior == "# cells":
                continue
            
            act_data = tmt_data[tmt_data['Actual'] == behavior]
            sum_act_cells = act_data.shape[0]
            pred_data = tmt_data[tmt_data['Predicted'] == behavior]
            sum_pred_cells = pred_data.shape[0]
            # print(tmt, behavior, sum_cells)
            percent_act_cells = (sum_act_cells / tot_tmt_cells) * 100
            percent_pred_cells = (sum_pred_cells / tot_tmt_cells) * 100
            
            act_df.loc[tmt, behavior] = percent_act_cells
            pred_df.loc[tmt, behavior] = percent_pred_cells
        
    if plot is not False:
        color_list = list(behavior_colors.values())
        cols_to_plot = act_df.columns[1:]
        
        if plot == "vert":
            ax = act_df[cols_to_plot].plot.bar(stacked=True, figsize=(6, 10), color = color_list)
            pred_df[cols_to_plot].plot.bar(stacked=True, figsize=(6, 10), color = color_list)
        
            plt.xlabel(r'[PGN] ($\mu$g/mL)', fontsize = 14)
            plt.ylabel('% cells', fontsize = 14)
            plt.xticks(rotation = 0)
            ax.tick_params(axis = "both", labelsize = 12)
            plt.title('Distribution of Relish dynamics', fontsize = 16)
            
            for i, tmt in enumerate(tmts):
                x = pred_df.columns[1:]
                y = pred_df.loc[tmt, x].values
                total_cells = pred_df.loc[tmt, "# cells"]
                
                ax.text(x = i, y = y.sum() + 1, s=f"({total_cells})", ha='center', va='bottom', fontsize=10, color='black', fontstyle = "italic")
            
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
            plt.show()
            
        elif plot == "hor":
            ax = percents_df[cols_to_plot].plot.barh(stacked=True, figsize=(10, 6), color = color_list)
        
            plt.ylabel(r'[PGN] ($\mu$g/mL)', fontsize = 14)
            plt.xlabel('% cells', fontsize = 14)
            plt.yticks(rotation = 0)
            ax.tick_params(axis = "both", labelsize = 12)
            plt.title('Distribution of Relish dynamics', fontsize = 16)
            
            for i, tmt in enumerate(tmts):
                y = pred_df.columns[1:]
                x = pred_df.loc[tmt, y].values
                total_cells = pred_df.loc[tmt, "# cells"]
                
                ax.text(y = i - 0.1, x = x.sum() + 2, s=f"({total_cells})", ha='center', va='bottom', fontsize=10, color='black', fontstyle = "italic", rotation = -90)
            
            plt.legend(bbox_to_anchor=(1, 1.02), loc='upper left')
            plt.show()
        
    return pred_df


#%% test calls

# # goodcomp1: 03.05, 03.21, 04.10 compiled data without cellular areas (8.30.24)
# goodcomp_locations_dict = extract_data(goodcomp_dict_intensities)
# goodcomp_locations_dict_process, goodcomp_location_averages_df = process_data(goodcomp_locations_dict, goodcomp_SVM_results_dict)
# goodcomp_predictor_SVM_model, goodcomp_predictor_SVM_best_params, goodcomp_predictor_SVM_results_df = run_SVM_predictor(goodcomp_location_averages_df)


# # goodcomp2: 03.05, 03.21, 04.10, 08.01 compiled data with cellular areas (09.24.24)
# goodcomp2_locations_dict_area = extract_data(goodcomp2_dict_intensities_area)
# goodcomp2_locations_dict_process_area, goodcomp2_location_averages_df_area = process_data(goodcomp2_locations_dict_area, goodcomp2_SVM_results_dict_noIc)
# goodcomp2_predictor_SVM_model_area, goodcomp2_predictor_SVM_best_params_area, goodcomp2_predictor_SVM_results_df_area = run_SVM_predictor(goodcomp2_location_averages_df_area)
# plot_prestim_traces(goodcomp2_subcluster_traces_div_smooth, goodcomp2_locations_dict_area, goodcomp2_predictor_SVM_results_df_area)
# goodcomp2_predictor_SVM_model_area_2cat, goodcomp2_predictor_SVM_best_params_area_2cat, goodcomp2_predictor_SVM_results_df_area_2cat = run_SVM_predictor(goodcomp2_location_averages_df_area, num_categories = 2)
# plot_prestim_traces(goodcomp2_subcluster_traces_div_smooth, goodcomp2_locations_dict_area, goodcomp2_predictor_SVM_results_df_area_2cat, num_categories = 2)
# goodcomp2_predictor_SVM_model_area_3cat, goodcomp2_predictor_SVM_best_params_area_3cat, goodcomp2_predictor_SVM_results_df_area_3cat = run_SVM_predictor(goodcomp2_location_averages_df_area, num_categories = 3)
# plot_prestim_traces(goodcomp2_subcluster_traces_div_smooth, goodcomp2_locations_dict_area, goodcomp2_predictor_SVM_results_df_area_3cat, num_categories = 3)
# goodcomp2_predictor_SVM_model_area_imm, goodcomp2_predictor_SVM_best_params_area_imm, goodcomp2_predictor_SVM_results_df_area_imm    = run_SVM_predictor(goodcomp2_location_averages_df_area, num_categories = "imm")
# plot_prestim_traces(goodcomp2_subcluster_traces_div_smooth, goodcomp2_locations_dict_area, goodcomp2_predictor_SVM_results_df_area_imm, num_categories = "imm")
# goodcomp2_predictor_SVM_best_params_area_dict = {
#     "All categories": goodcomp2_predictor_SVM_best_params_area,
#     "Responsive/nonresponsive": goodcomp2_predictor_SVM_best_params_area_2cat,
#     "Immediate/long-term/nonresponsive": goodcomp2_predictor_SVM_best_params_area_3cat,
#     "Immediate/non-immediate": goodcomp2_predictor_SVM_best_params_area_imm}
# goodcomp2_predictor_SVM_model_area_dict = {
#     "All categories": goodcomp2_predictor_SVM_model_area,
#     "Responsive/nonresponsive": goodcomp2_predictor_SVM_model_area_2cat,
#     "Immediate/long-term/nonresponsive": goodcomp2_predictor_SVM_model_area_3cat,
#     "Immediate/non-immediate": goodcomp2_predictor_SVM_model_area_imm}
# goodcomp2_predictor_SVM_results_df_area_dict = {
#     "All categories": goodcomp2_predictor_SVM_results_df_area,
#     "Responsive/nonresponsive": goodcomp2_predictor_SVM_results_df_area_2cat,
#     "Immediate/long-term/nonresponsive": goodcomp2_predictor_SVM_results_df_area_3cat,
#     "Immediate/non-immediate": goodcomp2_predictor_SVM_results_df_area_imm}

# # plot features
# plot_feature_traces(goodcomp2_locations_dict_area, goodcomp2_location_averages_df_area, goodcomp2_SVM_results_dict_noIc, goodcomp2_predictor_SVM_results_df_area)
# plot_feature_traces(goodcomp2_locations_dict_area, goodcomp2_location_averages_df_area, goodcomp2_SVM_results_dict_noIc, goodcomp2_predictor_SVM_results_df_area_3cat, num_categories = 3)
# plot_feature_traces(goodcomp2_locations_dict_area, goodcomp2_location_averages_df_area, goodcomp2_SVM_results_dict_noIc, goodcomp2_predictor_SVM_results_df_area_2cat, num_categories = 2)
# plot_feature_traces(goodcomp2_locations_dict_area, goodcomp2_location_averages_df_area, goodcomp2_SVM_results_dict_noIc, goodcomp2_predictor_SVM_results_df_area_imm, num_categories = "imm")


# # goodcomp3: resegmented 03.05, 03.21, 04.10, 08.01 compiled data with cellular areas (10.07.24)
# goodcomp3_locations_dict_area = extract_data(goodcomp3_dict_intensities_area)
# goodcomp3_locations_dict_process_area, goodcomp3_location_averages_df_area = process_data(goodcomp3_locations_dict_area, goodcomp3_SVM_results_dict_noIc)
# goodcomp3_predictor_SVM_model_area, goodcomp3_predictor_SVM_best_params_area, goodcomp3_predictor_SVM_results_df_area = run_SVM_predictor(goodcomp3_location_averages_df_area)
# plot_prestim_traces(goodcomp3_subcluster_traces_div_smooth, goodcomp3_locations_dict_area, goodcomp3_predictor_SVM_results_df_area)
# # goodcomp3_predictor_SVM_percents_df_area = percent_behaviors(goodcomp3_subcluster_traces_div_smooth, goodcomp3_predictor_SVM_results_df_area, plot = "vert")
# goodcomp3_predictor_SVM_model_area_2cat, goodcomp3_predictor_SVM_best_params_area_2cat, goodcomp3_predictor_SVM_results_df_area_2cat = run_SVM_predictor(goodcomp3_location_averages_df_area, num_categories = 2)
# plot_prestim_traces(goodcomp3_subcluster_traces_div_smooth, goodcomp3_locations_dict_area, goodcomp3_predictor_SVM_results_df_area_2cat, num_categories = 2)
# # goodcomp3_predictor_SVM_percents_df_area_2cat = percent_behaviors(goodcomp3_subcluster_traces_div_smooth, goodcomp3_predictor_SVM_results_df_area_2cat, plot = "vert", num_categories = 2)
# goodcomp3_predictor_SVM_model_area_3cat, goodcomp3_predictor_SVM_best_params_area_3cat, goodcomp3_predictor_SVM_results_df_area_3cat = run_SVM_predictor(goodcomp3_location_averages_df_area, num_categories = 3)
# plot_prestim_traces(goodcomp3_subcluster_traces_div_smooth, goodcomp3_locations_dict_area, goodcomp3_predictor_SVM_results_df_area_3cat, num_categories = 3)
# # goodcomp3_predictor_SVM_percents_df_area_3cat = percent_behaviors(goodcomp3_subcluster_traces_div_smooth, goodcomp3_predictor_SVM_results_df_area_3cat, plot = "vert", num_categories = 3)
# goodcomp3_predictor_SVM_model_area_imm, goodcomp3_predictor_SVM_best_params_area_imm, goodcomp3_predictor_SVM_results_df_area_imm    = run_SVM_predictor(goodcomp3_location_averages_df_area, num_categories = "imm")
# plot_prestim_traces(goodcomp3_subcluster_traces_div_smooth, goodcomp3_locations_dict_area, goodcomp3_predictor_SVM_results_df_area_imm, num_categories = "imm")
# # goodcomp3_predictor_SVM_percents_df_area_imm = percent_behaviors(goodcomp3_subcluster_traces_div_smooth, goodcomp3_predictor_SVM_results_df_area_imm, plot = "vert", num_categories = "imm")
# goodcomp3_predictor_SVM_best_params_area_dict = {
#     "All categories": goodcomp3_predictor_SVM_best_params_area,
#     "Responsive/nonresponsive": goodcomp3_predictor_SVM_best_params_area_2cat,
#     "Immediate/long-term/nonresponsive": goodcomp3_predictor_SVM_best_params_area_3cat,
#     "Immediate/non-immediate": goodcomp3_predictor_SVM_best_params_area_imm}
# goodcomp3_predictor_SVM_model_area_dict = {
#     "All categories": goodcomp3_predictor_SVM_model_area,
#     "Responsive/nonresponsive": goodcomp3_predictor_SVM_model_area_2cat,
#     "Immediate/long-term/nonresponsive": goodcomp3_predictor_SVM_model_area_3cat,
#     "Immediate/non-immediate": goodcomp3_predictor_SVM_model_area_imm}
# goodcomp3_predictor_SVM_results_df_area_dict = {
#     "All categories": goodcomp3_predictor_SVM_results_df_area,
#     "Responsive/nonresponsive": goodcomp3_predictor_SVM_results_df_area_2cat,
#     "Immediate/long-term/nonresponsive": goodcomp3_predictor_SVM_results_df_area_3cat,
#     "Immediate/non-immediate": goodcomp3_predictor_SVM_results_df_area_imm}
# # goodcomp3_predictor_SVM_percents_df_area_dict = {
# #     "All categories": goodcomp3_predictor_SVM_percents_df_area,
# #     "Responsive/nonresponsive": goodcomp3_predictor_SVM_percents_df_area_2cat,
# #     "Immediate/long-term/nonresponsive": goodcomp3_predictor_SVM_percents_df_area_3cat,
# #     "Immediate/non-immediate": goodcomp3_predictor_SVM_percents_df_area_imm}

# # plot features
# plot_feature_traces(goodcomp3_locations_dict_area, goodcomp3_location_averages_df_area, goodcomp3_SVM_results_dict_noIc, goodcomp3_predictor_SVM_results_df_area)
# plot_feature_traces(goodcomp3_locations_dict_area, goodcomp3_location_averages_df_area, goodcomp3_SVM_results_dict_noIc, goodcomp3_predictor_SVM_results_df_area_3cat, num_categories = 3)
# plot_feature_traces(goodcomp3_locations_dict_area, goodcomp3_location_averages_df_area, goodcomp3_SVM_results_dict_noIc, goodcomp3_predictor_SVM_results_df_area_2cat, num_categories = 2)
# plot_feature_traces(goodcomp3_locations_dict_area, goodcomp3_location_averages_df_area, goodcomp3_SVM_results_dict_noIc, goodcomp3_predictor_SVM_results_df_area_imm, num_categories = "imm")