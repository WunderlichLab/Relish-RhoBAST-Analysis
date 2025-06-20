# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 08:12:43 2024

Name: Emma Rits
Date: Friday, August 30, 2024
Description: Cell fate predictor SVM

"""


#%% import packages, functions, and data

# packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.integrate import trapezoid
from collections import Counter
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler


# current date
import datetime
current_date   = datetime.date.today()
formatted_date = current_date.strftime("%Y-%m-%d")


# functions
# from ratiotimetrace_clustering_compiled import save_data, import_data, fillna_custom, dataframe_treat, cull_nans, smooth
# from trace_behavior_sorting_SVM import test_SVM_params, test_param_grid, test_param_dist
# from cell_trace_classifying_interface import flatten_trace_df

from step9_Relish_Trace_PreProcessing_python import save_data, import_data, fillna_custom, dataframe_treat, cull_nans, smooth, flatten_trace_df
from step11_Relish_Trace_ClassifierSVM_python import test_SVM_params, param_grid, param_dist


#%% functions to prep data

def extract_data(dict_intensities, stim_time = 30, sg_order = 2, sg_factor = 5, area = True):
    """
    Extracts Relish traces and cellular area values from imaging data.
    
    Parameters
    ----------
    dict_intensities : dict
        Dictionary of Relish intensities and cellular area (Step 7 output).
    
    Returns
    -------
    locations_dict : dict
        Dictionary of dfs containing:
            Nuclear, cytoplasmic, total, and ratio Relish timecourse values for all treatments compiled for all dates
            Nuclear, cytoplasmic, and total area timecourse values for all treatments compiled for all dates
            
    Raises
    ------
    ValueError
        Occurs if cell indices are not the same for both nuclear and cytoplasmic data.
        Occurs if cytoplasmic and nuclear dataframes are equal.
    
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
            
            # compile subsets defined above
            keys_subsets      = [keys_PGNsubset, keys_1Xsubset, keys_10Xsubset, keys_100Xsubset]
            dict_subsets      = [dict_PGNsubset, dict_1Xsubset, dict_10Xsubset, dict_100Xsubset]
            subsets_dict      = {}
        
            for i in range(4): # looping through all four treatments    
                treatment         = treatments[i]    
                key_subset        = keys_subsets[i]
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
                    framenames        = [k for k in key_subset if viewframe in k]
                    
                    if len(framenames) != 0:
                        tif_key = framenames[0]
                        # print(tif_key)
                    else:
                        continue
                    
                    dct               = dict_subset[tif_key]
                    loc_reps          = []                                      # list of values across all three replicates
                    cell_name_reps    = []                                      # list of cell names across all three replicates
        
                    for key in list(dct[location].keys()): # looping through all cells
                        data          = []                                      # list of values across current replicate
                        
                        value         = dct[location][key]                      # list of values for current cell
                        data         += value
                        
                        cell_no       = int(key.strip("Cell "))                 # cell number of current cell
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
                temp_df           = pd.DataFrame(reps_dict)                     # convert to df
                temp_df           = fillna_custom(temp_df)                      # replace NaNs
                dates_dict        = pd.DataFrame.to_dict(temp_df)               # convert back to dict
                
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
    """
    Processes (flattens and truncates) Relish and area data and calculates average, SD, and AUC values for each feature for each cell.

    Parameters
    ----------
    locations_dict : dict
        Dictionary of dfs containing:
            Nuclear, cytoplasmic, total, and ratio Relish timecourse values for all treatments compiled for all dates.
            Nuclear, cytoplasmic, and total area timecourse values for all treatments compiled for all dates.
    SVM_results_dict : dict
        Summary of actual and predicted results per cell for each kernel tested (Step 11 output).
    stim_time : int, optional
        Time of PGN stimulus.  The default is 30.
    trunc : bool, optional
        Whether or not to extract only the pre-stimulus data.  The default is True.

    Raises
    ------
    ValueError
        Occurs if cell names are not consistent between sub-dfs in locations_dict_process.

    Returns
    -------
    locations_dict_process : dict
        Dictionary of dfs containing:
            Nuclear, cytoplasmic, total, and ratio Relish timecourse values compiled for all dates and treatments for t ∈ [0, stim_time].
            Nuclear, cytoplasmic, and total area timecourse values compiled for all dates and treatments for t ∈ [0, stim_time].
    averages_df : df
        Average, SD, and AUC calculated on t ∈ [0, stim_time] for each feature for each cell.
        SVM behavior classifications are also noted for each cell.

    """
    locations = list(locations_dict.keys())
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
            
            # update location_df
            if location_df.empty:
                location_df = treatment_df
            else:
                location_df = pd.concat([location_df, treatment_df], axis = 0)
        
        locations_dict_process[location] = location_df
        
    # check to make sure cell names and order are consistent
    consistency, cell_names = check_cell_names(locations_dict_process)
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
        averages_df.loc[:, f"{location} (avg)"] = avg_data
        averages_df.loc[:, f"{location} (SD)"]  = sd_data
        
        # calculate area under the curve
        times = cell_data.columns.astype(float)
        auc_data = cell_data.apply(lambda cell: trapezoid(cell, x = times), axis = 1)
        averages_df.loc[:, f"{location} (AUC)"] = auc_data
    
    # update behavior column with SVM results
    for cell in averages_df.index:
        cell_behavior = SVM_results_dict["All cells"].loc[cell, "Predicted"]
        averages_df.loc[cell, "Behavior"] = cell_behavior
    
    return locations_dict_process, averages_df


def check_cell_names(locations_dict_process):
    """
    Checks that cell names (df indices) are consistent between sub-dfs in locations_dict_process.

    Parameters
    ----------
    locations_dict_process : dict
        Dictionary of dfs containing:
            Nuclear, cytoplasmic, total, and ratio Relish timecourse values compiled for all dates and treatments for t ∈ [0, stim_time].
            Nuclear, cytoplasmic, and total area timecourse values compiled for all dates and treatments for t ∈ [0, stim_time].

    Returns
    -------
    True if cell names are consistent.
    False if cell names are not consistent.

    """
    # use the indices of the "cyto" df as a reference
    reference_idx = locations_dict_process[list(locations_dict_process.keys())[0]].index
    
    # check if the other indices are the same
    if all (location_df.index.equals(reference_idx) for location_df in locations_dict_process.values()):
        return True, reference_idx
    else:
        return False
    

#%% functions to visualize features

def plot_feature_traces(locations_dict, averages_df, SVM_results_dict, predictor_SVM_results_df, num_categories = 5, stim_time = 30):
    """
    Plots traces, averages, SDs, and AUCs for each feature.
        Row 1:    Feature traces plotted for the entire timecourse (main figure) and pre-stimulus (inset).
        Rows 2-4: Histogram of pre-stimulus feature averages (row 2), SDs (row 3) and AUCs (row 4).

    Parameters
    ----------
    locations_dict : dict
        Dictionary of dfs containing:
            Nuclear, cytoplasmic, total, and ratio Relish timecourse values for all treatments compiled for all dates.
            Nuclear, cytoplasmic, and total area timecourse values for all treatments compiled for all dates.
    averages_df : df
        Average, SD, and AUC calculated on t ∈ [0, stim_time] for each feature for each cell.
        SVM behavior classifications are also noted for each cell.
    SVM_results_dict : dict
        Summary of actual and predicted results per cell for each kernel tested (Step 11 output).
    predictor_SVM_results_df : TYPE
        DESCRIPTION.
    num_categories : int/str, optional
        Number of behavior categories for predictor SVM results.  The default is 5.
        The options are:
            2:     Responsive (I, Id, G, D) and nonresponsive (N)
            3:     Immediate (I, Id), long-term (G, D), and nonresponsive (N)
            5:     Immediate (I), immediate with decay (Id), gradual (G), delayed (D), and nonresponsive (N)
            "imm": Immediate (I, Id) and non-immediate (G, D, N)
    stim_time : int, optional
        Time of PGN stimulus.  The default is 30.
    
    Raises
    ------
    ValueError
        Occurs if num_categories is not set to 2, 3, 5, or "imm".

    Returns
    -------
    None.

    """
    
    if num_categories not in [2, 3, 5, "imm"]:
        raise ValueError(f"{num_categories} is not a valid input for num_categories.  Must be set to 2, 3, 5, or 'imm'.")
    
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

    if num_categories == 2: # responsive/nonresponsive classification
        real_behaviors = real_behaviors.replace({
            "N": "N",                                                           # keep nonresponsive category unchanged
            "G": "R",                                                           # label gradual cells as responsive
            "D": "R",                                                           # label delayed cells as responsive
            "I": "R",                                                           # label immediate cells as responsive
            "Id": "R"})                                                         # label immediate with plateau cells as responsive
        
    elif num_categories == "imm": # immediate/non-immediate classification
        real_behaviors = real_behaviors.replace({
            "N": "NI",                                                          # label nonresponsive cells as non-immediate
            "G": "NI",                                                          # label gradual cells as non-immediate
            "D": "NI",                                                          # label delayed cells as non-immediate
            "I": "I",                                                           # label immediate cells as immediate
            "Id": "I"})                                                         # label immediate with plateau cells as immediate
    
    elif num_categories == 3: # immediate/long-term/nonresponsive classification
        real_behaviors = real_behaviors.replace({
            "N": "N",                                                           # keep nonresponsive category unchanged
            "G": "L",                                                           # label gradual cells as long-term
            "D": "L",                                                           # label delayed cells as long-term
            "I": "I",                                                           # label immediate cells as immediate
            "Id": "I"})                                                         # label immediate with plateau cells as immediate
        
    for n, (location, location_df) in enumerate(locations_dict_full.items()):
        times         = list(location_df.columns)
        
        # pull min/max prestim values
        prestim_traces = location_df.loc[:, : stim_time]
        max_prestim    = round(max(prestim_traces.max()) + 0.05, 1)
        min_prestim    = round(min(prestim_traces.min()) - 0.05, 1)
        
        # define inset limits
        ins_x1, ins_x2, ins_y1, ins_y2 = 0, stim_time, min_prestim, max_prestim
        
        # plot zoomed inset axis
        axs_inset = inset_axes(axs[0, n], width = "30%", height = "15%", loc = "upper right", borderpad = 1.5)
        
        for behavior, behavior_name in behavior_keys.items():
            
            # ROW 1: FEATURE TRACES
            cell_idxs = real_behaviors.index[real_behaviors == behavior].tolist()
            
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
            max_bin   = data.max()
            min_bin   = data.min()
            hist_bins = np.linspace(min_bin, max_bin, 51)
            
            array     = []
            
            for behavior, behavior_name in behavior_keys.items():
                cell_idxs = real_behaviors.index[real_behaviors == behavior].tolist()
                behavior_data = data.loc[cell_idxs]
                array.append(behavior_data)
            
            # plot histogram
            axs[i + 1, n].hist(array, bins = hist_bins, density = True, color = colors)
            
            # set axes ticks and labels
            axs[i + 1, n].set_xlabel(f"{trace_ylabel_dict[location]} ({data_type})", fontsize = 14)
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
    
        
    plt.tight_layout()
    plt.savefig(f"classified_traces_feature_histogram{formatted_date}", dpi = 700)
    plt.show
    
    
#%% functions to train and run SVM

def run_SVM_predictor(averages_df, feat_select = None, grid = True, num_categories = 5):
    """
    Runs predictor SVM (all features) on pre-stimulus Relish and area data.

    Parameters
    ----------
    averages_df : df
        Average, SD, and AUC calculated on t ∈ [0, stim_time] for each feature for each cell.
        SVM behavior classifications are also noted for each cell.
    feat_select : list, optional
        Optimized features selected by RFE.  The default is None.
    grid : bool, optional
        Whether to use GridSearch (True) or RandomizedSearch (False).  The default is True.
    num_categories : int/str, optional
        Number of behavior categories for predictor SVM results.  The default is 5.
        The options are:
            2:     Responsive (I, Id, G, D) and nonresponsive (N)
            3:     Immediate (I, Id), long-term (G, D), and nonresponsive (N)
            5:     Immediate (I), immediate with decay (Id), gradual (G), delayed (D), and nonresponsive (N)
            "imm": Immediate (I, Id) and non-immediate (G, D, N)

    Returns
    -------
    svm_model : svm._classes.SVC
        Trained SVM predictor model.
    best_params : TYPE
        DESCRIPTION.
    results_df : df
        Summary of actual and predicted results per cell.

    """
    
    # extract features (X) and targets (y)
    X_SVM_df    = averages_df.drop(columns = ["Behavior"])
    
    if feat_select != None:
        X_SVM_df = X_SVM_df[feat_select]
    
    if num_categories == 5:
        y_SVM_df    = averages_df["Behavior"]
    
    elif num_categories == 2:
        y_SVM_df = averages_df["Behavior"].replace({
            "N": "N",                                                           # keep nonresponsive category unchanged
            "G": "R",                                                           # label gradual cells as responsive
            "D": "R",                                                           # label delayed cells as responsive
            "I": "R",                                                           # label immediate cells as responsive
            "Id": "R"})                                                         # label immediate with plateau cells as responsive
        
    elif num_categories == "imm":
        y_SVM_df = averages_df["Behavior"].replace({
            "N": "NI",                                                          # label nonresponsive cells as non-immediate
            "G": "NI",                                                          # label gradual cells as non-immediate
            "D": "NI",                                                          # label delayed cells as non-immediate
            "I": "I",                                                           # label immediate cells as immediate
            "Id": "I"})                                                         # label immediate with plateau cells as immediate
        
    elif num_categories == 3:
        y_SVM_df = averages_df["Behavior"].replace({
            "N": "N",                                                           # keep nonresponsive category unchanged
            "G": "L",                                                           # label gradual cells as long-term
            "D": "L",                                                           # label delayed cells as long-term
            "I": "I",                                                           # label immediate cells as immediate
            "Id": "I"})                                                         # label immediate with plateau cells as immediate
    
    print(y_SVM_df.value_counts())
    
    # scale data
    scaler      = MinMaxScaler()
    scaled_data = scaler.fit_transform(X_SVM_df)
    X_SVM_df    = pd.DataFrame(scaled_data, columns = X_SVM_df.columns, index = X_SVM_df.index)
    
    # train and optimize SVM
    if grid:
        svm_model, best_params, best_score = test_SVM_params(X_SVM_df, y_SVM_df, random = False, param_grid = param_grid)
    else:
        svm_model, best_params, best_score = test_SVM_params(X_SVM_df, y_SVM_df, random = True, param_dist = param_dist)
    
    # run SVM
    y_pred = svm_model.predict(X_SVM_df)
    
    # update results
    results_df = pd.DataFrame(index = averages_df.index, columns = ["Actual", "Predicted"])
    results_df.loc[:, "Actual"] = y_SVM_df
    results_df.loc[:, "Predicted"] = y_pred
    
    return svm_model, best_params, results_df


def run_SVM_predictor_RFE(averages_df, num_categories = 5, cv_folds = 5, print = True):
    """
    Runs the SVM predictor with recursive feature elimination (RFE) to optimize feature input.

    Parameters
    ----------
    averages_df : df
        Average, SD, and AUC calculated on t ∈ [0, stim_time] for each feature for each cell.
        SVM behavior classifications are also noted for each cell.
    num_categories : int/str, optional
        Number of behavior categories for predictor SVM results.  The default is 5.
        The options are:
            2:     Responsive (I, Id, G, D) and nonresponsive (N)
            3:     Immediate (I, Id), long-term (G, D), and nonresponsive (N)
            5:     Immediate (I), immediate with decay (Id), gradual (G), delayed (D), and nonresponsive (N)
            "imm": Immediate (I, Id) and non-immediate (G, D, N)
    cv_folds : int, optional
        Number of cross-validation folds.  The default is 5.
    print : bool, optional
        Whether or not to print the selected features and model accuracy.  The default is True.

    Returns
    -------
    selected_features : list
        List of selected features.

    """
    
    # extract features (X) and targets (y)
    X_SVM_df                               = averages_df.drop(columns=["Behavior"])
    y_SVM_df                               = averages_df["Behavior"]
        
    if num_categories == 5:
        y_SVM_df                           = averages_df["Behavior"]
    
    elif num_categories == 2:
        y_SVM_df = averages_df["Behavior"].replace({
            "N": "N",                                                           # keep nonresponsive category unchanged
            "G": "R",                                                           # label gradual cells as responsive
            "D": "R",                                                           # label delayed cells as responsive
            "I": "R",                                                           # label immediate cells as responsive
            "Id": "R"})                                                         # label immediate with plateau cells as responsive
        
    elif num_categories == "imm":
        y_SVM_df = averages_df["Behavior"].replace({
            "N": "NI",                                                          # label nonresponsive cells as non-immediate
            "G": "NI",                                                          # label gradual cells as non-immediate
            "D": "NI",                                                          # label delayed cells as non-immediate
            "I": "I",                                                           # label immediate cells as immediate
            "Id": "I"})                                                         # label immediate with plateau cells as immediate
        
    elif num_categories == 3:
        y_SVM_df = averages_df["Behavior"].replace({
            "N": "N",                                                           # keep nonresponsive category unchanged
            "G": "L",                                                           # label gradual cells as long-term
            "D": "L",                                                           # label delayed cells as long-term
            "I": "I",                                                           # label immediate cells as immediate
            "Id": "I"})                                                         # label immediate with plateau cells as immediate
    
    # scale features
    scaler                                 = MinMaxScaler()
    X_SVM_df_scaled                        = pd.DataFrame(scaler.fit_transform(X_SVM_df), columns = X_SVM_df.columns, index = X_SVM_df.index)
    
    # split into training and test sets
    X_train_df, X_test_df, y_train, y_test = train_test_split(X_SVM_df_scaled, y_SVM_df)
    
    # create RFE feature selector using linear kernel
    model                                  = SVC(kernel = "linear")
    cv                                     = StratifiedKFold(n_splits = cv_folds, shuffle = True)
    selector                               = RFECV(estimator = model, step = 1, cv = cv, scoring = "accuracy")
    
    # fit feature selector
    selector.fit(X_train_df, y_train)
    
    # get selected features
    selected_features                      = list(X_SVM_df.columns[selector.support_])
    
    # transform datasets to selected features
    X_train_select                         = selector.transform(X_train_df)
    X_test_select                          = selector.transform(X_test_df)
    
    # create and evaluate SVM model
    svm_model                              = SVC(kernel="linear")
    svm_model.fit(X_train_select, y_train)
    
    y_pred                                 = svm_model.predict(X_test_select)
    accuracy                               = accuracy_score(y_test, y_pred)
    
    if print:
        print(f"Selected Features: {selected_features}")
        print(f"Model Accuracy with Selected Features: {accuracy}")
    
    return selected_features


def optimize_SVM_predictor_RFE(location_averages_df, num_categories = 5, n_runs = 100, threshold = 0.8):
    """
    Runs the SVM feature selection using RFE multiple times and identifies the most consistent features.

    Parameters:
    ----------
    location_averages_df : pd.DataFrame
        DataFrame containing the features and target values. The target column should be named 'Behavior'.
    num_categories : int/str, optional
        Number of behavior categories for predictor SVM results.  The default is 5.
        The options are:
            2:     Responsive (I, Id, G, D) and nonresponsive (N)
            3:     Immediate (I, Id), long-term (G, D), and nonresponsive (N)
            5:     Immediate (I), immediate with decay (Id), gradual (G), delayed (D), and nonresponsive (N)
            "imm": Immediate (I, Id) and non-immediate (G, D, N)
    n_runs : int, optional
        Number of iterations to run the feature selection process.  The default is 100.
    threshold : float, optional
        The frequency threshold for selecting features. Features selected in at least `threshold` fraction of runs are kept.
        The default is 0.8.
    
    Returns:
    -------
    threshold_features : list
        List of features that were selected most consistently across all runs.
    """
    
    selected_features_all_runs = []

    # run RFE feature selection n_runs times
    for _ in range(n_runs):
        # RFE for each run and collect selected features
        selected_features = run_SVM_predictor_RFE(location_averages_df, num_categories, print = False)
        selected_features_all_runs.append(selected_features)

    # flatten the list of selected features from all runs
    flat_features = [item for sublist in selected_features_all_runs for item in sublist]
    
    # count the frequency of each feature across all runs
    feature_counter = Counter(flat_features)

    # retrieve features that appear at higher than threshold frequency (default 80%)
    threshold_features = [feature for feature, count in feature_counter.items() if count / n_runs >= threshold]

    # Print out the most common features
    print(f"Features selected in at least {threshold*100}% of runs: {threshold_features}")

    return threshold_features


#%% functions to visualize results

def plot_prestim_traces(subcluster_traces_smooth, locations_dict, results_df, treatment = "all", stim_time = 30, num_categories = 5):
    """
    Plots traces for comparison between original and predicted behaviors:
        One subplot for each behavior type.
        Each subplot contains traces for all cells predicted to display that behavior by the SVM.
        Each trace is colored according to the behavior assigned by the classifier SVM (Step 11).

    Parameters
    ----------
    subcluster_traces_smooth : dict, optional
        Dictionary of interpolated/smoothed ratio time trace values for cells in each subcluster within clusters for all treatments.
    results_dict : dict
        Summary of actual and predicted results per cell for each kernel tested.
    treatment : str, optional
        Treatment of interest for plotting.  The default is "all".
    stim_time : int, optional
        Time of PGN stimulus.  The default is 30.    
    num_categories : int/str, optional
        Number of behavior categories for predictor SVM results.  The default is 5.
        The options are:
            2:     Responsive (I, Id, G, D) and nonresponsive (N)
            3:     Immediate (I, Id), long-term (G, D), and nonresponsive (N)
            5:     Immediate (I), immediate with decay (Id), gradual (G), delayed (D), and nonresponsive (N)
            "imm": Immediate (I, Id) and non-immediate (G, D, N)

    Returns
    -------
    None.

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
        axs_inset = inset_axes(axs[n], width = "50%", height = "25%", loc = "upper right", borderpad = 1.5)
        
        for orig_behavior, orig_behavior_name in behavior_keys.items():
            mask    = (results_df["Predicted"] == SVM_behavior) & (results_df["Actual"] == orig_behavior)
            indices = results_df.index[mask].tolist()
            if treatment != "all":
                indices     = [index for index in indices if treatment in index]
            
            traces          = all_traces_df.loc[indices]
            behavior_traces = pd.concat([behavior_traces, traces], axis = 0)
            
            # plot traces
            color           = behavior_colors[orig_behavior]
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
        
    fig.supxlabel("Time (min)", fontsize = 16)
    fig.text(0, 0.5, 'Nuclear Relish fraction (fold change)', va = 'center', ha = 'center', fontsize = 16, rotation = 'vertical')
    
    plt.tight_layout()
    plt.savefig(f"prestim_traces_{num_categories}-cats_{formatted_date}", dpi = 700)
    plt.show()


#%% # === MAIN LOOP ===

# file paths
all_data         = "/path/to/your/data/2025-01-01_datasetName/"
file_path_step9  = all_data + "Processed Traces"
file_path_step10 = all_data + "SVM Classifier"
file_path_step12 = all_data + "SVM Predictor"
dict_intensities_area    = import_data()
SVM_results_dict         = import_data(file_path_step10, "SVM_results_dict")
subcluster_traces_smooth = import_data(file_path_step9, "subcluster_traces_smooth")

# extract data
locations_dict_area = extract_data(dict_intensities_area)
locations_dict_process_area, location_averages_df_area = process_data(locations_dict_area, SVM_results_dict)

# run predictor SVM
predictor_SVM_model, predictor_SVM_best_params, predictor_SVM_results_df    = run_SVM_predictor(location_averages_df_area, num_categories = "imm")
plot_prestim_traces(subcluster_traces_smooth, locations_dict_area, predictor_SVM_results_df, num_categories = "imm")

# plot features
plot_feature_traces(locations_dict_area, location_averages_df_area, SVM_results_dict, predictor_SVM_results_df, num_categories = "imm")

# RFE analysis
threshold_features_RFE = optimize_SVM_predictor_RFE(location_averages_df_area, num_categories = "imm")
predictor_SVM_model_RFE, predictor_SVM_best_params_RFE, predictor_SVM_results_df_RFE = run_SVM_predictor(location_averages_df_area, feat_select = threshold_features_RFE, num_categories = "imm")
plot_prestim_traces(subcluster_traces_smooth, locations_dict_area, predictor_SVM_results_df_RFE, num_categories = "imm")
plot_feature_traces(locations_dict_area, location_averages_df_area, SVM_results_dict, predictor_SVM_results_df_RFE, num_categories = "imm")

save_data(file_path_step12, predictor_SVM_model, "predictor_SVM_model")
save_data(file_path_step12, predictor_SVM_results_df, "predictor_SVM_results_df")
save_data(file_path_step12, threshold_features_RFE, "threshold_features_RFE")
save_data(file_path_step12, predictor_SVM_model_RFE, "predictor_SVM_model_RFE")
save_data(file_path_step12, predictor_SVM_results_df_RFE, "predictor_SVM_results_df_RFE")
