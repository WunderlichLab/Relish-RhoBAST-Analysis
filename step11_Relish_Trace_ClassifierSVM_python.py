# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 10:52:01 2024

Name: Emma Rits
Date: Monday, July 15, 2024
Description: Trace behavior sorting SVM

"""


#%% import packages, functions, and data

# packages
import copy
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from scipy.stats import uniform, randint
import seaborn as sns
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

# current date
import datetime
current_date   = datetime.date.today()
formatted_date = current_date.strftime("%Y-%m-%d")

# functions
# from ratiotimetrace_clustering_compiled import save_data, import_data
# from cell_trace_classifying_interface import flatten_trace_df

from step9_Relish_Trace_PreProcessing_python import save_data, import_data
from step10_Relish_Trace_ClassifierGUI_python import flatten_trace_df


#%% functions to prepare data

def trace_descriptor_subset(dict_trace_descriptors, cell_categories_df):
    """
    Extracts subset of data from dict_trace_descriptors corresponding to cells classified using GUI.

    Parameters
    ----------
    dict_trace_descriptors : dict
        Dictionary of dfs containing:   Cluster   = Cluster/subcluster identity
                                        Max value = Maximum value of the Relish nuclear fraction (fold change relative to pre-stim)
                                        Max time  = Time at which max value occurs
                                        Half val  = Half maximum value
                                        Half time = Time at which fold change reaches half max
                                        Area      = Area under the fold change curve
                                        Max in    = Maximal rate of nuclear entry, provided as a list of the form [time, rate]
                                        Max out   = Maximal rate of nuclear exit, provided as a list of the form [time, rate]
                                        Peaks     = Number of peaks
                                        Local Max = Information about local maxima (including global max), provided as a list of lists of the form [time, value]
            (Source: Cruz et al., 2021)
            Must contain additional "Behavior" column: N  = Nonresponsive
                                                       G  = Gradual
                                                       D  = Delayed
                                                       Id = Immediate with decrease
                                                       I  = Immediate
    cell_categories_df : pd.DataFrame
        Dataframe listing the cells manually assigned to each behavior category
        
    Returns
    -------
    dict_subset : dict
        Subset of dict_trace_descriptors; same format, but only including the cells in cell_categories_df

    """
    
    treatments = [treatment for treatment in dict_trace_descriptors if treatment != "Averages"]
    behavior_keys = {"N": "Nonresponsive",
                     "D": "Delayed",
                     "G": "Gradual",
                     "I": "Immediate",
                     "Id": "Immediate with decay"}
            
    # convert cell_categories_df to dict
    cell_categories_dict = cell_categories_df.set_index("Category")["Cells"].to_dict()
    
    # generate list of cells to extract 
    cell_names_subset = set(cell for cells in cell_categories_dict.values() for cell in cells)
    
    # initialize dicts for data and cell names
    dict_subset     = dict.fromkeys(treatments)
    cell_names_dict = dict.fromkeys(treatments)
    
    for treatment in treatments:
        # retrieve cells in treatment
        treatment_df = dict_trace_descriptors[treatment]
        cell_names_treatment = [cell_name.split(" ", 1)[1] for cell_name in cell_names_subset if cell_name.split(" ", 1)[0] == treatment]
        cell_names_dict[treatment] = cell_names_treatment
        
        # extract data and update df_subset and dict_subset
        df_subset = treatment_df.loc[treatment_df.index.intersection(cell_names_treatment)]
        dict_subset[treatment] = df_subset
        
    # correct behavior classifications
    for behavior_name in cell_categories_dict:
        behavior = {v: k for k, v in behavior_keys.items()}.get(behavior_name)
        for cell_name in cell_categories_dict[behavior_name]:
            treatment, cell = cell_name.split(" ", 1)
            dict_subset[treatment].loc[cell, "Behavior"] = behavior
    
    return dict_subset


def prep_descriptor_vals_SVM(dict_trace_descriptors, key_params = True, drop_nans = True, weights_dict = None, scale_range = None):
    """
    Prepares and formats dict_trace_descriptors for use in SVM.

    Parameters
    ----------
    dict_trace_descriptors : dict
        Dictionary of dfs containing:   Cluster   = Cluster/subcluster identity
                                        Max value = Maximum value of the Relish nuclear fraction (fold change relative to pre-stim)
                                        Max time  = Time at which max value occurs
                                        Half val  = Half maximum value
                                        Half time = Time at which fold change reaches half max
                                        Area      = Area under the fold change curve
                                        Max in    = Maximal rate of nuclear entry, provided as a list of the form [time, rate]
                                        Max out   = Maximal rate of nuclear exit, provided as a list of the form [time, rate]
                                        Peaks     = Number of peaks
                                        Local Max = Information about local maxima (including global max), provided as a list of lists of the form [time, value]
            (Source: Cruz et al., 2021)
            Must contain additional "Behavior" column: N  = Nonresponsive
                                                       G  = Gradual
                                                       D  = Delayed
                                                       I  = Immediate
                                                       Id = Immediate with decrease
    key_params : bool, optional
        Whether to include only the "key" parameters (max value, max time, half max, half time, and area). The default is True.
    drop_nans : bool, optional
        Whether to exclude all cells with data containing NaNs. The default is True.
    weights_dict : dict, optional
        Dictionary of descriptors of interest with their corresponding weights. The default is None.
    
    Returns
    -------
    descriptor_vals_SVM_df : pd.DataFrame
        Compiled and modified version of dict_trace_descriptors.
    X_SVM_df : pd.DataFrame
        Feature matrix for SVM.
    y_SVM_df : pd.Series
        Target vector for SVM.

    """
    
    treatments             = [treatment for treatment in list(dict_trace_descriptors.keys()) if treatment != "Averages"]
    if key_params:
        descriptors            = ["Behavior",
                                  "Max Value",
                                  "Max Time",
                                  "Half Time",
                                  "Area"]
        
    else:
        descriptors            = ["Behavior",
                                  "Max Value",
                                  "Max Time",
                                  "Half Max",
                                  "Half Time",
                                  "Area",
                                  "Max In Time",
                                  "Max In Rate",
                                  "Max Out Time",
                                  "Max Out Rate",
                                  "Peaks"]
        
    # initialize df
    descriptor_vals_SVM_df = pd.DataFrame(columns = descriptors)
    
    for treatment in treatments:
        treatment_df       = dict_trace_descriptors[treatment]
        cells              = treatment_df.index.tolist()
        
        for index, row in treatment_df.iterrows():
            # add treatment to cell name
            cell_name                                             = treatment + " " + index
            
            # extract data
            descriptor_vals_SVM_df.loc[cell_name, "Behavior"]     = treatment_df.loc[index, "Behavior"]
            descriptor_vals_SVM_df.loc[cell_name, "Max Value"]    = treatment_df.loc[index, "Max Value"]
            descriptor_vals_SVM_df.loc[cell_name, "Max Time"]     = treatment_df.loc[index, "Max Time"]
            descriptor_vals_SVM_df.loc[cell_name, "Half Time"]    = treatment_df.loc[index, "Half Time"]
            descriptor_vals_SVM_df.loc[cell_name, "Area"]         = treatment_df.loc[index, "Area"]
    
            if not key_params:
                descriptor_vals_SVM_df.loc[cell_name, "Half Max"]     = treatment_df.loc[index, "Half Max"]
                descriptor_vals_SVM_df.loc[cell_name, "Max In Time"]  = treatment_df.loc[index, "Max In"][0]
                descriptor_vals_SVM_df.loc[cell_name, "Max In Rate"]  = treatment_df.loc[index, "Max In"][1]
                descriptor_vals_SVM_df.loc[cell_name, "Max Out Time"] = treatment_df.loc[index, "Max Out"][0]
                descriptor_vals_SVM_df.loc[cell_name, "Max Out Rate"] = treatment_df.loc[index, "Max Out"][1]
                descriptor_vals_SVM_df.loc[cell_name, "Peaks"]        = treatment_df.loc[index, "Peaks"]
            
    if drop_nans:
        total_cells  = descriptor_vals_SVM_df.shape[0]
        descriptor_vals_SVM_df.dropna(inplace = True)
        total_nans   = total_cells - descriptor_vals_SVM_df.shape[0]
        percent_nans = (total_nans / total_cells) * 100
        
    # update features (X) and targets (y)
    X_SVM_df    = descriptor_vals_SVM_df.drop(columns = ["Behavior"])
    y_SVM_df    = descriptor_vals_SVM_df["Behavior"]
    
    if scale_range is None:
        # scale data using min-max method
        scaler      = MinMaxScaler()
        scaled_data = scaler.fit_transform(X_SVM_df)
        X_SVM_df    = pd.DataFrame(scaled_data, columns = X_SVM_df.columns, index = X_SVM_df.index)
    
    else:
        # manual scaling
        scaled_data = pd.DataFrame()
        
        for feature, feature_range in scale_range.items():
            x_min   = feature_range[0]
            x_max   = feature_range[1]
            
            # scale data using min-max method
            scaled_data[feature] = (descriptor_vals_SVM_df[feature] - x_min) / (x_max - x_min)
            
        X_SVM_df    = pd.DataFrame(scaled_data, columns = X_SVM_df.columns, index = X_SVM_df.index)
    
    # weight data
    if weights_dict is not None:
        for descriptor in list(weights_dict.keys()):
            X_SVM_df[descriptor] = X_SVM_df[descriptor] * weights_dict[descriptor]
    
    
    if scale_range is None:
        return descriptor_vals_SVM_df, X_SVM_df, y_SVM_df, scaler
    else:
        return descriptor_vals_SVM_df, X_SVM_df, y_SVM_df


#%% functions to build and train SVM

def test_SVM_params(X_SVM_df, y_SVM_df, grid = True, **kwargs):
    """
    Tests different SVM params using either grid or randomized search.

    Parameters
    ----------
    X_SVM_df : pd.DataFrame
        Feature matrix for SVM.
    y_SVM_df : pd.Series
        Target vector for SVM.
    grid : bool, optional
        Whether to use GridSearch (True) or RandomizedSearch (False).  The default is True.
    **param_grid : dict
        Parameter grid to be used in GridSearch.  Must be provided if grid is True.
    **param_dist : dict
        Parameter space to be used in RandomizedSearch.  Must be provided if grid is False.
        
    Raises
    ------
    ValueError
        Occurs if grid is True and param_grid is not provided.
        Occurs if grid is False and param_dist is not provided.

    Returns
    -------
    best_params : dict
        Dictionary of the best parameter values.
    best_score : flt
        Best accuracy score.

    """
    
    if grid:
        if "param_grid" not in kwargs:
            raise ValueError("param_grid must be provided if grid is False.")
        param_grid = kwargs["param_grid"]
        search = GridSearchCV(estimator = SVC(probability = True), param_grid = param_grid, cv = 5, scoring = "accuracy", n_jobs = -1)
    else:
        if "param_dist" not in kwargs:
            raise ValueError("param_dist must be provided if grid is True.")
        param_dist = kwargs["param_dist"]
        search = RandomizedSearchCV(estimator = SVC(probability = True), param_distributions = param_dist, n_iter = 10, cv = 5, scoring = "accuracy", n_jobs = -1)
    
    search.fit(X_SVM_df, y_SVM_df)
    
    best_params = search.best_params_
    best_score  = search.best_score_
    
    # extract the best model
    svm_model   = search.best_estimator_
    
    search_type = "grid" if grid is False else "random"
    
    print(f"Best parameters ({search_type}): {best_params}")
    print(f"Best score ({search_type}): {best_score}\n")
    
    return svm_model, best_params, best_score


#%% functions to visualize classification results

def percent_behaviors(dict_trace_descriptors, plot = None):
    """
    Generates a barplot showing the percentage breakdown of each behavior by treatment.

    Parameters
    ----------
    dict_trace_descriptors : dict
        Dictionary of dfs containing trace descriptors as described above.
    plot : bool, optional
        Whether or not and how to plot the results.  The default is None (no plot).
        The other options are "vert" (vertically oriented barplot) or "hor" (horizontally oriented barplot).
    
    Returns
    -------
    percents_df : df
        Percentage of total cells sorted into each behavior category by treatment.

    """
    treatments = [treatment for treatment in list(dict_trace_descriptors.keys())]
    treatments = sorted(treatments, key = lambda x: (x != '-PGN', int(x[:-1]) if x != '-PGN' else -1))
    PGN_concs  = [tmt.strip("X") if "X" in tmt else 0 for tmt in treatments]
    behavior_dict = {"N": "Nonresponsive",
                     "D": "Delayed",
                     "G": "Gradual",
                     "I": "Immediate",
                     "Id": "Immediate with decrease"}
    col_names = ["# cells"] + [behavior for behavior in behavior_dict]
    
    percents_df = pd.DataFrame(index = treatments, columns = col_names)
    
    for tmt, tmt_df in dict_trace_descriptors.items():
        total_cells = tmt_df.shape[0]
        
        percents_df.loc[tmt, "# cells"] = total_cells
        
        for behavior in col_names:
            if behavior == "# cells":
                continue
            
            behavior_name = behavior_dict[behavior]
            filtered_data = tmt_df[tmt_df['Behavior'] == behavior]
            sum_cells = filtered_data.shape[0]
            percent_cells = (sum_cells / total_cells) * 100
            
            percents_df.loc[tmt, behavior] = percent_cells
        
    if plot is not False:
        colors          = ["#DC143C", "#FF6F61", "indigo", "dodgerblue", "grey"]
        color_palette   = sns.color_palette(colors)
        treatment_color = {"Immediate": color_palette[0], 
                           "Immediate with decrease": color_palette[1],
                           "Gradual": color_palette[2],
                           "Delayed": color_palette[3],
                           "Nonresponsive": color_palette[4]}
        color_list = [treatment_color[behavior_dict[behavior]] for behavior in col_names if behavior != "# cells"]
        
        cols_to_plot = percents_df.columns[1:]
        
        if plot == "vert":
            ax = percents_df[cols_to_plot].plot.bar(stacked = True, figsize = (6, 10), color = color_list)
        
            plt.xlabel(r'[PGN] ($\mu$g/mL)', fontsize = 16)
            plt.ylabel('% cells', fontsize = 16)
            plt.xticks(np.arange(len(PGN_concs)), PGN_concs, rotation = 0)
            ax.tick_params(axis = "both", labelsize = 14)
            
            for i, treatment in enumerate(treatments):
                x = percents_df.columns[1:]
                y = percents_df.loc[treatment, x].values
                total_cells = percents_df.loc[treatment, "# cells"]
                
                ax.text(x = i, y = y.sum() + 1, s = f"({total_cells})", ha = 'center', va = 'bottom', fontsize = 14, color = 'black', fontstyle = "italic")
            
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
            plt.savefig(f"behavior_distribution_{formatted_date}", dpi = 700)
            plt.show()
            
        elif plot == "hor":
            ax = percents_df[cols_to_plot].plot.barh(stacked = True, figsize = (10, 6), color = color_list)
        
            plt.ylabel(r'[PGN] ($\mu$g/mL)', fontsize = 16)
            plt.xlabel('% cells', fontsize = 16)
            plt.yticks(np.arange(len(PGN_concs)), PGN_concs, rotation = 0)
            ax.tick_params(axis = "both", labelsize = 14)
            
            for i, treatment in enumerate(treatments):
                y = percents_df.columns[1:]
                x = percents_df.loc[treatment, y].values
                total_cells = percents_df.loc[treatment, "# cells"]
                
                ax.text(y = i - 0.1, x = x.sum() + 2, s=f"({total_cells})", ha='center', va='bottom', fontsize=10, color='black', fontstyle = "italic", rotation = -90)
            
            plt.legend(bbox_to_anchor=(1, 1.02), loc='upper left')
            plt.savefig(f"behavior_distribution_{formatted_date}", dpi = 700)
            plt.show()
        
    return percents_df


#%% functions to build, train, and run SVM on complete dataset

def plot_sample_and_all_traces(subcluster_traces_smooth, results_dict, cell_categories_df, dict_trace_descriptors_SVM, treatment = "all", random_cell = True, cell_names = None, stim_time = 30):
    """
    Plots both a sample trace and traces for all cells classified into each behavior category.

    Parameters
    ----------
    subcluster_traces_smooth : dict, optional
        Dictionary of interpolated/smoothed ratio time trace values for cells in each subcluster within clusters for all treatments.
    results_dict : dict
        Summary of actual and predicted results per cell for each kernel tested.
    cell_categories_df : TYPE
        DESCRIPTION.
    dict_trace_descriptors_SVM : TYPE
        DESCRIPTION.
    treatment : str, optional
        Type of treatment (-PGN, 1X, 10X, 100X) or "all". The default is "all".
    random_cell : bool, optional
        Whether or not to plot a random cell trace.  The default is True.
    cell_names : dict, optional
        If random_cell is False, the sample cell traces to plot by behavior category. The default is None.
    stim_time : int, optional
        Time of PGN stimulus.  The default is 30.

    Raises
    ------
    ValueError
        Occurs if cell_names is provided and random_cell is True.

    Returns
    -------
    None.

    """
    if cell_names != None and random_cell == True:
        raise ValueError("random_cell must be False if cell_name is provided.")  
    
    colors          = ["#DC143C", "#FF6F61", "indigo", "dodgerblue", "grey"]
    color_palette   = sns.color_palette(colors)
    behavior_colors = {"I": color_palette[0], "Id": color_palette[1], "G": color_palette[2], "D": color_palette[3], "N": color_palette[4]}
    behavior_keys   = {"I": "Immediate", "Id": "Immediate with decay", "G": "Gradual", "D": "Delayed", "N": "Nonresponsive"}
    behavior_rev    = {v: k for k, v in behavior_keys.items()}
    
    # flatten trace data
    all_traces_df   = flatten_trace_df(subcluster_traces_smooth)
    times           = list(all_traces_df.columns)
    
    # initialize figure
    fig, axs    = plt.subplots(2, 5, figsize = (30, 10), sharex = True, sharey = True)
        
    # extract dataframe of all cells
    results_df  = results_dict["All cells"]
    
    # convert cell_categories_df to dict and reorder
    cell_categories_dict = cell_categories_df.set_index("Category")["Cells"].to_dict()
    
    for n, (SVM_behavior, behavior_name) in enumerate(behavior_keys.items()):
        # plot all traces for each behavior
        behavior_traces = pd.DataFrame()
        behavior_color  = behavior_colors[SVM_behavior]
        num_cells       = 0
        
        mask        = (results_df["Predicted"] == SVM_behavior)
        indices     = results_df.index[mask].tolist()
        if treatment != "all":
            indices = [index for index in indices if treatment in index]
        
        num_cells   = len(indices)
        traces      = all_traces_df.loc[indices]
        color       = behavior_colors[SVM_behavior]
        axs[1, n].plot(times, traces.T, linewidth = 0.5, color = color)
        axs[1, n].plot(times, traces.mean(axis=0), color = "black", linewidth = 1.5)
        
        # set axes labels and title
        title_color = behavior_color
        axs[1, n].tick_params(axis = "both", labelsize = 14)
        axs[1, n].text(y = 1.9, x = times[-1] - 60, s = f"n = {num_cells}", ha = "center", va = "center", fontsize = 16)
        
        # highlight pre- and post-stim times
        y_min, y_max = axs[1, n].get_ylim()
        prestim  = patches.Rectangle((0, 0.6), stim_time, 0.025, color = "black")
        hatch = patches.Rectangle((stim_time, 0.6), times[-1] - stim_time, 0.025, edgecolor="black", facecolor="none", hatch='///')
        axs[1, n].add_patch(prestim)
        axs[1, n].add_patch(hatch)
        
        # plot test trace for each behavior
        behavior_cells = cell_categories_dict[behavior_name]
        if random_cell:
            cell_name = random.choice(behavior_cells)
        else:
            cell_name = cell_names[SVM_behavior]
        print(cell_name)
        tmt, cell = cell_name.split(maxsplit = 1)
        
        # retrieve timecourse data
        cell_trace = all_traces_df.loc[cell_name].values
        cell_data  = dict_trace_descriptors_SVM[tmt].loc[cell]
        print(f"{cell_data}\n")
        
        # plot trace
        axs[0, n].plot(times, cell_trace, color = behavior_color)
        
        # set title and labels
        axs[0, n].set_title(f"{behavior_name}", fontsize = 20, color = behavior_color)
        axs[0, n].tick_params(axis = "both", labelsize = 14)
        
        # set y-axis limits and ticks
        axs[0, n].set_ylim(0.6, 2.0)
        axs[0, n].set_yticks([0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
        
        # plot horizontal line at 1.2
        axs[0, n].axhline(y = 1.2, color = "grey", linestyle = "dashed", linewidth = 0.5)
        
        max_val = cell_data["Max Value"]
        max_time = cell_data["Max Time"]
        
        if SVM_behavior != "N":
            # add vertical lines at t = stim_time + 120
            axs[0, n].axvline(x = stim_time + 120, color = "grey", linestyle = "dashed", linewidth = 0.5)
            axs[0, n].text(y = 1.85, x = stim_time + 110, s = f"t = {stim_time + 120}", ha ='center', va = 'center', fontsize = 14, color = 'grey', fontstyle = "italic", rotation = 90)
            t150_val = cell_trace[stim_time + 120]
            
            # add "Initial behavior" label
            axs[0, n].text(y = 0.8, x = (stim_time + 120) / 2, s = "Initial\nbehavior", ha = 'center', va = 'center', fontsize = 14, color = 'black')
        
        if cell_data["Local Max"] != []:
            peak1_time = cell_data["Local Max"][0][0]
            peak1_val  = cell_data["Local Max"][0][1]
            # print(peak1_time)
        else:
            peak1_time = max_time
            peak1_val = max_val
        
        if "I" in SVM_behavior:
            # plot max point
            axs[0, n].plot(max_time, max_val, "go", color = "black")
            
            # add "Max value"
            axs[0, n].text(y = max_val + 0.03, x = max_time - 5, s = "Max", ha = 'right', va = 'center', fontsize = 14, color = 'black')
            
            # plot first peak point
            if peak1_time != max_time and peak1_val != max_val:
                axs[0, n].axvline(x = peak1_time, color = "grey", linestyle = "dashed", linewidth = 0.5)
                axs[0, n].text(y = 1.85, x = peak1_time - 5, s = f"t = {peak1_time}", ha = 'center', va = 'center', fontsize = 14, color = 'grey', fontstyle = "italic", rotation = 90)
                axs[0, n].plot(peak1_time, peak1_val, "go", color = "black")
                axs[0, n].text(y = peak1_val + 0.03, x = peak1_time - 5, s = "First peak", ha = 'right', va = 'center', fontsize = 14, color = 'black')
            
            # shade -0.2 from first peak
            axs[0, n].fill_between(x = [peak1_time, times[-1]], y1 = peak1_val, y2 = peak1_val - 0.2, color = "lightgrey", alpha = 0.5)
            axs[0, n].text(y = 0.8, x = ((times[-1] - peak1_time) / 2) + peak1_time, s = "Long-term\nbehavior", ha = 'center', va = 'center', fontsize = 14, color = 'black')
            
            # add height arrow
            fin_val = cell_trace[-1]
            
            axs[0, n].annotate('', xy = (times[-1] + 10, fin_val), xytext = (times[-1] + 10, max_val),
                arrowprops = dict(
                    arrowstyle     = '|-|',
                    color          = behavior_color,
                    linewidth      = 0.5,
                    mutation_scale = 2.5))
            
            # add text annotation
            axs[0, n].text(times[-1] + 35, (fin_val + max_val) / 2, "Δ FC", color = behavior_color, ha = 'center', va = 'center', fontsize = 14, rotation = 90)
        
        else:
            
            if SVM_behavior != "N":
                axs[0, n].axvline(x = (times[-1] / 2), color = "grey", linestyle = "dashed", linewidth = 0.5)
                axs[0, n].text(y = 1.85, x = (times[-1] / 2) - 10, s = f"t = {int(times[-1] / 2)}", ha ='center', va = 'center', fontsize = 14, color = 'grey', fontstyle = "italic", rotation = 90)
                t430_val = cell_trace[int(times[-1] / 2)]
                
                sig_filter = [(index, value) for index, value in enumerate(cell_trace) if value >= 1.2]
                sig_time, sig_val = sig_filter[0]
                axs[0, n].plot(sig_time, sig_val, "go", color = "black")
                axs[0, n].text(y = sig_val + 0.03, x = sig_time - 5, s = "FC ≥ 1.2", ha = "right", va = "center", fontsize = 14, color = "black")
    
    fig.supxlabel("Time (min)", fontsize = 16)
    fig.text(0, 0.5, 'Nuclear Relish fraction (fold change)', va='center', ha='center', fontsize=16, rotation='vertical')
    
    plt.tight_layout(pad = 2)     
    plt.savefig(f"classified_traces_{formatted_date}", dpi = 700)
    plt.show()
    

def run_SVM(subcluster_traces_smooth, dict_trace_descriptors, cell_categories_df, param_grid, treatment = "all", random_cell = False, cell_names = None, keep_SVM = True, return_vals = True, stim_time = 30, scale_range = None):
    """
    Trains and runs SVM on all cells using test_SVM_params.
    Plots traces by category using plot_sample_and_all_traces and generates percent behaviors barplot using percent_behaviors.

    Parameters
    ----------
    subcluster_traces_smooth : dict
        Dictionary of interpolated/smoothed ratio time trace values for cells in each subcluster within clusters for all treatments.
    dict_trace_descriptors : dict
        Dictionary of dfs containing trace descriptors as described above.
    cell_categories_df : df
        Manually sorted training set; list of cell names manually assigned to each behavior in step 10.
    param_grid : dict
        Dictionary of parameter values (C, degree, gamma, kernel) for grid searching.
    treatment : str, optional
        Type of treatment (-PGN, 1X, 10X, 100X) or "all". The default is "all".
    random_cell : bool, optional
        Whether or not to plot a random cell trace.  The default is False.
    cell_names : dict, optional
        If random_cell is False, the sample cell traces to plot by behavior category. The default is None.
    keep_SVM : bool, optional
        Whether or not to save the SVM model.  The default is True.
    return_vals : bool, optional
        Whether or not to return values.  The default is True.
    stim_time : int, optional
        Time of PGN stimulus.  The default is 30.
    scale_range : dict, optional
        If manually scaling, the scale ranges for each parameter.  The default is None.

    Raises
    ------
    ValueError
        Occurs if cell_names is provided and random_cell is True.
        Occurs if cell indices do not match for classified cells.
        Occurs if cell indices do not match for unclassified cells.

    Returns
    -------
    dict_trace_descriptors_SVM : dict
        dict_trace_descriptors with "Behavior" column populated with SVM results.
    df_descriptor_vals_all : df
        Flattened version of dict_trace_descriptors containing only the features used to train the SVM.
        Because key_params is currently set to True, this includes ["Max Value", "Max Time", "Half Time", "Area"].
    scaler : preproccessing._data.MinMaxScaler
        MinMaxScaler object.  Returned only if scale_range is not provided.
    svm_model : svm._classes.SVC
        Trained SVM classification model.
    results_dict : dict
        Summary of actual and predicted results per cell.
    percents_df : df
        Percentage of total cells sorted into each behavior category by treatment.
        

    """
    if cell_names != None and random_cell == True:
        raise ValueError("random_cell must be False if cell_name is provided.")  
    
    tmts = [tmt for tmt in subcluster_traces_smooth]
    
    # prep dict to store results
    results_dict = dict.fromkeys(["Classified cells", "Unclassified cells", "All cells"])
    
    dict_trace_descriptors_subset                      = trace_descriptor_subset(dict_trace_descriptors, cell_categories_df)
        
    # display(dict_trace_descriptors_subset)
    if scale_range is None:
        df_descriptor_vals_train, X_df_train, y_ser_train, scaler_train = prep_descriptor_vals_SVM(dict_trace_descriptors_subset, scale_range = scale_range)
        df_descriptor_vals_all, X_df_all, y_ser_all, scaler             = prep_descriptor_vals_SVM(dict_trace_descriptors, scale_range = scale_range)
    else:
        df_descriptor_vals_train, X_df_train, y_ser_train  = prep_descriptor_vals_SVM(dict_trace_descriptors_subset, scale_range = scale_range)
        df_descriptor_vals_all, X_df_all, y_ser_all        = prep_descriptor_vals_SVM(dict_trace_descriptors, scale_range = scale_range)
    
    # update "Behavior" columns of non-classified cells to None
    classified_cells                                       = set(df_descriptor_vals_train.index)
    df_descriptor_vals_all.loc[~df_descriptor_vals_all.index.isin(classified_cells), "Behavior"] = None
    y_ser_all.loc[~y_ser_all.index.isin(classified_cells)] = None
    
    # extract data for non-classified cells
    df_descriptor_vals_unk                                 = df_descriptor_vals_all[df_descriptor_vals_all["Behavior"].isna()]
    unclassified_cells                                     = set(df_descriptor_vals_unk.index)
    X_df_unk                                               = X_df_all.loc[X_df_all.index.isin(unclassified_cells)]
    
    # train and optimize SVM on classified cells
    svm_model, best_params, best_score                     = test_SVM_params(X_df_train, y_ser_train, param_grid = param_grid, grid = True)
    y_pred_train                                           = svm_model.predict(X_df_train)
    
    # prep results_dict for plotting
    kernel                                                 = best_params["kernel"] 
    results_df_train                                       = pd.DataFrame(index = list(X_df_train.index), columns = ["Actual", "Predicted"])
    if set(results_df_train.index) != classified_cells:
        print(f"Classified cells (len = {len(classified_cells)}): {classified_cells}")
        print(f"X_df_train (len = {len(X_df_train.index)}: {X_df_train.index}")
        raise ValueError("Cell indices do not match for classified cells.")
    else:
        results_df_train["Actual"]                         = y_ser_train
        results_df_train["Predicted"]                      = y_pred_train
    # print(results_df_train)
    results_dict_train                                     = {}
    results_dict_train[kernel]                             = results_df_train
    
    # update main dict
    results_dict["Classified cells"]                       = results_df_train
    
    # calculate SVM accuracy
    total_cells_train                                      = results_df_train.shape[0]
    cor_cells_train                                        = (results_df_train["Actual"] == results_df_train["Predicted"]).sum()
    train_acc                                              = (cor_cells_train / total_cells_train) * 100
    print(f"SVM accuracy for classified cells (n = {total_cells_train}): {train_acc:.3f}%")
        
    # run SVM on remaining cells and prep results_dict for plotting
    y_pred_unk                                             = svm_model.predict(X_df_unk)
    results_df_unk                                         = pd.DataFrame(index = list(X_df_unk.index), columns = ["Predicted"])
    if set(results_df_unk.index) != unclassified_cells:
        print(f"Unclassified cells (len = {len(unclassified_cells)}): {unclassified_cells}")
        print(f"X_df_unk (len = {len(X_df_unk.index)}: {X_df_unk.index}")
        raise ValueError("Cell indices do not match for unclassified cells.")
    else:
        results_df_unk["Predicted"]                        = y_pred_unk
        results_df_unk["Actual"]                           = None
    results_dict_unk                                       = {}
    results_dict_unk[kernel]                               = results_df_unk
    
    # update main dict
    results_dict["Unclassified cells"]                     = results_df_unk
    
    # update main dict with results for all cells
    results_df                                             = pd.concat([results_df_train, results_df_unk], axis = 0)
    results_dict["All cells"]                              = results_df
    
    # create summary entry in results_dict
    col_names                                              = list(np.unique(y_pred_train))
    col_names.insert(0, "# cells")
    summary_df                                             = pd.DataFrame(index = ["Classified cells", "Unclassified cells", "All cells"] + tmts, columns = col_names)
    for data_type, data in results_dict.items():
        total_cells                                        = data.shape[0]
        summary_df.loc[data_type, "# cells"]               = total_cells
        
        for behavior in col_names:
            if behavior == "# cells":
                continue
            behavior_mask                                  = (data["Predicted"] == behavior)
            num_cells                                      = behavior_mask.sum()
            per_cells                                      = (num_cells / total_cells) * 100
            summary_df.loc[data_type, behavior]            = per_cells
    
    for tmt in tmts:
        tmt_data                                           = results_df[results_df.index.str.contains(tmt)]
        tmt_cells                                          = tmt_data.shape[0]
        summary_df.loc[tmt, "# cells"]                     = tmt_cells
        
        for behavior in col_names:
            if behavior == "# cells":
                continue
            tmt_behavior_mask                              = (tmt_data["Predicted"] == behavior)
            num_cells                                      = tmt_behavior_mask.sum()
            per_cells                                      = (num_cells / tmt_cells) * 100
            summary_df.loc[tmt, behavior]                  = per_cells
    
    results_dict["Distribution summary (%)"]               = summary_df
    
    dict_trace_descriptors_SVM = copy.deepcopy(dict_trace_descriptors)
    
    del dict_trace_descriptors_SVM["Averages"]
    
    # update "Behavior" column in dict_trace_descriptors_SVM
    for tmt, tmt_df in dict_trace_descriptors_SVM.items():
        
        # filter cells by treatment from results_df
        tmt_idx = [idx for idx in results_df.index if idx.startswith(tmt)]
        
        # extract corresponding sub-df
        tmt_pred_df = results_df.loc[tmt_idx]
        
        # remove tmt prefix from indices
        tmt_pred_df.index = [idx[len(tmt) + 1:] for idx in tmt_pred_df.index]
        # display(tmt_pred_df)
            
        # map behavior values to tmt_df
        behavior_map = tmt_pred_df[["Predicted"]].fillna(tmt_pred_df["Actual"]).to_dict()["Predicted"]
        # display(behavior_map)
        tmt_df["Behavior"] = tmt_df.index.map(behavior_map)
                    
        # update dict
        dict_trace_descriptors_SVM[tmt] = tmt_df
    
    # plot traces
    plot_sample_and_all_traces(subcluster_traces_smooth, results_dict, cell_categories_df, dict_trace_descriptors_SVM, random_cell = True, stim_time = stim_time)
    
    # plot stacked bar plot
    percents_df = percent_behaviors(dict_trace_descriptors_SVM, plot = "vert")
    
    if return_vals:
        
        if scale_range is None:
            return dict_trace_descriptors_SVM, df_descriptor_vals_all, scaler, svm_model, results_dict, percents_df
        else:
            return dict_trace_descriptors_SVM, df_descriptor_vals_all, svm_model, results_dict, percents_df


#%% # === MAIN LOOP ===

param_grid  = {"C": [0.1, 1, 10, 100],
                   "gamma": [0.1, 0.01, 0.001, 0.0001],
                   "kernel": ["linear", "rbf", "poly", "sigmoid"],
                   "degree": [2, 3, 4]}

param_dist  = {"C": uniform(loc = 0.1, scale = 1.0),
                   "gamma": [0.1, 0.01, 0.001, 0.0001],
                   "kernel": ["linear", "rbf", "poly", "sigmoid"],
                   "degree": randint(2, 5)}

scale_range = {"Max Value": [1.0, 1.8],
                   "Max Time": [0, 1000],
                   "Half Time": [0, 500],
                   "Area": [900, 1500]}

# file paths
all_data         = "/path/to/your/data/2025-01-01_datasetName/"
file_path_step9  = all_data + "Processed Traces"
file_path_step10 = all_data + "SVM Classifier"
subcluster_traces_smooth = import_data(file_path_step9, "subcluster_traces_smooth")
dict_trace_descriptors   = import_data(file_path_step9, "dict_trace_descriptors")
behavior_categories_df   = import_data(file_path_step10, "behavior_categories_df")

# run complete SVM
dict_trace_descriptors_SVM, df_descriptor_vals_all, SVM_model, SVM_results_dict, percent_behaviors_df = run_SVM(subcluster_traces_smooth, dict_trace_descriptors, behavior_categories_df, param_grid = param_grid, random_cell = True, scale_range = scale_range)
save_data(file_path_step10, dict_trace_descriptors_SVM, "dict_trace_descriptors_SVM")
save_data(file_path_step10, df_descriptor_vals_all, "df_descriptor_vals_all")
save_data(file_path_step10, SVM_model, "SVM_model")
save_data(file_path_step10, SVM_results_dict, "SVM_results_dict")
save_data(file_path_step10, percent_behaviors_df, "percent_behaviors_df")
