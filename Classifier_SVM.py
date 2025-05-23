# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 10:52:01 2024

@author: emmar

Name: Emma Rits
Date: Monday, July 15, 2024
Description: Trace behavior sorting SVM

"""

#%% File Paths

file_path_trace  = "/path/to/your/data/AllDatasets_TraceDescriptors"
file_path_clus   = "/path/to/your/data/AllDatasets_SubclusterTraces"
file_path_fft    = "/path/to/your/data/AllDatasets_FFT"
file_path_svm    = "/path/to/your/data/AllDatasets_SVM"

#%% import packages, functions, and data

# packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import colorsys
import os
import pickle
import copy
import random
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import uniform, randint, chisquare
from IPython.display import display
from matplotlib.lines import Line2D

# current date
import datetime
current_date   = datetime.date.today()
formatted_date = current_date.strftime("%Y-%m-%d")

# functions
from ratiotimetrace_clustering_compiled import save_data, import_data, add_behavior_descriptor
from ratiotimetrace_clustering_compiled import test_goodcomp_dict_descriptors_select, test_goodcomp_subcluster_behavior_020_r2
from cell_trace_classifying_interface import flatten_trace_df

# UNCOMMENT DURING FIRST RUN
# data (from lab drive)
goodcomp_dict_trace_descriptors           = import_data(file_path_trace, "goodcomp_dict_trace_descriptors_subcluster020-020")
goodcomp_subcluster_traces_div_smooth     = import_data(file_path_clus, "goodcomp_subcluster_traces_div_smooth_subcluster020")

# data (from local drive)
goodcomp_dict_trace_descriptors           = import_data(file_path_df, "goodcomp_dict_trace_descriptors")
goodcomp_subcluster_traces_div_smooth     = import_data(file_path_df, "goodcomp_subcluster_traces_div_smooth")
goodcomp_dict_trace_descriptors           = add_behavior_descriptor(goodcomp_dict_trace_descriptors, test_goodcomp_subcluster_behavior_020_r2)
# goodcomp_trace_sorting_cell_categories_v1 = import_data(file_path_SVM, "goodcomp_trace_sorting_cell_categories_v1")
# goodcomp_trace_sorting_cell_categories_v1.rename(columns = {"Cell Index": "Category", "Category": "Cells"}, inplace = True)
goodcomp_trace_sorting_cell_categories_v2 = import_data(file_path_SVMc, "goodcomp_trace_sorting_cell_categories_v2")
goodcomp2_dict_trace_descriptors          = import_data(file_path_df, "goodcomp2_dict_trace_descriptors")
goodcomp2_subcluster_traces_div_smooth    = import_data(file_path_df, "goodcomp2_subcluster_traces_div_smooth")
goodcomp3_dict_trace_descriptors          = import_data(file_path_df, "goodcomp3_dict_trace_descriptors")
goodcomp3_subcluster_traces_div_smooth    = import_data(file_path_df, "goodcomp3_subcluster_traces_div_smooth")


#%% functions to prepare data

def trace_descriptor_subset(dict_trace_descriptors, cell_categories_df, remove_Ic = True):
    """
    Extract subset of data from dict_trace_descriptors corresponding to cells classified using GUI.

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
                                                       Ic = Immediate with continued
                                                       Ip = Immediate with plateau
                                                       Id = Immediate with decrease
                                                       (I = Immediate)
    cell_categories_df : pd.DataFrame
        Dataframe listing the cells manually assigned to each behavior category
        
    remove_Ic : bool, optional
        Whether or not to remove the "Immediate with continued" category.  The default is True.

    Returns
    -------
    dict_subset : dict
        Subset of dict_trace_descriptors; same format, but only including the cells in cell_categories_df

    """
    
    treatments = [treatment for treatment in dict_trace_descriptors if treatment != "Averages"]
    # print(treatments)
    behavior_keys = {
        "N": "Nonresponsive",
        "D": "Delayed",
        "G": "Gradual",
        "Ic": "Immediate with continued",
        "Id": "Immediate with decay",
        "Ip": "Immediate with plateau"} if not remove_Ic else {
            "N": "Nonresponsive",
            "D": "Delayed",
            "G": "Gradual",
            "I": "Immediate",
            "Id": "Immediate with decay"}
            
    # convert cell_categories_df to dict
    cell_categories_dict = cell_categories_df.set_index("Category")["Cells"].to_dict()
    # print(cell_categories_dict)
    
    # generate list of cells to extract 
    cell_names_subset = set(cell for cells in cell_categories_dict.values() for cell in cells)
    # print(cell_names_subset)
    # print(len(cell_names_subset))
    
    # initialize dicts for data and cell names
    dict_subset     = dict.fromkeys(treatments)
    cell_names_dict = dict.fromkeys(treatments)
    
    for treatment in treatments:
        # retrieve cells in treatment
        treatment_df = dict_trace_descriptors[treatment]
        cell_names_treatment = [cell_name.split(" ", 1)[1] for cell_name in cell_names_subset if cell_name.split(" ", 1)[0] == treatment]
        # print(f"{treatment}: {cell_names_treatment}")
        cell_names_dict[treatment] = cell_names_treatment
        
        # extract data and update df_subset and dict_subset
        df_subset = treatment_df.loc[treatment_df.index.intersection(cell_names_treatment)]
        dict_subset[treatment] = df_subset
        
    # correct behavior classifications
    for behavior_name in cell_categories_dict:
        behavior = {v: k for k, v in behavior_keys.items()}.get(behavior_name)
        # print(f"\n{behavior}: {behavior_name}")
        for cell_name in cell_categories_dict[behavior_name]:
            treatment, cell = cell_name.split(" ", 1)
            # print(f"{treatment}: {cell}, {behavior}")
            dict_subset[treatment].loc[cell, "Behavior"] = behavior
    
    return dict_subset


def prep_descriptor_vals_SVM(dict_trace_descriptors, key_params = True, drop_nans = True, weights_dict = None, collapse_imm = False):
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
                                                       Ic = Immediate with continued
                                                       Ip = Immediate with plateau
                                                       Id = Immediate with decrease
    key_params : bool, optional
        Whether to include only the "key" parameters (max value, max time, half max, half time, and area). The default is True.
    drop_nans : bool, optional
        Whether to exclude all cells with data containing NaNs. The default is True.
    weights_dict : dict, optional
        Dictionary of descriptors of interest with their corresponding weights. The default is None.
    collapse_imm : bool, optional
        Whether or not to collapse "Immediate with continued", "Immediate with plateau", and "Immediate with decay" into one "Immediate category."
        The default is False.

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
    # print(treatments)
    if key_params:
        descriptors            = ["Behavior",
                                  "Max Value",
                                  "Max Time",
                                  "Half Max",
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
        # print(treatment, cells)
        
        for index, row in treatment_df.iterrows():
            # add treatment to cell name
            cell_name                                             = treatment + " " + index
            # print(cell_name)
            
            # extract data
            descriptor_vals_SVM_df.loc[cell_name, "Behavior"]     = treatment_df.loc[index, "Behavior"]
            descriptor_vals_SVM_df.loc[cell_name, "Max Value"]    = treatment_df.loc[index, "Max Value"]
            descriptor_vals_SVM_df.loc[cell_name, "Max Time"]     = treatment_df.loc[index, "Max Time"]
            descriptor_vals_SVM_df.loc[cell_name, "Half Max"]     = treatment_df.loc[index, "Half Max"]
            descriptor_vals_SVM_df.loc[cell_name, "Half Time"]    = treatment_df.loc[index, "Half Time"]
            descriptor_vals_SVM_df.loc[cell_name, "Area"]         = treatment_df.loc[index, "Area"]
    
            if not key_params:
                descriptor_vals_SVM_df.loc[cell_name, "Max In Time"]  = treatment_df.loc[index, "Max In"][0]
                descriptor_vals_SVM_df.loc[cell_name, "Max In Rate"]  = treatment_df.loc[index, "Max In"][1]
                descriptor_vals_SVM_df.loc[cell_name, "Max Out Time"] = treatment_df.loc[index, "Max Out"][0]
                descriptor_vals_SVM_df.loc[cell_name, "Max Out Rate"] = treatment_df.loc[index, "Max Out"][1]
                descriptor_vals_SVM_df.loc[cell_name, "Peaks"]        = treatment_df.loc[index, "Peaks"]
        
    # collapse "Immediate" categories if collapse_imm = True
    if collapse_imm:
        descriptor_vals_SVM_df["Behavior"] = descriptor_vals_SVM_df["Behavior"].replace({"Ic": "I", "Ip": "I", "Id": "I"})
        
    if drop_nans:
        total_cells  = descriptor_vals_SVM_df.shape[0]
        descriptor_vals_SVM_df.dropna(inplace = True)
        total_nans   = total_cells - descriptor_vals_SVM_df.shape[0]
        percent_nans = (total_nans / total_cells) * 100
        # print(f"{percent_nans:.3f}% of cells were dropped due to containing NaN values.")
    
    # update features (X) and targets (y)
    X_SVM_df    = descriptor_vals_SVM_df.drop(columns = ["Behavior"])
    y_SVM_df    = descriptor_vals_SVM_df["Behavior"]
    
    # scale data using min-max method
    scaler      = MinMaxScaler()
    scaled_data = scaler.fit_transform(X_SVM_df)
    X_SVM_df    = pd.DataFrame(scaled_data, columns = X_SVM_df.columns, index = X_SVM_df.index)
    
    # weight data
    if weights_dict is not None:
        for descriptor in list(weights_dict.keys()):
            X_SVM_df[descriptor] = X_SVM_df[descriptor] * weights_dict[descriptor]
    
    return descriptor_vals_SVM_df, X_SVM_df, y_SVM_df


def split_train_test(X_SVM_df, y_SVM_df, test_size = None, train_size = None):
    """
    Splits the data into training and testing sets.

    Parameters
    ----------
    X_SVM_df : pd.DataFrame
        Feature matrix for SVM.
    y_SVM_df : pd.Series
        Target vector for SVM.
    test_size : flt, optional
        Proportion of the dataset to include in the test split (should be between 0.0 and 1.0). The default is None (test set is 25% of total data).
    train_size : flt, optional
        Proportion of the dataset to include in the train split (should be between 0.0 and 1.0). The default is None (training set is 75% of total data).

    Returns
    -------
    X_train : pd.DataFrame
        Training subset of X_SVM_df
    X_test : pd.DataFrame
        Testing subset of X_SVM_df
    y_train : pd.Series
        Training subset of y_SVM_df
    y_test : pd.Series
        Testing subset of y_SVM_df

    """
    
    X_train, X_test, y_train, y_test = train_test_split(X_SVM_df, y_SVM_df, test_size = test_size, train_size = train_size)
    return X_train, X_test, y_train, y_test
    
    
#%% functions to build and train SVM

def test_SVM_kernels(X_SVM_df, y_SVM_df, kernel_type, show = False, cv = False, cv_val = 5, **kwargs):
    """
    Test SVM with different kernels and perform cross-validation (optional).

    Parameters
    ----------
    X_SVM_df : pd.DataFrame
        Feature matrix for SVM.
    y_SVM_df : pd.Series
        Target vector for SVM.
    kernel_type : str
        Type of SVM kernel to use.  Options:
            "linear":   Linear kernel
            "poly":     Polynomial kernel
            "rbf":      Radial basis function (RBF) kernel
            "sigmoid":  Sigmoid kernel
    cv : bool, optional
        Whether or not to perform cross-validation.  The default is False.
    cv_val : int, optional
        Number of folds for cross-validation.  The default is 5.
    show : bool, optional
        Whether or not to print classification reports and confusion matrices. The default is False.
    **kwargs : dict
        Additional (optional) parameters to pass to SVC.

    Returns
    -------
    If cv is False:
        results_df / results_dict : pd.DataFrame / dict
            Summary of actual and predicted results per cell for each kernel tested.
        class_rpt / class_rpt_dict : str / dict
            Classification report (returned as a dict if kernel_type = "test")
            Interpreting classification report (TP = true positives, FP = False positives, TN = True negatives, FN = False negatives):
                Precision:  Accuracy of positive predictions for a specific class
                            Precision = TP / (TP + FP)
                            High precision --> When a model predicts a class, it is likely to be correct
                Recall:     Ability to correctly identify all instances of a class
                            Recall    = TP / (TP + FN)
                            High recall    --> Model correctly identifies a high proportion of actual positives
                F1-score:   Harmonic mean of precision and recall
                            F1-score  = 2 * (P * R) / (P + R)
                            High F1-score  --> Better overall performance (balance of precision/recall)
                Support:    # of actual occurrences of the class in the test dataset
                Accuracy:   Proportion of true results among the total number of cases
                            Accuracy  = (TP + TN) / (TP + TN + FP + FN)
        conf_mtx / conf_mtx_dict : array / dict
            Confusion matrix (returned as a dict if kerneL-type = "test")
            Interpreting confusion matrix:
                Rows:       Each row corresponds to the actual (true) class labels
                Columns:    Each column corresponds to the predicted class labels
    If cv is True:
        scores / scores_dict : array / dict
            Cross-validation scores (returned as a dict if kernel_type = "test")
    """
    
    # split data into training and testing sets
    X_train, X_test, y_train, y_test = split_train_test(X_SVM_df, y_SVM_df)
    all_cells                        = list(X_SVM_df.index)
    test_cells                       = list(X_test.index)
    
    if kernel_type == "test":
        kernels        = ["linear", "poly", "rbf", "sigmoid"]
        results_dict   = dict.fromkeys(kernels)
        class_rpt_dict = dict.fromkeys(kernels)
        conf_mtx_dict  = dict.fromkeys(kernels)
        scores_dict    = dict.fromkeys(kernels) if cv else None
        
        for kernel in kernels:
            svm_model               = SVC(kernel = kernel, **kwargs)
            
            # if performing cross-validation
            if cv:
                scores              = cross_val_score(svm_model, X_SVM_df, y_SVM_df, cv = cv_val)
                scores_dict[kernel] = scores
                svm_model.fit(X_SVM_df, y_SVM_df)
                y_pred              = svm_model.predict(X_SVM_df)
                
            # if splitting into training/testing sets
            else:
                results_df          = pd.DataFrame(index = test_cells, columns = ["Actual", "Predicted"])
                if not results_df.index.equals(X_test.index):
                    raise ValueError("Test and result cell indices do not match.")
                else:
                    results_df["Actual"] = y_test
                
                svm_model.fit(X_train, y_train)
                y_pred              = svm_model.predict(X_test)
                # print(y_pred)
                results_df["Predicted"] = y_pred
                results_dict[kernel]    = results_df
            
                # evaluate model performance
                class_rpt_dict[kernel] = classification_report(y_test, y_pred)
                conf_mtx_dict[kernel]  = confusion_matrix(y_test, y_pred)
                
            if show:
                if not cv:
                    print(f"\nClassification report (kernel = {kernel}):")
                    print(classification_report(y_test, y_pred))
                    print(f"\nConfusion matrix (kernel = {kernel}):")
                    print(confusion_matrix(y_test, y_pred))
                if cv:
                    print(f"\nCross-validation scores (kernel = {kernel}):")
                    print(scores)
                    print(f"Mean accuracy: {np.mean(scores):.2f}\n")
        
        if not cv:
            # create df to summarize accuracy of SVM results by kernel
            accuracy_df = pd.DataFrame(columns = ["Correct", "Incorrect", "% correct"])
            
            for kernel, kernel_df in results_dict.items():
                num_cells = len(list(kernel_df.index))
                # print(f"{kernel}: {num_cells} cells")
                equal_mask = kernel_df["Actual"] == kernel_df["Predicted"]
                num_correct = equal_mask.sum()
                num_incorrect = (~equal_mask).sum()
                per_correct = (num_correct / num_cells) * 100
                accuracy_df.loc[kernel] = {"Correct": num_correct, "Incorrect": num_incorrect, "% correct": per_correct}
                
            results_dict["Accuracy summary"] = accuracy_df
            
            # create df to summarize behavior distribution of SVM results by kernel
            col_names = list(y_test.unique())
            total_cells = len(y_test)
            counts_df       = pd.DataFrame(columns = col_names)
            distribution_df = pd.DataFrame(columns = col_names)

            orig_behavior_ct = dict.fromkeys(col_names)
            orig_behavior_per = dict.fromkeys(col_names)
            for behavior in col_names:
                orig_behavior_mask = (y_test == behavior)
                num_cells = orig_behavior_mask.sum()
                per_cells = (num_cells / total_cells) * 100
                orig_behavior_ct[behavior] = num_cells
                orig_behavior_per[behavior] = per_cells
            counts_df.loc["original"] = orig_behavior_ct
            distribution_df.loc["original"] = orig_behavior_per

            for kernel, kernel_df in results_dict.items():
                kernel_behavior_ct = dict.fromkeys(col_names)
                kernel_behavior_per = dict.fromkeys(col_names)
                if kernel == "Accuracy summary":
                    continue
                
                for behavior in col_names:
                    SVM_behavior_mask = kernel_df["Predicted"] == behavior
                    num_cells = SVM_behavior_mask.sum()
                    per_cells = (num_cells / total_cells) * 100
                    kernel_behavior_ct[behavior] = num_cells
                    kernel_behavior_per[behavior] = per_cells
                counts_df.loc[kernel] = kernel_behavior_ct
                distribution_df.loc[kernel] = kernel_behavior_per
            
            results_dict["Distribution summary (counts)"] = counts_df
            results_dict["Distribution summary (%)"] = distribution_df
            
            return results_dict, class_rpt_dict, conf_mtx_dict
        else:
            return scores_dict
            
    else:
        svm_model      = SVC(kernel = kernel_type, **kwargs)
        
        # if performing cross-validation
        if cv:
            scores                 = cross_val_score(svm_model, X_SVM_df, y_SVM_df, cv = cv_val)
            svm_model.fit(X_SVM_df, y_SVM_df)
            y_pred                 = svm_model.predict(X_SVM_df)
            
        # if splitting into training/testing sets
        else:
            results_df          = pd.DataFrame(index = test_cells, columns = ["Actual", "Predicted"])
            if not results_df.index.equals(X_test.index):
                raise ValueError("Test and result cell indices do not match.")
            else:
                results_df["Actual"] = y_test
                
            svm_model.fit(X_train, y_train)
            y_pred                   = svm_model.predict(X_test)
            results_df["Predicted"]  = y_pred
         
        if not cv:
            # Evaluate model performance
            class_rpt      = classification_report(y_test, y_pred)
            conf_mtx       = confusion_matrix(y_test, y_pred)
        
        if show:
            if not cv:
                print("Classification report  (kernel = {kernel_type}):")
                print(classification_report(y_test, y_pred))
                print("\nConfusion matrix (kernel = {kernel_type}):")
                print(confusion_matrix(y_test, y_pred))
            if cv:
                print(f"\nCross-validation scores (kernel = {kernel_type}):")
                print(scores)
                print(f"\nMean accuracy: {np.mean(scores):.2f}\n")
        
        if not cv:
            return results_df, class_rpt, conf_mtx
        else:
            return scores
        

def test_SVM_params(X_SVM_df, y_SVM_df, random = False, **kwargs):
    """
    Test different SVM params using either grid or randomized search.

    Parameters
    ----------
    X_SVM_df : pd.DataFrame
        Feature matrix for SVM.
    y_SVM_df : pd.Series
        Target vector for SVM.
    random : bool, optional
        Whether to use GridSearch (False) or RandomizedSearch (True). The default is False.
    **param_grid : dict
        Parameter grid to be used in GridSearch.  Must be provided if random is False.
    **param_dist : dict
        Parameter space to be used in RandomizedSearch.  Must be provided if random is True.
        
    Raises
    ------
    ValueError
        Occurs if random is False and param_grid is not provided.
        Occurs if random is True and param_dist is not provided.

    Returns
    -------
    best_params : dict
        Dictionary of the best parameter values.
    best_score : flt
        Best accuracy score.

    """
    
    if not random:
        if "param_grid" not in kwargs:
            raise ValueError("param_grid must be provided if random is False.")
        param_grid = kwargs["param_grid"]
        search = GridSearchCV(estimator=SVC(), param_grid=param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    else:
        if "param_dist" not in kwargs:
            raise ValueError("param_dist must be provided if random is True.")
        param_dist = kwargs["param_dist"]
        search = RandomizedSearchCV(estimator=SVC(), param_distributions=param_dist, n_iter=10, cv=5, scoring="accuracy", n_jobs=-1)
    
    search.fit(X_SVM_df, y_SVM_df)
    
    best_params = search.best_params_
    best_score  = search.best_score_
    
    # Extract the best model
    svm_model   = search.best_estimator_
    
    search_type = "grid" if random is False else "random"
    
    print(f"Best parameters ({search_type}): {best_params}")
    print(f"Best score ({search_type}): {best_score}\n")
    
    return svm_model, best_params, best_score


#%% functions to visualize classification results

def plot_trace_descriptors(subcluster_traces_smooth, dict_trace_descriptors_SVM, cell_name = None, num_plots = 2):
    # flatten trace data
    all_traces_df   = flatten_trace_df(subcluster_traces_smooth)
    times           = list(all_traces_df.columns)
    
    # randomly select cell
    good_cell = False
    if cell_name == None:
            
        while not good_cell:
            # randomly select a cell if not provided
            cell_name = all_traces_df.sample(n = 1).index[0]
            tmt, cell = cell_name.split(maxsplit = 1)
            cell_descriptors = dict_trace_descriptors_SVM[tmt].loc[cell, :]
            # print(cell_name, cell_descriptors)
            
            if cell_descriptors["Max In"][0] == None or cell_descriptors["Max Out"][0] == None or cell_descriptors["Peaks"] < 2 or len(cell_descriptors["Local Max"]) < 2:
                print(f"Failed cell: {cell_name}")
                good_cell = False
            elif cell_descriptors["Max In"][0] < 30 or cell_descriptors["Max Out"][0] > 830:
                print(f"Failed cell: {cell_name}")
                good_cell = False
            else:
                good_cell = True
        
    # retrieve cell data
    cell_data = all_traces_df.loc[cell_name, :]
    # print(cell, cell_data)
    
    # plot trace
    fig, axs = plt.subplots(1, num_plots, figsize = (num_plots * 6, 5), sharex = True, sharey = True)
    
    for ax_num in range(num_plots):
        axs[ax_num].plot(times, cell_data, linewidth = 1, color = "black")
        axs[ax_num].set_title(cell_name)
        axs[ax_num].set_xlabel("Time (min)", fontsize = 14)
        axs[ax_num].set_ylabel("Nuclear Relish fraction (fold change)", fontsize = 14)
        axs[ax_num].tick_params(labelsize = 12)
        
        # set y-axis limits and ticks
        axs[ax_num].set_ylim(0.6, 2.0)
        axs[ax_num].set_yticks([0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
        
        # plot max value/time
        max_val = cell_descriptors["Max Value"]
        max_time = cell_descriptors["Max Time"]
        axs[ax_num].plot(max_time, max_val, "go", color = "black")
        axs[0].text(y = max_val + 0.03, x = max_time - 10, s = "[1, 2]", ha = "right", va = "center", fontsize = 10, color = "black")
        
        if num_plots == 2:
            axs[1].axvline(x = max_time, color = "red", linestyle = "dashed", linewidth = 0.5)
            axs[1].text(y = 2.0, x = max_time - 5, s = "Max time", ha ='center', va = 'top', fontsize = 8, color = 'red', fontstyle = "italic", rotation = 90)
            axs[1].axhline(y = max_val, color = "red", linestyle = "dashed", linewidth = 0.5)
            axs[1].text(y = max_val + 0.025, x = 860, s = "Max value", ha ='right', va = 'center', fontsize = 8, color = 'red', fontstyle = "italic")
        
        # plot half max value/time
        half_val = cell_descriptors["Half Max"]
        half_time = cell_descriptors["Half Time"]
        axs[ax_num].plot(half_time, half_val, "go", color = "black")
        axs[0].text(y = half_val + 0.03, x = half_time - 10, s = "[3, 4]", ha = "right", va = "center", fontsize = 10, color = "black")
        
        if num_plots == 2:
            axs[1].axvline(x = half_time, color = "green", linestyle = "dashed", linewidth = 0.5)
            axs[1].text(y = 2.0, x = half_time - 5, s = "Half max time", ha ='center', va = 'top', fontsize = 8, color = 'green', fontstyle = "italic", rotation = 90)
            axs[1].axhline(y = half_val, color = "green", linestyle = "dashed", linewidth = 0.5)
            axs[1].text(y = half_val + 0.025, x = 860, s = "Half max value", ha ='right', va = 'center', fontsize = 8, color = 'green', fontstyle = "italic")
        
        # shade area under curve
        axs[ax_num].fill_between(times, cell_data, color='lightgrey', alpha=0.5)
        axs[0].text(y = 0.8, x = times[-1] / 2, s = "[5]", ha = "center", va = "center", fontsize = 10, color = "black")
        
        if num_plots == 2:
            axs[1].fill_between(times, cell_data, color = 'lightblue', alpha = 0.5)
            axs[1].text(y = 0.8, x = times[-1] / 2, s = "Area", ha = "center", va = "center", fontsize = 10, color = "blue")
    
    # calculate tangent line for max rate in
    # derivative = np.gradient(cell_data, times)
    in_time = cell_descriptors["Max In"][0]
    in_rate = cell_descriptors["Max In"][1]
    in_idx  = np.argmin(np.abs(times - in_time))
    in_val  = cell_data[in_idx]
    
    # if in_idx < 0 or in_idx >= len(derivative):
    #     raise ValueError(f"{cell_name}: Index out of bounds for derivative array.")
    # else:
    #     print(f"{cell_name}: Index = {in_idx}, time = {in_time}, rate = {in_rate}")
    #     slope = derivative[in_idx]
    
    # plot the max rate in
    xin_tan = np.linspace(max(times[0], in_time - 50), min(times[-1], in_time + 50), 101)
    yin_tan = in_rate * (xin_tan - in_time) + in_val
    print(f"Tangent x-values (time): {xin_tan}")
    print(f"Tangent y-values (Relish): {yin_tan}")
    axs[0].plot(xin_tan, yin_tan, color = "black", linewidth = 0.5, linestyle = "dashed")
    
    # calculate tangent line for max rate out
    # derivative = np.gradient(cell_data, times)
    out_time = cell_descriptors["Max Out"][0]
    out_rate = cell_descriptors["Max Out"][1]
    out_idx  = np.argmin(np.abs(times - out_time))
    out_val  = cell_data[out_idx]
    
    # if in_idx < 0 or in_idx >= len(derivative):
    #     raise ValueError(f"{cell_name}: Index out of bounds for derivative array.")
    # else:
    #     print(f"{cell_name}: Index = {in_idx}, time = {in_time}, rate = {in_rate}")
    #     slope = derivative[in_idx]
    
    # plot the max rate in
    xout_tan = np.linspace(max(times[0], out_time - 50), min(times[-1], out_time + 50), 101)
    yout_tan = -out_rate * (xout_tan - out_time) + out_val
    print(f"Tangent x-values (time): {xout_tan}")
    print(f"Tangent y-values (Relish): {yout_tan}")
    axs[0].plot(xout_tan, yout_tan, color = "black", linewidth = 0.5, linestyle = "dashed")
    
    plt.tight_layout()
    plt.show()
    
    """
    Trace descriptors:
        1. Max value
        2. Max time
        3. Half max value
        4. Half max time
        5. Area under the curve
        6. Max rate in [time, value]
        7. Max rate out [time, value]
        8. # peaks
        9. Local maxima
    """
    

def plot_comp_traces(subcluster_traces_smooth, results_dict, treatment = "all", classified = True, collapse_imm = False):
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
    classified : bool, optional
        Whether or not the data consists of preclassified cells or not.  The default is True.
    collapse_imm : bool, optional
        Whether or not to collapse "Immediate with continued", "Immediate with plateau", and "Immediate with decay" into one "Immediate category."
        The default is False.

    Returns
    -------
    all_traces_df : pd.DataFrame
        Timecourse values for all cells in all treatments, clusters, and subclusters in subcluster_traces_smooth flattened into one df.

    """
    
    colors          = ["red", "orange", "purple", "green", "blue", "grey"]
    color_palette   = sns.color_palette(colors)
    behavior_colors = {"Ic": color_palette[0], "Ip": color_palette[1], "Id": color_palette[2], "G": color_palette[3], "D": color_palette[4], "N": color_palette[5]} if collapse_imm == False else {"I": color_palette[0], "G": color_palette[3], "D": color_palette[4], "N": color_palette[5]}
    behavior_keys   = {"Ic": "Immediate with continued", "Ip": "Immediate with plateau", "Id": "Immediate with decay", "G": "Gradual", "D": "Delayed", "N": "Nonresponsive"} if collapse_imm == False else {"I": "Immediate", "G": "Gradual", "D": "Delayed", "N": "Nonresponsive"}
    
    # flatten trace data
    all_traces_df   = flatten_trace_df(subcluster_traces_smooth)
    times           = list(all_traces_df.columns)
    
    for kernel, kernel_df in results_dict.items():
        if "summary" in kernel:
            continue
        # print(f"\n{kernel}:")
        
        # initialize figure
        if not collapse_imm:
            fig, axs    = plt.subplots(2, 3, figsize = (20, 10), sharex = True, sharey = True)
        else:
            fig, axs    = plt.subplots(2, 2, figsize = (15, 10), sharex = True, sharey = True)
        
        for n, (SVM_behavior, behavior_name) in enumerate(behavior_keys.items()):
            # axes indices
            i       = n % 2
            j       = n // 2
            
            behavior_traces = pd.DataFrame()
            num_cells       = 0
            
            if classified:
                for orig_behavior, orig_behavior_name in behavior_keys.items():
                    mask    = (kernel_df["Predicted"] == SVM_behavior) & (kernel_df["Actual"] == orig_behavior)
                    indices = kernel_df.index[mask].tolist()
                    # print(f"Cells predicted to be {behavior_name}: {indices}")
                    if treatment != "all":
                        indices = [index for index in indices if treatment in index]
                    
                    traces  = all_traces_df.loc[indices]
                    # display(traces)
                    behavior_traces = pd.concat([behavior_traces, traces], axis = 0)
                    
                    # plot traces
                    color   = behavior_colors[orig_behavior]
                    axs[i, j].plot(times, traces.T, linewidth = 0.5, color = color)
                
                num_cells   = behavior_traces.shape[0]
                
                # plot averages
                axs[i, j].plot(times, behavior_traces.mean(axis=0), color="black", linewidth=1.5)
            
            else:
                mask        = (kernel_df["Predicted"] == SVM_behavior)
                indices     = kernel_df.index[mask].tolist()
                if treatment != "all":
                    indices = [index for index in indices if treatment in index]
                
                num_cells   = len(indices)
                traces      = all_traces_df.loc[indices]
                color       = behavior_colors[SVM_behavior]
                axs[i, j].plot(times, traces.T, linewidth = 0.5, color = color)
                axs[i, j].plot(times, traces.mean(axis=0), color="black", linewidth=1.5)
                
            # set axes labels and title
            # title_color = behavior_colors[SVM_behavior]
            title_color = "black"
            axs[i, j].set_title(f"{behavior_name} (n = {num_cells})", fontsize = 16, color = title_color)
            axs[i, j].set_xlabel("Time (min)", fontsize = 14)
            axs[i, j].set_ylabel("Nuclear Relish fraction (fold change)", fontsize = 14)
            axs[i, j].tick_params(axis = "both", labelsize = 12)
        
        if classified:
            # create custom legend
            leg_lines   = [Line2D([0], [0], color = line_color, lw = 0.5) for line_color in list(behavior_colors.values())]
            leg_labels  = [behavior_name for behavior_name in list(behavior_keys.values())]
            # print(leg_lines)
            fig.legend(leg_lines, leg_labels, loc = "upper right", bbox_to_anchor = (1.05, 1), title = "Original classification", title_fontsize = 14, fontsize = 12)
        
        treatment_cells = len([cell for cell in list(kernel_df.index) if treatment in cell])
        if len(results_dict) > 1:
            total_cells = results_dict[list(results_dict.keys())[0]].shape[0]
            kernel_acc  = results_dict["Accuracy summary"].loc[kernel, "% correct"]
            if treatment == "all":
                plt.suptitle(f"SVM behavior classification (n = {total_cells}, kernel = {kernel})\nAccuracy: {kernel_acc:.3f}%", fontsize = 20)
            else:
                plt.suptitle(f"SVM behavior classification ({treatment}, n = {treatment_cells}, kernel = {kernel})\nAccuracy: {kernel_acc:.3f}%", fontsize = 20)
        else:
            if classified:
                total_cells = results_dict[list(results_dict.keys())[0]].shape[0]
                num_cells   = (results_dict[list(results_dict.keys())[0]]["Actual"] == results_dict[list(results_dict.keys())[0]]["Predicted"]).sum()
                kernel_acc  = (num_cells / total_cells) * 100
                if treatment == "all":
                    plt.suptitle(f"SVM behavior classification (n = {total_cells})\nAccuracy: {kernel_acc:.3f}%", fontsize = 20)
                else:
                    plt.suptitle(f"SVM behavior classification ({treatment}, n = {treatment_cells})\nAccuracy: {kernel_acc:.3f}%", fontsize = 20)
            else:
                total_cells = results_dict[list(results_dict.keys())[0]].shape[0]
                if treatment == "all":
                    plt.suptitle(f"SVM behavior classification (n = {total_cells})", fontsize = 20)
                else:
                    plt.suptitle(f"SVM behavior classification ({treatment}, n = {treatment_cells})", fontsize = 20)
        
        plt.tight_layout()
        plt.show()
        
    return all_traces_df
    

def percent_behaviors(dict_trace_descriptors, plot = None, remove_Ic = True):
    treatments = [treatment for treatment in list(dict_trace_descriptors.keys())]
    # print(treatments)
    treatments = sorted(treatments, key = lambda x: (x != '-PGN', int(x[:-1]) if x != '-PGN' else -1))
    # print(treatments)
    PGN_concs  = [tmt.strip("X") if "X" in tmt else 0 for tmt in treatments]
    # print(treatments)
    # print(PGN_concs)
    behavior_dict = {
        "N": "Nonresponsive",
        "D": "Delayed",
        "G": "Gradual",
        "Ic": "Immediate with continued",
        "Id": "Immediate with decrease",
        "Ip": "Immediate with plateau"} if not remove_Ic else {
            "N": "Nonresponsive",
            "D": "Delayed",
            "G": "Gradual",
            "I": "Immediate",
            "Id": "Immediate with decrease"}
    col_names = ["# cells"] + [behavior for behavior in behavior_dict]
    
    # print(treatments)
    # print(behaviors)
    # print(col_names)
    
    percents_df = pd.DataFrame(index = treatments, columns = col_names)
    
    for tmt, tmt_df in dict_trace_descriptors.items():
        # behavior_dict = dict.fromkeys(behaviors)
        total_cells = tmt_df.shape[0]
        # print(tmt, total_cells)
        
        percents_df.loc[tmt, "# cells"] = total_cells
        
        for behavior in col_names:
            if behavior == "# cells":
                continue
            
            behavior_name = behavior_dict[behavior]
            filtered_data = tmt_df[tmt_df['Behavior'] == behavior]
            sum_cells = filtered_data.shape[0]
            # print(tmt, behavior, sum_cells)
            percent_cells = (sum_cells / total_cells) * 100
            
            percents_df.loc[tmt, behavior] = percent_cells
        
    if plot is not False:
        colors          = ["#DC143C", "#FF6F61", "indigo", "green", "dodgerblue", "grey"]
        color_palette   = sns.color_palette(colors)
        treatment_color = {
            "Immediate with continued": color_palette[0], 
            "Immediate with plateau": color_palette[1],
            "Immediate with decrease": color_palette[2],
            "Gradual": color_palette[3],
            "Delayed": color_palette[4],
            "Nonresponsive": color_palette[5]} if not remove_Ic else {
                "Immediate": color_palette[0], 
                "Immediate with decrease": color_palette[1],
                "Gradual": color_palette[2],
                "Delayed": color_palette[4],
                "Nonresponsive": color_palette[5]}
        color_list = [treatment_color[behavior_dict[behavior]] for behavior in col_names if behavior != "# cells"]
        
        cols_to_plot = percents_df.columns[1:]
        
        if plot == "vert":
            ax = percents_df[cols_to_plot].plot.bar(stacked=True, figsize=(6, 10), color = color_list)
        
            plt.xlabel(r'[PGN] ($\mu$g/mL)', fontsize = 16)
            plt.ylabel('% cells', fontsize = 16)
            # plt.xticks(rotation = 0)
            plt.xticks(np.arange(len(PGN_concs)), PGN_concs, rotation = 0)
            ax.tick_params(axis = "both", labelsize = 14)
            # plt.title('Distribution of Relish dynamics', fontsize = 16)
            
            for i, treatment in enumerate(treatments):
                x = percents_df.columns[1:]
                y = percents_df.loc[treatment, x].values
                total_cells = percents_df.loc[treatment, "# cells"]
                
                ax.text(x = i, y = y.sum() + 1, s = f"({total_cells})", ha = 'center', va = 'bottom', fontsize = 14, color = 'black', fontstyle = "italic")
            
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
            plt.savefig(f"behavior_distribution_{formatted_date}", dpi = 700)
            plt.show()
            
        elif plot == "hor":
            ax = percents_df[cols_to_plot].plot.barh(stacked=True, figsize=(10, 6), color = color_list)
        
            plt.ylabel(r'[PGN] ($\mu$g/mL)', fontsize = 16)
            plt.xlabel('% cells', fontsize = 16)
            plt.yticks(np.arange(len(PGN_concs)), PGN_concs, rotation = 0)
            ax.tick_params(axis = "both", labelsize = 14)
            # plt.title('Distribution of Relish dynamics', fontsize = 16)
            
            for i, treatment in enumerate(treatments):
                y = percents_df.columns[1:]
                x = percents_df.loc[treatment, y].values
                total_cells = percents_df.loc[treatment, "# cells"]
                
                ax.text(y = i - 0.1, x = x.sum() + 2, s=f"({total_cells})", ha='center', va='bottom', fontsize=10, color='black', fontstyle = "italic", rotation = -90)
            
            plt.legend(bbox_to_anchor=(1, 1.02), loc='upper left')
            plt.savefig(f"behavior_distribution_{formatted_date}", dpi = 700)
            plt.show()
        
    return percents_df


def plot_test_trace(cell_categories_df, subcluster_traces_smooth, dict_trace_descriptors_SVM, random_cell = True, cell_names = None, collapse_imm = False, show_grid = False):
    if cell_names != None and random == True:
        raise ValueError("random must be False if cell_name is provided.")  
    
    # plot colors
    colors          = ["#DC143C", "#FF6F61", "purple", "green", "blue", "grey"]
    color_palette   = sns.color_palette(colors)
    behavior_colors = {"Ic": color_palette[0], "Ip": color_palette[1], "Id": color_palette[2], "G": color_palette[3], "D": color_palette[4], "N": color_palette[5]} if collapse_imm == False else {"I": color_palette[0], "G": color_palette[3], "D": color_palette[4], "N": color_palette[5]}
    behavior_keys   = {"Ic": "Immediate with continued", "Ip": "Immediate with plateau", "Id": "Immediate with decay", "G": "Gradual", "D": "Delayed", "N": "Nonresponsive"} if collapse_imm == False else {"I": "Immediate", "G": "Gradual", "D": "Delayed", "N": "Nonresponsive"}
    behavior_rev    = {v: k for k, v in behavior_keys.items()}
    
    # convert cell_categories_df to dict and reorder
    cell_categories_dict = cell_categories_df.set_index("Category")["Cells"].to_dict()
    plot_order = ["Nonresponsive", "Gradual", "Immediate with plateau", "Delayed", "Immediate with decay", "Immediate with continued"]
    cell_categories_dict = {key: cell_categories_dict[key] for key in plot_order}
    
    # flatten trace data
    all_traces_df   = flatten_trace_df(subcluster_traces_smooth)
    times           = list(all_traces_df.columns)
    
    # create plot
    fig, axs = plt.subplots(2, 3, figsize = (20, 12), sharex = True, sharey = True)
    
    for n, (behavior, behavior_cells) in enumerate(cell_categories_dict.items()):
        # print(behavior_cells)
        # axes indices
        i       = n % 2
        j       = n // 2
        
        behavior_abbr   = behavior_rev[behavior]
        behavior_color  = behavior_colors[behavior_abbr]
        
        if random_cell:
            cell_name = random.choice(behavior_cells)
            # cell_name = all_traces_df.index.to_series().sample(n = 1).index[0]
        else:
            cell_name = cell_names[behavior_abbr]
        print(cell_name)
        tmt, cell = cell_name.split(maxsplit = 1)
        # print(f"{cell_name}\n{tmt}: {cell}")
        
        # retrieve timecourse data
        cell_trace = all_traces_df.loc[cell_name].values
        cell_data  = dict_trace_descriptors_SVM[tmt].loc[cell]
        print(f"{cell_data}\n")
        
        # plot trace
        axs[i, j].plot(times, cell_trace, color = behavior_color)
        
        # set title and labels
        axs[i, j].set_title(f"{behavior}\n{cell_name}", fontsize = 16, color = behavior_color)
        axs[i, j].set_xlabel("Time (min)", fontsize = 14)
        axs[i, j].set_ylabel("Nuclear Relish fraction (fold change)", fontsize = 14)
        axs[i, j].tick_params(axis = "both", labelsize = 12)
        
        # show major gridlines if show_grid is True
        if show_grid:
            axs[i, j].grid(which = "major", color = "lightgrey", linestyle = "dashed", linewidth = 0.5)
        
            # plot dashed lines at t = 30 and y = 1
            axs[i, j].axvline(x = 30, color = "black", linewidth = 0.5)
            axs[i, j].axhline(y = 1, color = "black", linewidth = 0.5)
        
        # set y-axis limits and ticks
        axs[i, j].set_ylim(0.6, 2.0)
        axs[i, j].set_yticks([0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
        
        # plot horizontal line at 1.2
        axs[i, j].axhline(y = 1.2, color = "grey", linestyle = "dashed", linewidth = 0.5)
        
        max_val = cell_data["Max Value"]
        max_time = cell_data["Max Time"]
        
        if "I" in behavior_abbr:
            # # plot max lines
            # axs[i, j].axvline(x = max_time, color = "grey", linestyle = "dashed", linewidth = 0.5)
            # axs[i, j].text(y = 1.9, x = max_time - 5, s = f"t = {max_time}", ha = 'center', va = 'center', fontsize = 8, color = 'grey', fontstyle = "italic", rotation = 90)
            # max_time_line = Line2D([max_time, max_time], [0, max_val], color = "black", linewidth = 0.5)
            # max_val_line  = Line2D([0, max_time], [max_val, max_val], color = "black", linewidth = 0.5)
            # axs[i, j].add_line(max_time_line)
            # axs[i, j].add_line(max_val_line)
            axs[i, j].plot(max_time, max_val, "go", color = "black")
            
            # add "Max value"
            axs[i, j].text(y = max_val + 0.03, x = max_time - 5, s = "Max", ha = 'right', va = 'center', fontsize = 10, color = 'black')
        
        # add vertical lines at t = 150, 430
        axs[i, j].axvline(x = 150, color = "grey", linestyle = "dashed", linewidth = 0.5)
        axs[i, j].text(y = 1.9, x = 145, s = "t = 150", ha ='center', va = 'center', fontsize = 8, color = 'grey', fontstyle = "italic", rotation = 90)
        t150_val = cell_trace[150]
        # axs[i, j].plot(150, t150_val, "go", color = "black")
        # axs[i, j].text(y = t150_val + 0.03, x = 145, s = f"$t_{{150}}$", ha = "right", va = "center", fontsize = 10, color = "black")
        
        # add "Initial behavior" label
        axs[i, j].text(y = 1.75, x = 75, s = "Initial\nbehavior", ha = 'center', va = 'center', fontsize = 10, color = 'black')
        
        if "I" not in behavior_abbr:
            axs[i, j].axvline(x = 430, color = "grey", linestyle = "dashed", linewidth = 0.5)
            axs[i, j].text(y = 1.9, x = 425, s = "t = 430", ha ='center', va = 'center', fontsize = 8, color = 'grey', fontstyle = "italic", rotation = 90)
            t430_val = cell_trace[430]
            
            if behavior_abbr != "N":
                axs[i, j].plot(430, t430_val, "go", color = "black")
                axs[i, j].text(y = t430_val + 0.03, x = 425, s = f"$t_{{430}}$", ha = "right", va = "center", fontsize = 10, color = "black")
    
        if cell_data["Local Max"] != []:
            peak1_time = cell_data["Local Max"][0][0]
            peak1_val  = cell_data["Local Max"][0][1]
            # print(peak1_time)
        else:
            peak1_time = max_time
            peak1_val = max_val
          
        # if peak1_time > 150:
        #     t150_line = Line2D([0, 150], [t150_val, t150_val], color = "grey", linestyle = "dashed", linewidth = 0.5)
        #     axs[i, j].add_line(t150_line)
        # else:
        #     peak1_line = Line2D([0, peak1_time], [peak1_val, peak1_val], color = "grey", linestyle = "dashed", linewidth = 0.5)
        #     axs[i, j].add_line(peak1_line)
        
        if "I" in behavior_abbr:
            # plot first peak point
            if peak1_time != max_time and peak1_val != max_val:
                axs[i, j].axvline(x = peak1_time, color = "grey", linestyle = "dashed", linewidth = 0.5)
                axs[i, j].text(y = 1.9, x = peak1_time - 5, s = f"t = {peak1_time}", ha = 'center', va = 'center', fontsize = 8, color = 'grey', fontstyle = "italic", rotation = 90)
                axs[i, j].plot(peak1_time, peak1_val, "go", color = "black")
                axs[i, j].text(y = peak1_val + 0.03, x = peak1_time - 5, s = "First peak", ha = 'right', va = 'center', fontsize = 10, color = 'black')
            
            # shade -0.2 from first peak
            axs[i, j].fill_between(x = [peak1_time, 860], y1 = peak1_val, y2 = peak1_val - 0.2, color = "lightgrey", alpha = 0.5)
            # peak1_line = Line2D([peak1_time, 860], [peak1_val, peak1_val], color = "grey", linestyle = "dashed", linewidth = 0.5)
            # axs[i, j].add_line(peak1_line)
            # axs[i, j].text(y = peak1_val, x = 860 + 20, s = "±0.2", ha = 'center', va = 'center', fontsize = 8, color = 'grey', rotation = 90)
            axs[i, j].text(y = 1.75, x = ((860 - peak1_time) / 2) + peak1_time, s = "Long-term\nbehavior", ha = 'center', va = 'center', fontsize = 10, color = 'black')
            
            # add height arrow
            fin_val = cell_trace[-1]
            
            axs[i, j].annotate('', xy = (870, fin_val), xytext = (870, max_val),
                arrowprops = dict(
                    arrowstyle = '|-|',
                    color = behavior_color,
                    linewidth = 0.5,
                    mutation_scale = 2.5))
            
            # Add text annotation
            axs[i, j].text(890, (fin_val + max_val) / 2, "Δ FC", color = behavior_color, ha = 'center', va = 'center', fontsize = 10, rotation = 90)
            
        # # plot long-term behavior line
        # fin_val = cell_trace[-1]
        # times_subset = np.array([peak1_time, 860])
        # vals_subset = np.array([peak1_val, fin_val])
        # lt_slope, lt_int = np.polyfit(times_subset, vals_subset, 1)
        # lt_line = lt_slope * np.array(times_subset) + lt_int
        # axs[i, j].plot(times_subset, lt_line, color = "grey", linewidth = 0.5)
        
        # # if you want to plot best-fit line
        # idxs_subset = np.where((np.array(times) >= peak1_time) & (np.array(times) <= 860))[0]
        # times_subset = np.array(times)[idxs_subset]
        # vals_subset = cell_trace[idxs_subset]
        
        # lt_slope, lt_int = np.polyfit(times_subset, vals_subset, 1)
        # lt_line = lt_slope * np.array(times_subset) + lt_int
        # ax.plot(times_subset, lt_line, color = "grey", linewidth = 0.5)

    plt.tight_layout()        
    plt.show()
    

#%% functions to test significance

def chisquare_sigtest(results_dict):
    """
    

    Parameters
    ----------
    results_dict : TYPE
        DESCRIPTION.

    Returns
    -------
    p_dict : TYPE
        DESCRIPTION.
        p-value ≤ 0.05 --> Statistically significant difference (reject null hypothesis)

    """
    # initialize dictionary to store p-values
    p_dict = {}
    
    distribution_df = results_dict["Distribution summary (counts)"]
    f_exp  = np.array([count for count in list(distribution_df.loc["original"])])
    # print(f"Expected: {f_exp}")
    
    for kernel in list(distribution_df.index):
        if kernel == "original":
            continue
        
        f_obs           = np.array([count for count in list(distribution_df.loc[kernel])])
        # print(f"Observed ({kernel}): {f_obs}")
        chi_stat, p_val = chisquare(f_obs, f_exp)
        # print(f"chi_stat = {chi_stat}, p_val = {p_val}")
        
        p_dict[kernel]  = p_val
    
    return p_dict
        

#%% functions to build, train, and run SVM on complete dataset

def plot_sample_and_all_traces(subcluster_traces_smooth, results_dict, cell_categories_df, dict_trace_descriptors_SVM, treatment = "all", random_cell = True, cell_names = None, show_title = False, stim_time = 30):
    if cell_names != None and random == True:
        raise ValueError("random must be False if cell_name is provided.")  
    
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
        # title_color = "black"
        # axs[1, n].set_title(f"{behavior_name} (n = {num_cells})", fontsize = 16, color = title_color)
        # axs[1, n].set_xlabel("Time (min)", fontsize = 14)
        # axs[1, n].set_ylabel("Nuclear Relish fraction (fold change)", fontsize = 14)
        axs[1, n].tick_params(axis = "both", labelsize = 14)
        axs[1, n].text(y = 1.9, x = 800, s = f"n = {num_cells}", ha = "center", va = "center", fontsize = 16)
        
        # highlight pre- and post-stim times
        y_min, y_max = axs[1, n].get_ylim()
        prestim  = patches.Rectangle((0, 0.6), 30, 0.025, color = "black")
        # poststim = patches.Rectangle((30, 0.6), times[-1] - 30, 0.025, color = "white", edgecolor = "black")
        hatch = patches.Rectangle((30, 0.6), times[-1] - 30, 0.025, edgecolor="black", facecolor="none", hatch='///')
        axs[1, n].add_patch(prestim)
        # axs[1, n].add_patch(poststim)
        axs[1, n].add_patch(hatch)
        # axs[1, n].add_patch(patches.Rectangle((0, 0.6), 30, 0.025, color = "darkgrey", alpha = 0.5))
        # axs[1, n].add_patch(patches.Rectangle((30, 0.6), times[-1], 0.025, color = "black", alpha = 0.5))
        # axs[1, n].arrow(stim_time, 0.65, 0, -0.04, length_includes_head = True, )
        
        
        # plot test trace for each behavior
        behavior_cells = cell_categories_dict[behavior_name]
        if random_cell:
            cell_name = random.choice(behavior_cells)
            # cell_name = all_traces_df.index.to_series().sample(n = 1).index[0]
        else:
            cell_name = cell_names[SVM_behavior]
        print(cell_name)
        tmt, cell = cell_name.split(maxsplit = 1)
        # print(f"{cell_name}\n{tmt}: {cell}")
        
        # retrieve timecourse data
        cell_trace = all_traces_df.loc[cell_name].values
        cell_data  = dict_trace_descriptors_SVM[tmt].loc[cell]
        print(f"{cell_data}\n")
        
        # plot trace
        axs[0, n].plot(times, cell_trace, color = behavior_color)
        
        # set title and labels
        # axs[0, n].set_title(f"{behavior_name}\n{cell_name}", fontsize = 16, color = behavior_color)
        axs[0, n].set_title(f"{behavior_name}", fontsize = 20, color = behavior_color)
        # axs[0, n].set_xlabel("Time (min)", fontsize = 14)
        # axs[0, n].set_ylabel("Nuclear Relish fraction (fold change)", fontsize = 14)
        axs[0, n].tick_params(axis = "both", labelsize = 14)
        
        # set y-axis limits and ticks
        axs[0, n].set_ylim(0.6, 2.0)
        axs[0, n].set_yticks([0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
        
        # plot horizontal line at 1.2
        axs[0, n].axhline(y = 1.2, color = "grey", linestyle = "dashed", linewidth = 0.5)
        
        max_val = cell_data["Max Value"]
        max_time = cell_data["Max Time"]
        
        if SVM_behavior != "N":
            # add vertical lines at t = 150
            axs[0, n].axvline(x = 150, color = "grey", linestyle = "dashed", linewidth = 0.5)
            axs[0, n].text(y = 1.85, x = 140, s = "t = 150", ha ='center', va = 'center', fontsize = 14, color = 'grey', fontstyle = "italic", rotation = 90)
            t150_val = cell_trace[150]
            
            # add "Initial behavior" label
            axs[0, n].text(y = 0.8, x = 75, s = "Initial\nbehavior", ha = 'center', va = 'center', fontsize = 14, color = 'black')
        
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
            axs[0, n].fill_between(x = [peak1_time, 860], y1 = peak1_val, y2 = peak1_val - 0.2, color = "lightgrey", alpha = 0.5)
            axs[0, n].text(y = 0.8, x = ((860 - peak1_time) / 2) + peak1_time, s = "Long-term\nbehavior", ha = 'center', va = 'center', fontsize = 14, color = 'black')
            
            # add height arrow
            fin_val = cell_trace[-1]
            
            axs[0, n].annotate('', xy = (870, fin_val), xytext = (870, max_val),
                arrowprops = dict(
                    arrowstyle     = '|-|',
                    color          = behavior_color,
                    linewidth      = 0.5,
                    mutation_scale = 2.5))
            
            # Add text annotation
            axs[0, n].text(895, (fin_val + max_val) / 2, "Δ FC", color = behavior_color, ha = 'center', va = 'center', fontsize = 14, rotation = 90)
        
        else:
            
            if SVM_behavior != "N":
                axs[0, n].axvline(x = 430, color = "grey", linestyle = "dashed", linewidth = 0.5)
                axs[0, n].text(y = 1.85, x = 420, s = "t = 430", ha ='center', va = 'center', fontsize = 14, color = 'grey', fontstyle = "italic", rotation = 90)
                t430_val = cell_trace[430]
                
                # axs[0, n].plot(430, t430_val, "go", color = "black")
                # axs[0, n].text(y = t430_val + 0.03, x = 425, s = f"$t_{{430}}$", ha = "right", va = "center", fontsize = 10, color = "black")
                
                sig_filter = [(index, value) for index, value in enumerate(cell_trace) if value >= 1.2]
                sig_time, sig_val = sig_filter[0]
                # print(f"{behavior_name} ({cell_name}): [{sig_val}, {sig_time}]")
                axs[0, n].plot(sig_time, sig_val, "go", color = "black")
                axs[0, n].text(y = sig_val + 0.03, x = sig_time - 5, s = "FC ≥ 1.2", ha = "right", va = "center", fontsize = 14, color = "black")
    
    fig.supxlabel("Time (min)", fontsize = 16)
    # fig.supylabel("Nuclear Relish fraction (fold change)", fontsize = 14)
    fig.text(0, 0.5, 'Nuclear Relish fraction (fold change)', va='center', ha='center', fontsize=16, rotation='vertical')
    
    if show_title:
        treatment_cells = len([cell for cell in list(results_df.index) if treatment in cell])
        total_cells     = all_traces_df.shape[0]
        if treatment == "all":
            plt.suptitle(f"SVM behavior classification (n = {total_cells})", fontsize = 20)
        else:
            plt.suptitle(f"SVM behavior classification ({treatment}, n = {treatment_cells})", fontsize = 20)      
    
    plt.tight_layout(pad = 2)     
    plt.savefig(f"classified_traces_{formatted_date}", dpi = 700)
    plt.show()
    

def run_SVM(subcluster_traces_smooth, dict_trace_descriptors, cell_categories_df, param_grid, treatment = "all", remove_Ic = True, collapse_imm = False, random = False, cell_names = None, keep_SVM = True, return_vals = True):
    if cell_names != None and random == True:
        raise ValueError("random must be False if cell_name is provided.")  
    
    tmts = [tmt for tmt in subcluster_traces_smooth]
    
    # remove "Immediate with continued" behavior category
    if remove_Ic:
        cell_categories_df.loc[cell_categories_df["Category"].isin(["Immediate with plateau", "Immediate with continued"]), "Category"] = "Immediate"
        cell_categories_df_noIc = cell_categories_df.groupby("Category", as_index = False).agg(lambda x: [item for sublist in x for item in sublist])
        # display(cell_categories_df_noIc)
    
    # prep dict to store results
    results_dict = dict.fromkeys(["Classified cells", "Unclassified cells", "All cells"])
    
    # extract data for classified cells
    if not remove_Ic:
        dict_trace_descriptors_subset                      = trace_descriptor_subset(dict_trace_descriptors, cell_categories_df)
    else:
        dict_trace_descriptors_subset                      = trace_descriptor_subset(dict_trace_descriptors, cell_categories_df_noIc)
        
    # display(dict_trace_descriptors_subset)
    df_descriptor_vals_train, X_df_train, y_ser_train      = prep_descriptor_vals_SVM(dict_trace_descriptors_subset, collapse_imm = collapse_imm)
    df_descriptor_vals_all, X_df_all, y_ser_all            = prep_descriptor_vals_SVM(dict_trace_descriptors, collapse_imm = collapse_imm)
    
    # update "Behavior" columns of non-classified cells to None
    classified_cells                                       = set(df_descriptor_vals_train.index)
    df_descriptor_vals_all.loc[~df_descriptor_vals_all.index.isin(classified_cells), "Behavior"] = None
    y_ser_all.loc[~y_ser_all.index.isin(classified_cells)] = None
    # print(df_descriptor_vals_all)
    # print(y_ser_all)
    
    # extract data for non-classified cells
    df_descriptor_vals_unk                                 = df_descriptor_vals_all[df_descriptor_vals_all["Behavior"].isna()]
    unclassified_cells                                     = set(df_descriptor_vals_unk.index)
    X_df_unk                                               = X_df_all.loc[X_df_all.index.isin(unclassified_cells)]
    # print(df_descriptor_vals_unk)
    # print(X_df_unk)
    
    # train and optimize SVM on classified cells
    svm_model, best_params, best_score                     = test_SVM_params(X_df_train, y_ser_train, param_grid = param_grid, random = random)
    y_pred_train                                           = svm_model.predict(X_df_train)
    # print(f"y_pred_train (len = {len(y_pred_train)}): {y_pred_train}")
    
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
    if not remove_Ic:
        # plot traces for classified cells
        all_traces_df_train                                = plot_comp_traces(subcluster_traces_smooth, results_dict_train, treatment = treatment, collapse_imm = collapse_imm)
    
        # plot traces for unclassified cells
        all_traces_df_unk                                  = plot_comp_traces(subcluster_traces_smooth, results_dict_unk, classified = False, treatment = treatment, collapse_imm = collapse_imm)
    
        # plot traces for all cells
        results_dict_all                                   = {}
        results_dict_all[kernel]                           = results_df
        all_traces_df                                      = plot_comp_traces(subcluster_traces_smooth, results_dict_all, classified = False, treatment = treatment, collapse_imm = collapse_imm)
    
    else:
        plot_sample_and_all_traces(subcluster_traces_smooth, results_dict, cell_categories_df, dict_trace_descriptors_SVM, random_cell = False, cell_names = cell_names)
    
    # plot stacked bar plot
    percents_df = percent_behaviors(dict_trace_descriptors_SVM, plot = "hor", remove_Ic = True)
    
    if return_vals:
        return dict_trace_descriptors_SVM, svm_model, results_dict, percents_df


#%% test calls

treatments = ["-PGN", "1X", "10X", "100X"]

"""
Description of SVM parameters:
    C:            Penalty parameter of the error term (tradeoff between smooth decision boundary and correct classification of training points)
                  High C     --> High error
    gamma:        Kernel coefficient for "rbf", "poly", and "sigmoid" (curvature weight in decision boundary)
                  High gamma --> High curvature
    kernel:       Kernel type
    class_weight: Troubleshoot unbalanced data sampling
    
Source: https://kopaljain95.medium.com/how-to-improve-support-vector-machine-9561ab96ed18
"""

test_param_grid = {"C": [0.1, 1, 10, 100],
                   "gamma": [0.1, 0.01, 0.001, 0.0001],
                   "kernel": ["linear", "rbf", "poly", "sigmoid"],
                   "degree": [2, 3, 4]}

test_param_dist = {"C": uniform(loc = 0.1, scale = 1.0),
                   "gamma": [0.1, 0.01, 0.001, 0.0001],
                   "kernel": ["linear", "rbf", "poly", "sigmoid"],
                   "degree": randint(2, 5)}

test_cells2     = {"N": "1X Cell 20240801-2-112",
                   "G": "1X Cell 20240410-2-81",
                   "I": "10X Cell 20240321-1-150",
                   "D": "10X Cell 20240801-3-75",
                   "Id": "100X Cell 20240321-1-57"}


# # run complete SVM (trace sorting v2, NO Ic, 03.05, 03.21, 04.10, 08.01 RESEGMENTED data)
goodcomp3_dict_trace_descriptors_SVM_noIc, goodcomp3_SVM_model_noIc, goodcomp3_SVM_results_dict_noIc, goodcomp3_percent_behaviors_df_SVM_noIc     = run_SVM(goodcomp3_subcluster_traces_div_smooth, goodcomp3_dict_trace_descriptors, goodcomp_trace_sorting_cell_categories_v2, param_grid = test_param_grid, random = False, cell_names = test_cells2)
