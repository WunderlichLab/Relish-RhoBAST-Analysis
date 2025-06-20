# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 09:22:24 2024

Author: Emma Rits
Date: Thursday, June 5, 2025
Description: Step 9 - Relish timecourse data preprocessing

"""

#%% import packages
from collections import defaultdict
import copy
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex, colorConverter
import numpy as np
import pandas as pd
import pickle
import re
import scipy.cluster.hierarchy as hac
from scipy.cluster.hierarchy import dendrogram, fcluster
from scipy.integrate import trapezoid
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import linregress
from statistics import mean
import time

datetime_str = time.strftime("%m%d%y_%H:%M")


#%% functions to run clustering

def fillna_custom(df):
    """
    Parameters
    ----------
    df : dataframe
        Dataframe containing Relish ratio timecourse data

    Returns
    -------
    df : dataframe
        Returns original dataframe with NaNs replaced with average of three nearest non-NaNs
    """
    
    for column in df.columns:
        nan_indices = df[df[column].isna()].index
        for idx in nan_indices:
            non_nan_values = df[column][[i for i in range(idx - 3, idx + 4) if i != idx and i >= 0 and i < len(df)]].dropna()
            df.at[idx, column] = non_nan_values.mean() if not non_nan_values.empty else 0
    return df


def cluster_compiled(root_dir, dict_intensities_file_name):
    """
    Parameters
    ----------
    root_dir : str
        Root directory name for current dataset
    type_dict_intensities : str
        Name of dict_intensities file (Step 7 output) to be imported
    
    Returns
    -------
    dict_comp : dict
        Dictionary of dfs containing ratio timecourse values for all PGN treatments compiled for all dates
            dict_comp:             Original (with time column)
            dict_comp_smooth:      Smoothed (with time column)
            dict_comp_smooth_copy: Smoothed (without time column)
    Z_dict : dict
        Dictionary of arrays containing linkage matrices for PGN, 1X, 10X, and 100X
    """
    
    # retrieve dictionary from file
    dict_intensities  = import_data(root_dir, "dict_intensities_alldatasets_goodcells_areas_101024", file_type = "")
    dates             = list(dict_intensities.keys())
    treatments        = ["-PGN", "1X", "10X", "100X"]
    dict_comp         = dict.fromkeys(dates)
    dict_comp_smooth  = dict.fromkeys(treatments)
    dict_comp_smooth_copy = dict.fromkeys(treatments)
    trunc_time        = None
    
    for date, date_data in dict_intensities.items():
        for viewframe, viewframe_data in date_data.items():
            for cell, cell_data in viewframe_data["ratio"].items():
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
            ratio_subset      = []
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
                else:
                    continue
                
                dct               = dict_subset[tif_key]
                ratio_reps        = []                      # list of ratio values across all three replicates
                cell_name_reps    = []                      # list of cell names across all three replicates
    
                for key in list(dct["ratio"].keys()): # looping through all cells
                    ratio         = []                      # list of ratio values across current replicate
                    
                    value         = dct["ratio"][key]       # list of ratio values for current cell
                    ratio        += value
                    
                    cell_no       = int(key.strip("Cell ")) # cell number of current cell
                    cell_name     = "Cell " + str(date_str) + "-" + str(viewframe_no) + "-" + str(cell_no)
                    cell_name_reps.append(cell_name)
                    
                    # get pre-stimulus values
                    pre_stim      = [value[0][1], value[1][1], value[2][1], value[3][1]]
                    avg           = mean(pre_stim)
                    
                    # normalize values and update list
                    data          = [[item[0], item[1] / avg] for item in ratio]
                    ratio_reps   += [data]
            
                # add replicate data to list
                ratio_subset     += ratio_reps
                cell_name_subset += cell_name_reps
            
            # create dict with all values (across three replicates for each dates) for each treatment
            reps_dict        = dict(zip(cell_name_subset, ratio_subset))
    
            # replace NaNs
            temp_df           = pd.DataFrame(reps_dict)              # convert to df
            temp_df           = fillna_custom(temp_df)                # replace NaNs
            dates_dict        = pd.DataFrame.to_dict(temp_df)         # convert back to dict
            
            # associate each dict with its corresponding treatment
            subsets_dict[treatment] = dates_dict
                
        # create dataframe
        temp_df_ratio      = pd.DataFrame(subsets_dict)
        # print(temp_df_ratio)
        
        # make dataframe (by treatment); "copies" all have time column deleted
        df_ratio_PGN       = dataframe_treat(temp_df_ratio, "-PGN")
        df_ratio_PGN_org, df_ratio_PGN_copy      = cull_nans(df_ratio_PGN)
        df_ratio_PGN_copy  = df_ratio_PGN_copy.transpose()
        df_ratio_1X        = dataframe_treat(temp_df_ratio, "1X")
        df_ratio_1X_org, df_ratio_1X_copy        = cull_nans(df_ratio_1X)
        df_ratio_1X_copy   = df_ratio_1X_copy.transpose()
        df_ratio_10X       = dataframe_treat(temp_df_ratio, "10X")
        df_ratio_10X_org, df_ratio_10X_copy      = cull_nans(df_ratio_10X)
        df_ratio_10X_copy  = df_ratio_10X_copy.transpose()
        df_ratio_100X      = dataframe_treat(temp_df_ratio, "100X")
        df_ratio_100X_org, df_ratio_100X_copy    = cull_nans(df_ratio_100X)
        df_ratio_100X_copy = df_ratio_100X_copy.transpose()
        
        df_dict_org        = {"-PGN": df_ratio_PGN_org, "1X": df_ratio_1X_org, "10X": df_ratio_10X_org, "100X": df_ratio_100X_org}
        df_dict_copy       = {"-PGN": df_ratio_PGN_copy, "1X": df_ratio_1X_copy, "10X": df_ratio_10X_copy, "100X": df_ratio_100X_copy}
        
        times              = list(df_ratio_PGN_org["Time"])
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
    
    # run clustering
    Z_PGN            = hac.linkage(dict_comp_smooth_copy["-PGN"], method='ward', metric='euclidean')
    Z_1X             = hac.linkage(dict_comp_smooth_copy["1X"], method='ward', metric='euclidean')
    Z_10X            = hac.linkage(dict_comp_smooth_copy["10X"], method='ward', metric='euclidean')
    Z_100X           = hac.linkage(dict_comp_smooth_copy["100X"], method='ward', metric='euclidean')

    # make dictionary of clustering data
    Z_dict           = {"-PGN": Z_PGN, "1X": Z_1X, "10X": Z_10X, "100X": Z_100X}

    return dict_comp, dict_comp_smooth, dict_comp_smooth_copy, Z_dict


#%% functions to process dfs

def dataframe_treat(df, treatment):
    """
    Parameters
    ----------
    df : dataframe
        df containing dicts of ratio values from timecourse data
    treatment : str
        Type of treatment (-PGN, 1X, 10X, 100X) or "all"
    
    Returns
    ----------
    df_ratio : dataframe
        Dataframe containing unzipped timecourse data
    """
    if treatment != "all":    
        # make dataframe (by treatment)
        dict_ratio      = df[treatment]                       # dictionary of values corresponding to that treatment
        sub_ratio       = dict_ratio.items()                  # retrieve items
        [*idxs_ratio], [*dict_ratio] = zip(*sub_ratio)        # unzip
        
        cells_ratio     = []
        df_ratio_treat  = pd.DataFrame()
    
        for i in range(len(dict_ratio)): # iterate over all cells in all three replicates
            cell        = idxs_ratio[i]                       # get cell name
            dct         = dict_ratio[i]                       # get dict of values
            
            if type(dct) is dict: # check that dct is not NaN
                cells_ratio.append(cell)                      # add cell name to master list
                lst     = list(dct.values())                  # get list of ratio values [time, value]
                
                temp_df = pd.DataFrame(lst, columns=["Time", cell])
                        
                if df_ratio_treat.empty: # if df is empty, copy temp_df to df
                    df_ratio_treat = temp_df.copy()
                else: # if df is not empty, merge temp_df with df
                    df_ratio_treat = pd.merge(df_ratio_treat, temp_df, on='Time', how='outer')
    
        return df_ratio_treat
    
    else:
        dict_ratio      = df.to_dict()
        idxs_list       = list(dict_ratio.keys())
        values_dict     = list(dict_ratio.values())
        
        cells_ratio     = []
        df_ratio_treat  = pd.DataFrame()
    
        for i in range(len(dict_ratio)): # iterate over all cells in all three replicates
            cell        = idxs_list[i]                         # get cell name
            dct         = values_dict[i]                       # get dict of values
            
            if type(dct) is dict: # check that dct is not NaN
                cells_ratio.append(cell)                     # add cell name to master list
                lst     = list(dct.values())                  # get list of ratio values [time, value]
                
                temp_df = pd.DataFrame(lst, columns=["Time", cell])
                        
                if df_ratio_treat.empty: # if df is empty, copy temp_df to df
                    df_ratio_treat = temp_df.copy()
                else: # if df is not empty, merge temp_df with df
                    df_ratio_treat = pd.merge(df_ratio_treat, temp_df, on='Time', how='outer')
    
        return df_ratio_treat


def cull_nans(df):
    """
    Parameters
    ----------
    df : dataframe
        Dataframe containing ratio timecourse data

    Returns
    -------
    df : dataframe
        Original dataframe with NaNs removed
    df_copy : dataframe
        Returns original dataframe with: Cells containing NaNs excised
                                         Values sorted by time
                                         Time column removed
    """
    check_nans = df.isnull().any()                      # check which columns contain NaNs
    
    for key in check_nans.keys():
        if check_nans[key] == True:
            df = df.drop(key, axis=1)                   # delete all cells that still contain NaNs after fillna_custom
            
    # sort values by 'Time'
    df.sort_values('Time', inplace=True)
    
    # delete first column (just time)
    df_copy    = df.copy()
    del df_copy[df_copy.columns[0]]
    
    return df, df_copy


def interpolate_row(row_values, time, time_int):
    """
    Interpolate row-wise.

    Parameters
    ----------
    time : list
        List of timepoints from data collection
    row_values : pd.Series
        Series of values corresponding to the times in `time`
    time_int : list
        List of time values at which to interpolate at

    Returns a series with the interpolated values and sets the index to `time_int`
    """
    # Perform interpolation
    int_values = np.interp(time_int, time, row_values)

    # Convert to series with appropriate index
    return pd.Series(int_values, index=time_int)


def smooth(df_dict_copy, time, window_factor, order, trunc_time = None):
    """
    Parameters
    ----------
    df_dict_copy : dict
        Dictionary of dataframes containing output of cull_nans function
    time : list
        List of timepoints from data collection
    window_factor: int
        Factor by which to scale window length
    order : int
        Polynomial order for Savitzky–Golay smoothing
    trunc_time : int, optional
        Truncation time for compiled data

    Returns
    -------
    df_copy : dict
        Returns Savitzky–Golay smoothed version of original dataframe

    """
    keys                  = list(df_dict_copy.keys())
    df_dict_smoothed_org  = dict.fromkeys(keys)
    df_dict_smoothed_copy = dict.fromkeys(keys)
    
    if trunc_time is None:
        time_int          = list(range(int(time[0]), int(time[-1]) + 1, 1))     
    else:
        time_int          = list(range(int(time[0]), trunc_time + 1, 1))     
    
    
    for key, df in df_dict_copy.items(): # loop over all treatments
        # interpolate
        df       = df.apply(interpolate_row, args=(time, time_int), axis=1)
        
        window   = int(len(df.columns)/window_factor)
        
        # apply Savitzky-Golay filter (works along columns, so requires transposition first)
        df = df.T.apply(lambda x: savgol_filter(x, window_length = window, polyorder = order)).T
        
        # add back in time column to copy
        df_temp  = df.copy()
        df_temp  = df_temp.transpose()
        df_temp.insert(0, "Time", time_int)
        
        df_dict_smoothed_org[key]  = df_temp
        df_dict_smoothed_copy[key] = df
    
    return df_dict_smoothed_org, df_dict_smoothed_copy
    

#%% functions to plot dendrograms

def make_dendrogram(df_dict, Z_dict):
    """
    Parameters
    ----------
    df_dict: dict
        Dictionary containing dataframes with ratio values for PGN, 1X, 10X, and 100X.
    Z_dict : dict
        Dictionary containing linkage matrices for PGN, 1X, 10X, and 100X.

    Returns
    -------
    den_dict: dict
        Dictionary containing the dendrogram data for PGN, 1X, 10X, and 100X hierarchical clustering

    """
    
    keys = list(df_dict.keys())
    den_dict = dict.fromkeys(keys)
    
    for i, key in enumerate(keys):
        df   = df_dict[key].transpose()
        Z    = Z_dict[key]
        
        if not df.empty:
            
            # function to map the original observation indices to the column names
            def label_func(id):
                non_empty_columns = [col for col in df.columns if not df[col].isnull().all()]  # filter out empty columns
                
                if id < len(non_empty_columns):
                    return non_empty_columns[id]
    
            # create dendrogram
            plt.figure(figsize=(25,15))
            den = hac.dendrogram(
                Z,
                orientation     = 'left',
                # leaf_rotation   = 90.,                     # rotates the x axis labels
                leaf_font_size  = 8.,                        # font size for the x axis labels
                leaf_label_func = lambda id: label_func(id), # use the label function to get labels
            )
            plt.title(f'Hierarchical Clustering Dendrogram\n ({key})')
            plt.show()
            
            den_dict[key] = den
        
    return den_dict


def trunc_dendrogram(*args, **kwargs):
    """
    Parameters
    ----------
    *args : Non-keyword arguments
    **kwargs : Keyword arguments
        
    Returns
    ----------
    ddata : den
        Plots truncated dendrogram with cutoff line
    """
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram \n(+ 100µg/mL PGN)')
        plt.xlabel('Sample Index or (Cluster Size)')
        plt.ylabel('Distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

# From example: https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/#Selecting-a-Distance-Cut-Off-aka-Determining-the-Number-of-Clusters


#%% functions to retrieve clusters and subclusters

class Clusters(dict):
    """
    Create Cluster class.
    
    Parameters
    ----------
    dict : dict
        Output from dendrogram function
    
    Returns
    -------
    html : table
        Generates IPython notebook compatible HTML representation of cluster dictionary.
    """
    def _repr_html_(self):
        html = '<table style="border: 0;">'
        for c in self:
            hx = rgb2hex(colorConverter.to_rgb(c))
            html += '<tr style="border: 0;">' \
            '<td style="background-color: {0}; ' \
                       'border: 0;">' \
            '<code style="background-color: {0};">'.format(hx)
            html += c + '</code></td>'
            html += '<td style="border: 0"><code>' 
            html += repr(self[c]) + '</code>'
            html += '</td></tr>'

        html += '</table>'

        return html


def cluster_indices(den):
    """
    Parameters
    ----------
    den : dict
        Output from dendrogram function
        den["color_list"] contains colors
        den["icoord"] contains x coordinates (index coordinates)
        den["dcoord"] contains y coordinates (distances)
    
    Returns
    -------
    cluster_idxs : dict
        dict sorting leaf indices by color
    """
    
    cluster_idxs = defaultdict(list)
    
    for c, pi in zip(den['color_list'], den['icoord']):
        for leg in pi[1:len(den)]:
            i = (leg - 5.0) / 10.0
            
            # check if i is close to an integer index
            # if yes: i = leaf index
            if abs(i - int(i)) < 1e-5:
                if int(i) not in cluster_idxs[c]:
                    cluster_idxs[c].append(int(i))

    return cluster_idxs
    

def get_cluster_classes(den, lab='ivl'):
    """
    Parameters
    ----------
    den : dict
        Output from dendrogram function

    Returns
    -------
    cluster_classes : Cluster
        Fetches clusters and labels from dendrogram.

    """
    cluster_idxs = cluster_indices(den)

    cluster_classes = Clusters()
    for c, l in cluster_idxs.items():
        i_l = [den[lab][i] for i in l]
        cluster_classes[c] = i_l

    return cluster_classes

# From example: https://www.nxn.se/valent/extract-cluster-elements-by-color-in-python


def retrieve_clusters(den_dict):
    """
    Parameters
    ----------
    den_dict : dict
        Dictionary containing the dendrogram data for PGN, 1X, 10X, and 100X hierarchical clustering

    Returns
    -------
    all_clusters : dict
        Dictionary containing the cells associated with each cluster for PGN, 1X, 10X, and 100X

    """
    all_clusters = {}
    for key, den in den_dict.items():
        if den is not None:
            clusters = get_cluster_classes(den)
            all_clusters[key] = clusters
    
    return all_clusters


def interleave_lists(list1, list2):
    """
    Parameters
    ----------
    list1, list2 : list
        Lists of the same length (e.g. [1, 2, 3] and [4, 5, 6])
    
    Returns
    -------
    interleaved : list
        Returns interleaved list of lists (e.g. [[1,3], [2, 5], [3, 6]])

    """
    interleaved = []
    for sublist1, sublist2 in zip(list1, list2):
        interleaved.append([sublist1, sublist2])
    return interleaved


def get_cluster_traces(df_dict, clusters, comp = False, rhobast = False):
    """
    Parameters
    ----------
    df_dict : dict
        Dictionary containing dataframes with ratio values for PGN, 1X, 10X, and 100X.
    clusters : dict
        Dictionary containing the cells associated with each cluster for PGN, 1X, 10X, and 100X
    compile: bool, optional
        Whether or not the data is compiled (True or False)
    
    Returns
    -------
    all_traces_dict : dict
        Dictionary containing the ratio time trace values for each cell in each cluster

    """
    keys                    = clusters.keys()
    all_traces_dict         = dict.fromkeys(keys)
    
    for key in keys: # iterate over treatments
        clus_names          = clusters[key].keys()
        cluster_traces_dict = dict.fromkeys(clus_names)
        
        for cluster in clusters[key]: # iterate over clusters
            df_cluster = pd.DataFrame()
            
            for cell in clusters[key][cluster]: # iterate over cells in cluster    
                if comp is True:    
                    # extract date    
                    date_str = re.search(r'\b(\d{8})\b', cell).group(1) if re.search(r'\b(\d{8})\b', cell) else None
                    date     = '-'.join([date_str[:4], date_str[4:6], date_str[6:]])
                
                df       = df_dict[date][key] if comp else df_dict[key]      # specify which df to use based on treatment
                data     = [list(df["Time"]), list(df[cell])]
                data     = interleave_lists(data[0], data[1])
                
                temp_df  = pd.DataFrame(data, columns=["Time", cell])
                        
                if df_cluster.empty: # if df is empty, copy temp_df to df
                    df_cluster = temp_df.copy()
                
                else: # if df is not empty, merge temp_df with df
                    df_cluster = pd.merge(df_cluster, temp_df, on='Time', how='outer')
                    
                cluster_traces_dict[cluster] = df_cluster
                
            all_traces_dict[key] = cluster_traces_dict
        
    return all_traces_dict

         
#%% functions to pull subclusters

def subclusters(df_dict_copy, Z_dict, method, factor):
    """
    Flattens clustering data to retrieve subclusters.
    
    Parameters
    ----------
    df_dict_copy : dict
        Dictionary of dfs containing ratio timecourse values for PGN, 1X, 10X, and 100X
    Z_dict : dict
        Dictionary of arrays containing linkage matrices for PGN, 1X, 10X, and 100X
    method : str
        Criterion to use in forming flat clusters; e.g. "inconsistent", "distance", "maxclust", "monocrit", "maxclust_monocrit"
    factor : int
        Factor by which to scale max cophenetic distance to set threshold value

    Returns
    -------
    subclusters_dict : dict
        Dictionary of dfs containing the extracted subcluster labels associated with the corresponding cell name

    """
    
    keys             = list(df_dict_copy.keys())                    # list of treatments
    subclusters_dict = dict.fromkeys(keys)              # initialize empty dict to store results
    thresholds       = []
        
    for key in keys: # iterate over all treatments
        df           = df_dict_copy[key]
        Z            = Z_dict[key]                      # extract Z matrix for treatment
        max_cophe    = max(hac.cophenet(Z))             # calculate maximum cophenetic distance
        threshold    = max_cophe / factor               # set threshold value
        
        # form flat clusters from hierarchical clustering
        fclus        = fcluster(Z, threshold, criterion = method)
        
        # associate flat clusters with cell numbers
        idx_clus_map = pd.DataFrame({'Cell': df.index, 'Subcluster': fclus})
        
        subclusters_dict[key] = idx_clus_map
        thresholds.append(threshold)
            
    return subclusters_dict, thresholds


def make_dendrogram_subclusters(df_dict_copy, Z_dict, subclusters_dict, thresholds, ax = None):
    """
    Parameters
    ----------
    df_dict_copy : dict
        Dictionary containing dataframes with ratio values for PGN, 1X, 10X, and 100X.
    Z_dict : dict
        Dictionary containing linkage matrices for PGN, 1X, 10X, and 100X.
    subclusters_dict: dict
        Output of `subclusters` function; dictionary of dfs containing the extracted cluster labels associated with the corresponding cell name
    threshold: list
        List of threshold value returned by `subclusters`
    ax : axes
        Axis on which to plot the dendrogram, optional

    Returns
    -------
    den_dict: dict
        Dictionary containing the dendrogram data for PGN, 1X, 10X, and 100X hierarchical clustering
    """
    
    keys = list(df_dict_copy.keys())
    den_dict = dict.fromkeys(keys)
    
    for i, key in enumerate(keys):
        df          = df_dict_copy[key].transpose()
        Z           = Z_dict[key]
        subclusters = subclusters_dict[key].iloc[:, 1].tolist()
        numclusters = len(set(subclusters))
        threshold   = thresholds[i]
        
        if not df.empty:
                        
            # function to map the original observation indices to the column names
            def label_func(id):
                non_empty_columns = [col for col in df.columns if not df[col].isnull().all()]  # filter out empty columns
                
                if id < len(non_empty_columns):
                    return non_empty_columns[id]
                
            # create dendrogram
            if ax is not None and isinstance(ax, np.ndarray): # if the ax parameter is an array
                den     = hac.dendrogram(
                    Z,
                    color_threshold = threshold,                                # set color threshold based on subclusters
                    orientation     = 'left',                                   # set orientation
                    ax              = ax[i],                                    # specify the axes
                    # leaf_rotation   = 90.,                                    # rotates the x axis labels
                    leaf_font_size  = 8.,                                       # font size for the x axis labels
                    leaf_label_func = lambda id: label_func(id),                # use the label function to get labels
                )
                ax[i].axvline(x=threshold, c='grey', linestyle='dashed')        # plot vertical line at color threshold
                ax[i].set_title(f'Hierarchical Clustering Dendrogram\n ({key}, {numclusters} subclusters)')
                
            elif ax is not None and isinstance(ax, list): # if the ax parameter is a list
                den     = hac.dendrogram(
                    Z,
                    color_threshold = threshold,                                # set color threshold based on subclusters
                    orientation     = 'left',                                   # set orientation
                    ax              = ax,                                       # specify the axes
                    # leaf_rotation   = 90.,                                    # rotates the x axis labels
                    leaf_font_size  = 8.,                                       # font size for the x axis labels
                    leaf_label_func = lambda id: label_func(id),                # use the label function to get labels
                )
                ax.axvline(x=threshold, c='grey', linestyle='dashed')           # plot vertical line at color threshold
                ax.set_title(f'Hierarchical Clustering Dendrogram\n ({key}, {numclusters} subclusters)')
            
            else: # if the ax parameter is not provided
                plt.figure(figsize = (25,15))
                den     = hac.dendrogram(
                    Z,
                    color_threshold = threshold,                                # set color threshold based on subclusters
                    orientation     = 'left',                                   # set orientation
                    # leaf_rotation   = 90.,                                    # rotates the x axis labels
                    leaf_font_size  = 8.,                                       # font size for the x axis labels
                    leaf_label_func = lambda id: label_func(id),                # use the label function to get labels
                )
                plt.axvline(x=threshold, c='grey', linestyle='dashed')          # plot vertical line at color threshold
                plt.title(f'Hierarchical Clustering Dendrogram\n ({key}, {numclusters} subclusters)')
                
                plt.show()
            
            den_dict[key] = den
        
    return den_dict


def rename_subclusters(clusters_dict, subclusters_dict):
    """
    Parameters
    ----------
    clusters_dict : dict
        Dictionary of dfs containing the extracted cluster labels associated with the corresponding cell name
    subclusters_dict : dict
        Dictionary of dfs containing the extracted subcluster labels associated with the corresponding cell name

    Returns
    -------
    subclusters_dict_copy : dict
        Dictionary of dfs containing the updated subcluster labels associated with the corresponding cell name
        Naming format: "SC{cluster}-{subcluster}"
    """
    
    subclusters_dict_copy = copy.deepcopy(subclusters_dict)
    treatments = list(clusters_dict.keys())
    
    for treat in treatments: # iterate through treatments
    
        for subcluster, subcluster_cells in subclusters_dict[treat].items(): # iterate through subclusters
        
            for cluster, cluster_cells in clusters_dict[treat].items(): # iterate through clusters
                # check which cluster the given subcluster is contained in
                any_contained = any(item in cluster_cells for item in subcluster_cells)
                
                if any_contained is True:        
                    # update the subcluster name
                    new_subcluster = "S" + str(cluster) + "-" + str(subcluster).strip("C")
                    
                    # update the subcluster key
                    if subcluster in subclusters_dict_copy[treat]:
                        subclusters_dict_copy[treat][new_subcluster] = subclusters_dict_copy[treat].pop(subcluster)
                    
    return subclusters_dict_copy


def nest_subcluster_traces(subcluster_traces):
    """
    Parameters
    ----------
    subcluster_traces : dict
        Dictionary of ratio time trace values for cells in each subcluster.

    Returns
    -------
    nested_subcluster_traces : dict
        Updated dictionary of ratio time trace values with subclusters nested within clusters

    """
    
    nested_subcluster_traces = {}

    for treat, subclusters in subcluster_traces.items():
        nested_subcluster_traces[treat] = {}
        
        for subcluster, data in subclusters.items():
            # extract cluster name from subcluster name (assuming format from rename_subcluster)
            cluster_name = subcluster.strip("S").split("-")[0]
            
            if cluster_name not in nested_subcluster_traces[treat]:
                nested_subcluster_traces[treat][cluster_name] = {}
            
            nested_subcluster_traces[treat][cluster_name][subcluster] = data

    return nested_subcluster_traces
    

def flatten_trace_df(subcluster_traces_smooth):
    """
    Flattens the subcluster_traces_smooth dictionary (sorted by treatment --> cluster --> subcluster) into one df.

    Parameters
    ----------
    subcluster_traces_smooth : dict, optional
        Dictionary of interpolated/smoothed ratio time trace values for cells in each subcluster within clusters for all treatments.
        
    Returns
    -------
    all_data_df : df
        Flattened version of subcluster_traces_smooth

    """
    
    all_data_df = pd.DataFrame()
    
    for treatment, treatment_data in subcluster_traces_smooth.items():
        for cluster, cluster_data in treatment_data.items():
            for subcluster, subcluster_data in cluster_data.items():
                # drop "Time" column and transpose
                subcluster_data = subcluster_data.drop("Time", axis=1).T
                
                # add treatment prefix to indices
                subcluster_data.index = [f"{treatment} " + idx for idx in subcluster_data.index]
                
                # concatenate with all_data_df
                if all_data_df.empty:
                    all_data_df = subcluster_data
                else:
                    all_data_df = pd.concat([all_data_df, subcluster_data])
                
    return all_data_df


#%% functions to retrieve trace parameters

def trace_descriptors(df_dict_copy_smooth_org, subclusters_dict, prom_value = 0.1):
    """
    Parameters
    ----------
    df_dict_copy_smooth_org : dict
        Dictionary of dfs containing output of `smooth` function (original version)
    subclusters_dict : dict
        Dictionary of dfs containing the extracted subcluster labels associated with the corresponding cell name 
    prom_value : int, optional
        Required prominence of peaks for find_peaks (default 0.1)

    Returns
    -------
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
                                        

    """
    keys        = list(df_dict_copy_smooth_org.keys())                          # list of treatments
    dict_des    = dict.fromkeys(keys)                                           # initialize empty dict
    descriptors = ["Cluster", "Max Value", "Max Time", "Half Max", "Half Time", "Area", "Max In", "Max Out", "Peaks", "Local Max"]
    
    for key in keys: # iterate over treatments
        cells   = list(df_dict_copy_smooth_org[key].columns)[1:]
        times   = list(df_dict_copy_smooth_org[key]["Time"].values)
        df      = df_dict_copy_smooth_org[key].set_index("Time")
        subclusters = subclusters_dict[key]
        
        df_des  = pd.DataFrame(index = cells, columns = descriptors)
        
        for cell in cells: # iterate over cells
            cluster_name = next((key for key, value in subclusters.items() if cell in value), None)
            cell_vals  = df[cell].values
        
            start_val  = mean(cell_vals[0:30])
            max_value  = df[cell][30:].max()                                    # calculate max value
            max_idx    = df.iloc[30:][df[cell].iloc[30:] == max_value].index.tolist()[0]
            max_time   = times[max_idx]                                         # calculate max time
            
            # calculate half max value
            half_value = (0.5 * (max_value - start_val)) + start_val
            
            # calculate half max time
            premax_mask = (np.array(times) >= 30) & (np.array(times) <= max_time)
            differences = np.abs(cell_vals[premax_mask] - half_value)
            
            if differences.size > 0:
                half_idx = np.argmax(differences == np.min(differences))
                half_idx = np.where(premax_mask)[0][half_idx]
                half_time = times[half_idx]
        
            # integrate
            area       = trapezoid(cell_vals, times)                            # calculate area
            
            # calculate max rate in
            max_in_rate  = 0.0
            max_in_time  = None
            for i in range(30, max_idx): # iterate over all time points before max_time
                # define time range for calculating slope (three points)
                start_idx         = i - 1
                end_idx           = i + 2
        
                # extract time points and values within that range
                selected_times    = times[start_idx:end_idx]
                selected_vals     = cell_vals[start_idx:end_idx]
        
                # fitting via linear regression
                slope, _, _, _, _ = linregress(selected_times, selected_vals)
                rate_in           = abs(slope) if slope > 0 else None
        
                # update the maximum rate of entry and corresponding time point if necessary
                if rate_in is not None and rate_in > max_in_rate:
                    max_in_rate   = rate_in
                    max_in_time   = times[i]
            max_in      = [max_in_time, max_in_rate]
            
            # calculate max rate out
            max_out_rate = 0.0
            max_out_time = None
            for i in range(max_idx + 1, len(times)): # iterate over all time points after max_time
                # define time range for calculating slope (three points)
                start_idx         = i - 1
                end_idx           = i + 2
        
                # extract time points and values within that range
                selected_times    = times[start_idx:end_idx]
                selected_vals     = cell_vals[start_idx:end_idx]
        
                # fitting via linear regression
                slope, _, _, _, _ = linregress(selected_times, selected_vals)
                rate_out          = abs(slope) if slope < 0 else None
        
                # update the maximum rate of entry and corresponding time point if necessary
                if rate_out is not None and rate_out > max_out_rate:
                    max_out_rate = rate_out
                    max_out_time = times[i]
            max_out    = [max_out_time, max_out_rate]
            
            # find peaks
            peaks, _   = find_peaks(cell_vals, prominence = prom_value)
            num_peaks  = len(peaks)
            
            # local maxima
            local_maxs = []
            for peak_idx in peaks:
                peak_time = times[peak_idx]
                peak_val  = cell_vals[peak_idx]
                local_max = [peak_time, peak_val]
                local_maxs.append(local_max)
            
            # update df
            df_des.loc[cell, "Cluster"]   = cluster_name
            df_des.loc[cell, "Max Value"] = max_value
            df_des.loc[cell, "Max Time"]  = max_time
            df_des.loc[cell, "Half Max"]  = half_value
            df_des.loc[cell, "Half Time"] = half_time
            df_des.loc[cell, "Area"]      = area
            df_des.loc[cell, "Max In"]    = max_in
            df_des.loc[cell, "Max Out"]   = max_out
            df_des.loc[cell, "Peaks"]     = num_peaks
            df_des.loc[cell, "Local Max"] = local_maxs
        
        # update dictionary
        dict_des[key] = df_des
    
    return dict_des
 
    
#%% functions to save/import data

def save_data(file_path, object_data, object_name):
    """
    Saves object locally to provided file path.

    Parameters
    ----------
    file_path : str
        Desired destination folder
    object_data : N/A
        Object you wish to save
    object_name : str
        Desired file name

    Returns
    -------
    None.

    """
    full_file_path = file_path + "\\" + object_name + ".pkl"
    # print(full_file_path)
    
    with open(full_file_path, "wb") as f:
        pickle.dump(object_data, f)


def import_data(file_path, object_name, file_type = ".pkl"):
    """
    Loads object from the provided file path.

    Parameters
    ----------
    file_path : str
        Desired source folder
    object_name : str
        Desired object name
    file_type : str, optional
        File type of desired object.  The default is ".pkl"

    Returns
    -------
    object_data : N/A
        Object you wish to import
    """
    
    full_file_path = file_path + "\\" + object_name + file_type
    # print(full_file_path)
    with open(full_file_path, "rb") as f:
        object_data = pickle.load(f)
        
    return object_data


#%% # === MAIN LOOP ===

# file paths
# Emma's computer (for testing)
curr_dataset    = "Intensities"
all_data        = "C:\\Users\\emmar\\OneDrive\\Documents\\Boston University\\PhD\\Wunderlich Lab\\Python Data and Code\\Dataframes\\"
rootdir         = all_data + curr_dataset
file_name       = "dict_intensities_alldatasets_goodcells_areas_101024"

# # generalized (for MRE)
# all_data        = "/path/to/your/data/2025-01-01_datasetName/"
# curr_dataset    = "Python/<maskSettings>/IntensitiesDF"
# rootdir         = all_data + curr_dataset
# file_name       = "dictIntensitiesNomask_{datetimeStr}"
# file_path_step9 = all_data + "Processed Traces"

# smoothing parameters
sg_factor       = 5
sg_order        = 2
dist_factor     = 5

# # run clustering
# df_dict_org, df_dict_smooth_org, df_dict_smooth_copy, Z_dict = cluster_compiled(rootdir, file_name)

# # retrieve subclusters
# subclustersflat_smooth, thresholds                           = subclusters(df_dict_smooth_copy, Z_dict, "distance", dist_factor)

# # make dendrograms
# den_dict_smooth                                              = make_dendrogram(df_dict_smooth_copy, Z_dict)
# den_dict_smooth_subclusters                                  = make_dendrogram_subclusters(df_dict_smooth_copy, Z_dict, subclustersflat_smooth, thresholds)

# # retrieve cluster identities
# clusters_smooth                                              = retrieve_clusters(den_dict_smooth)
# subclusters_smooth                                           = retrieve_clusters(den_dict_smooth_subclusters)
# subclusters_smooth                                           = rename_subclusters(clusters_smooth, subclusters_smooth)

# # retrive cluster traces
# cluster_traces                                               = get_cluster_traces(df_dict_org, clusters_smooth, comp = True)
# cluster_traces_smooth                                        = get_cluster_traces(df_dict_smooth_org, clusters_smooth)
# subcluster_traces                                            = get_cluster_traces(df_dict_org, subclusters_smooth, comp = True)
# subcluster_traces_smooth                                     = get_cluster_traces(df_dict_smooth_org, subclusters_smooth)
# subcluster_traces                                            = nest_subcluster_traces(subcluster_traces)
# subcluster_traces_smooth                                     = nest_subcluster_traces(subcluster_traces_smooth)

# # flatten subcluster traces
# all_traces_df                                                = flatten_trace_df(subcluster_traces_smooth)
# times_smooth                                                 = list(all_traces_df.columns)

# # retrieve trace properties
# dict_trace_descriptors                                       = trace_descriptors(df_dict_smooth_org, subclusters_smooth)
# for treatment, treatment_df in dict_trace_descriptors.items():
#     treatment_df.insert(1, "Behavior", "")
    
# # save data
# save_data(file_path_step9, subcluster_traces_smooth, "subcluster_traces_smooth")
# save_data(file_path_step9, dict_trace_descriptors, "dict_trace_descriptors")
# save_data(file_path_step9, all_traces_df, "all_traces_df")
# save_data(file_path_step9, times_smooth, "times_smooth")