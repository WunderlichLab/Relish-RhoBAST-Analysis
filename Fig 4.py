# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 11:16:06 2025

@author: noshin
"""
import pandas as pd
import seaborn as sns
import tifffile as tf
from tifffile import imread
from tifffile import imshow
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import savgol_filter
import pickle
import time
from datetime import timedelta
datetime_str = time.strftime("%m%d%y_%H:%M")

#!!! Update path file!!!
gitdir = 'G:/path/' 
#!!! Update path file!!!

files_import = gitdir+'Figure 4 Files/'
fig_output = gitdir+'Temp Output/Fig 4/'

plt.rcParams['font.family'] = 'Arial'
labelfsize = 12
fsize = 10
tickfsize = 9
mag = 1.2
shrink = 0.78

sg_factor_rel = 5   #5 for relish data
sg_factor_rb = 8   #5 for relish data
sg_order_rel  = 2   #Polynomial order for Savitzky–Golay smoothing

resolution=  3.4756 #pixels per micron
units_per_pix = 1/resolution


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
                
                # if all_data_df.empty:
                #     all_data_df = subcluster_data.drop("Time", axis = 1).T
                # else:
                #     all_data_df = pd.concat([all_data_df, subcluster_data.drop("Time", axis = 1).T])
    
    return all_data_df

def import_data(file_path, object_name, file_type = ".pkl"):   
    full_file_path = file_path + "/" + object_name + file_type
    # print(full_file_path)
    with open(full_file_path, "rb") as f:
        object_data = pickle.load(f)
        
    return object_data

#%% Import Data
peaks_dict_flat                    = import_data(files_import+'SVM results/', "goodcomp7_rb_AMP_peaks_dict_flat")
subcluster_traces_smooth           = import_data(files_import+'SVM results/', "goodcomp7_rel_AMP_subcluster_traces_div_smooth")
SVM_results_dict                   = import_data(files_import+'SVM results/', "goodcomp7_rel_AMP_SVM_results_dict_noIc")
cell_categories_df                 = import_data(files_import+'SVM results/', "goodcomp7_trace_sorting_cell_categories_v4")

stim_time   = 60
interval_forplt = np.concatenate([np.arange(0, 121, 15) , np.arange(150, 631, 30), np.arange(690, 971, 60)])

treatments_to_plot = ["100X", "10X",'-PGN']
peaks_dict_flat = {k:v for k,v in peaks_dict_flat.items()}

#%% Fig 4A Plot classification of traces
colors          = ["#DC143C", "#FF6F61", "indigo", "dodgerblue", "grey"]
color_palette   = sns.color_palette(colors)
behavior_colors = {"I": color_palette[0], "Id": color_palette[1], "G": color_palette[2], "D": color_palette[3], "N": color_palette[4]}
behavior_keys   = {"I": "Immediate", "Id": "Immediate with decay", "G": "Gradual", "D": "Delayed", "N": "Nonresponsive"}
behavior_rev    = {v: k for k, v in behavior_keys.items()}
line_styles     = {"Dpt": "solid", "IM2": "dashed", "Mtk": "dotted"}

AMPs = list(peaks_dict_flat.keys())

treatments = []
for AMP, AMP_df in peaks_dict_flat.items():
    AMP_treatments = AMP_df.index.get_level_values(0).map(str).str.split(' ').str[0].unique().tolist()
    # print(treatments)
    if len(treatments) == 0:
        treatments = AMP_treatments
    else:
        if treatments == AMP_treatments:
            continue
        else:
            raise ValueError("Treatments must be consistent across AMPs.")
            
# column and row names
col_names              = [behavior for behavior in behavior_rev]
row_names              = ["FC $R_{nuc:tot}$"] + [treatment for treatment in treatments_to_plot]

# ROW 1: Relish traces
# flatten trace data
all_traces_df   = flatten_trace_df(subcluster_traces_smooth)
times           = list(all_traces_df.columns)

# initialize figure
figsize1 = (6.3, 1 * (len(treatments_to_plot) + 1))

fig, axs    = plt.subplots(len(treatments_to_plot) + 1, 5, figsize = figsize1, sharey = "row")
    
# extract dataframe of all cells
results_df  = SVM_results_dict["All cells"]

# convert cell_categories_df to dict and reorder
cell_categories_dict = cell_categories_df.set_index("Category")["Cells"].to_dict()

for n, (SVM_behavior, behavior_name) in enumerate(behavior_keys.items()):        
    # set y-axis limits and ticks
    axs[0, n].set_ylim(0.6, 2.0)
    axs[0, n].set_yticks([0.8, 1.2, 1.6, 2.0])
    
    # plot all traces for each behavior
    behavior_traces = pd.DataFrame()
    behavior_color  = behavior_colors[SVM_behavior]
    num_cells       = 0
    
    mask        = (results_df["Predicted"] == SVM_behavior)
    indices     = results_df.index[mask].tolist()
    
    num_cells   = len(indices)
    traces      = all_traces_df.loc[indices]
    color       = behavior_colors[SVM_behavior]
    axs[0, n].plot(times, traces.T, linewidth = 0.5, color = color, alpha = 0.4)
    axs[0, n].plot(times, traces.mean(axis=0), color = "black", linewidth = 1)
    
    #overlay max and time of max on average behavior
    temp = (traces.mean(axis=0)).tolist()[:200]
    max_value = max(temp)
    max_index = temp.index(max_value)
    max_time = times[max_index]
    axs[0, n].plot(max_time, max_value, marker = '.', color = "black", )
    axs[0, n].annotate(
        f"({max_time:.1f}, {max_value:.1f})",           # Text to display
        (max_time, max_value),                         # Point to annotate
        textcoords="offset points",                    # Position text with an offset
        xytext=(2, 8),                               # Offset text by 10 points above
        ha='left',                                  # Center text horizontally
        fontsize=8,                                  # (Optional) Set font size
        color='black'                                 # (Optional) Set text color
    )

    
    # set axes labels and title
    title_color = behavior_color
    axs[0, n].tick_params(axis = "both", labelsize = tickfsize)
    axs[0, n].text(y = 1.85, x = 760, s = f"n = {num_cells}", ha = "center", va = "center", fontsize = tickfsize)
    #axs[0, 0].set_ylabel("Fold-Change\n$R_{nuc:tot}$", fontsize = fsize)
    
    # highlight pre- and post-stim times
    y_min, y_max = axs[0, n].get_ylim()
    prestim  = patches.Rectangle((-60, 0.6), 60, .05, edgecolor="black", facecolor="none", hatch='///')
    hatch = patches.Rectangle((0, 0.6), times[-1] - 30, 0.05, color = "black")
    axs[0, n].add_patch(prestim)
    axs[0, n].add_patch(hatch)


# ROWS 2-5: RhoBAST peaks (by treatment)
percent_peaks = dict.fromkeys(AMPs)
standerr_behavior_tmt = {}
for AMP, AMP_df in peaks_dict_flat.items():
        
    percent_peaks_behavior = dict.fromkeys(behavior_keys)
    
    for n, (SVM_behavior, behavior_name) in enumerate(behavior_keys.items()):        
        percent_peaks_behavior_tmt = dict.fromkeys(["Combined"] + treatments)
        
        mask        = (results_df["Predicted"] == SVM_behavior)
        indices     = results_df.index[mask].tolist()
        # print(f"\n{SVM_behavior}: {len(indices)}")
        indices     = [index for index in indices if AMP in index]
        
        num_cells      = len(indices)
        # print(f"\n{AMP} ({SVM_behavior}): {num_cells}")
        behavior_peaks = AMP_df.loc[indices]
    
        peaks_binary_all = (behavior_peaks != 0).astype(int)  # convert non-zero values to 1 and zero values to 0
        peaks_ratio_all  = (peaks_binary_all.sum() / num_cells) * 100
        # print(peaks_ratio_all)
        percent_peaks_behavior_tmt["Combined"] = peaks_ratio_all
        # axs[1, n].plot(peaks_ratio_all, linewidth = 1.0, linestyle = line_styles[AMP], color = "black")
        # # axs[1, n].text(y = 1.9, x = 800, s = f"n = {num_cells}", ha = "center", va = "center", fontsize = 16)
        # axs[1, 0].set_ylabel("% cells with RhoBAST peaks", fontsize = 14)
        
        
        for i, treatment in enumerate(treatments_to_plot):
            tmt_indices = [index for index in indices if treatment in index]
            tmt_behavior_peaks = AMP_df.loc[tmt_indices]
            num_tmt_cells      = len(tmt_indices)
            # print(f"{treatment}: {num_tmt_cells}")
            
            peaks_binary_tmt = (tmt_behavior_peaks != 0).astype(int)
            # display(peaks_binary_tmt)
            # if AMP == "Dpt" and treatment == "10X":
            #     print(f"{peaks_binary_tmt.sum()}")
            peaks_ratio_tmt  = (peaks_binary_tmt.sum() / num_tmt_cells) * 100
            # if AMP == "Dpt" and treatment == "10X":
            #     print(f"{peaks_ratio_tmt}")
            percent_peaks_behavior_tmt[treatment] = peaks_ratio_tmt

            # Calculate the standard error
            p = peaks_ratio_tmt / 100  
            standerr_tmt = np.sqrt(p * (1 - p) / num_tmt_cells) * 100            
            standerr_behavior_tmt[treatment] = standerr_tmt
      
            # Plotting with error ribbon
            axs[i + 1, n].plot(peaks_ratio_tmt, linewidth = 1.5, linestyle = line_styles[AMP], color = "black", label = AMP)
            # Create an error ribbon
            peaks_ratio_tmt_y = peaks_ratio_tmt.values
            x = peaks_ratio_tmt.index
            mins = [peaks_ratio_tmt_y - standerr_tmt.values][0]
            maxs= [peaks_ratio_tmt_y + standerr_tmt.values][0]
            axs[i + 1, n].fill_between(x,mins,maxs , alpha=0.2, color="black")        
                     
            #axs[i + 2, n].text(y = 1.9, x = 800, s = f"n = {num_tmt_cells}", ha = "center", va = "center", fontsize = 16)
            #axs[i + 1, 0].set_ylabel("% cells +Foci", fontsize = tickfsize)
            axs[i + 1, n].set_yticks([0, 20, 40])
            axs[i + 1, n].tick_params(axis='y', labelsize=tickfsize)
            
            axs[i, n].set_xticks(ticks = [-60, 0, 200,400,600,800,1000], labels=[])
            axs[i + 1, n].set_xticks(ticks = [-60, 0, 200,400,600,800, 1000], labels=['-60','','','400','', '800', ''])

            if AMP != 'IM2' and treatment != '-PGN':
            #overlay max and time of max on average behavior
                temp = peaks_ratio_tmt.tolist()
                max_value = max(temp)
                max_index = temp.index(max_value)
                max_time = interval_forplt[max_index]
                axs[i + 1, n].plot(max_time, max_value, marker = '.', markersize = 10, 
                                  color= ((0, 0, 225/255) if AMP=='Dpt' else (50/255, 205/255, 50/255)) )
                axs[i + 1, n].annotate(
                    f"({max_time:.1f}, {max_value:.1f})",           # Text to display
                    (max_time, max_value),                         # Point to annotate
                    textcoords="offset points",                    # Position text with an offset
                    xytext=(15, 8),                               # Offset text by 10 points above
                    ha='left',                                  # Center text horizontally
                    fontsize=6,                                  # (Optional) Set font size
                    color=((0, 0, 225/255) if AMP=='Dpt' else (50/255, 205/255, 50/255))
                )
        percent_peaks_behavior[SVM_behavior] = percent_peaks_behavior_tmt
    
    percent_peaks[AMP] = percent_peaks_behavior

pad=5
for ax, col in zip(axs[0], col_names):
    if col == 'Immediate with decay':
        ax.annotate('Imm + Decay', xy = (0.5, 1), xytext = (0, pad),
                    xycoords = 'axes fraction', textcoords = 'offset points',
                    fontsize = fsize, ha = 'center', va = 'baseline',
                    color = behavior_colors[behavior_rev[col]])
    elif col == 'Immediate':
        ax.annotate('Imm', xy = (0.5, 1), xytext = (0, pad),
            xycoords = 'axes fraction', textcoords = 'offset points',
            fontsize = fsize, ha = 'center', va = 'baseline',
            color = behavior_colors[behavior_rev[col]])

    else:    
        ax.annotate(col, xy = (0.5, 1), xytext = (0, pad),
                    xycoords = 'axes fraction', textcoords = 'offset points',
                    fontsize = fsize, ha = 'center', va = 'baseline',
                    color = behavior_colors[behavior_rev[col]])

# print(row_names)
for ax, row in zip(axs[:,0], row_names):
    if row == '10X':
        row = "% cells $\mathrm{Foci}_{+}$\n 10X\n\n"
        ax.annotate(row, xy=(0, 0.5), xytext=(1, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    fontsize=fsize, ha='center', va='center', rotation=90)

    else:
        #ax.annotate(row, xy = (0, 0.5), xytext = (-ax.yaxis.labelpad - pad, 0),
        ax.annotate(row, xy = (0, 0.5), xytext = (3.7, 0),
                    xycoords = ax.yaxis.label, textcoords = 'offset points',
                    fontsize = fsize, ha = 'right', va = 'center', rotation = 90)

handles, labels = ax.get_legend_handles_labels()
labels = ['Dpt', 'BomS2', 'Mtk']
# Original labels: ['Dpt', 'BomS2', 'Mtk',  New order: ['Dpt', 'Mtk', 'BomS2']
new_order = [0, 2, 1]  # Indices of the desired new order
handles = [handles[i] for i in new_order]
labels = [labels[i] for i in new_order]

#plt.figlegend(handles, labels, loc = 'lower center', bbox_to_anchor=(0.5, -0.15), ncol = 5, labelspacing = 0., fontsize = tickfsize)
axs[3,4].legend(handles, labels, loc='upper right', bbox_to_anchor=(1.02, .99), 
                ncol=1, labelspacing=0.1, handlelength=1.2, fontsize=tickfsize)

fig.supxlabel("Time (min)", fontsize=fsize, y=0.01)

plt.tight_layout(pad = .2)     
#plt.show()

#---------------------------------------- Save
figname = 'RhoBAST Classified Traces'
savename = fig_output+'Fig 4A_'+figname+'.png'
fig.savefig(savename, bbox_inches = 'tight', dpi=1000)
