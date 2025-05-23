import pandas as pd
import seaborn as sns
import tifffile as tf
from tifffile import imread
from tifffile import imshow
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.cm import get_cmap
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import os
from scipy import ndimage
from scipy.signal import savgol_filter
from skimage import measure
import skimage.io
from skimage.measure import label, regionprops
from scipy.ndimage import label
from scipy.ndimage import binary_dilation
from scipy.signal import savgol_filter
from scipy.integrate import trapezoid as trapz
from skimage.segmentation import mark_boundaries
from skimage.feature.peak import peak_local_max
from PIL import Image
import pickle
import  microfilm.microplot
from microfilm import microplot
from microfilm.microplot import microshow, Micropanel
import gc
import copy
import datetime
import time
from datetime import timedelta
datetime_str = time.strftime("%m%d%y_%H:%M")

mac = '/Volumes/rkc_wunderlichLab/'
PC = 'R:/'
anacomp = 'Z:/'

computer = anacomp


plt.rcParams['font.family'] = 'Arial'

fsize = 10
tickfsize = 9
mag = 1.2
shrink = 0.78

if computer == mac:
    gdrive = '/Users/noshin/Library/CloudStorage/GoogleDrive-noshin@bu.edu/Shared drives/Wunderlich Lab/People/Noshin/'
else:
    gdrive = 'G:/Shared drives/Wunderlich Lab/People/Noshin/'

fig_output = gdrive+'Paper/Figures/'

#%% CRISPR Editing
# Data
days = [2, 3, 4, 5, 6, 7]
viability = [[93, 94, np.nan, 77, np.nan, 73],
             [90, 94, np.nan, 66, np.nan, 77],
             [91, 90, np.nan, 87, np.nan, 79],
             [68, 86, np.nan, 79, np.nan, 15],
             [70, 86, np.nan, 85, np.nan, 15],
             [82, 88, np.nan, 83, np.nan, 78],
             [76, 82, np.nan, 90, np.nan, 33]]

live_cells = [[1.29E+07,	1.57E+07,float('nan'),		1.70E+07,float('nan'),		1.85E+07],
              [1.28E+07,	1.71E+07,float('nan'),		1.29E+07,float('nan'),		2.06E+07],
              [1.20E+07,	1.59E+07,float('nan'),		1.99E+07,float('nan'),		1.95E+07],
              [5.07E+06,	1.24E+07,float('nan'),		1.44E+07,float('nan'),		3.24E+06],
              [5.60E+06,	1.16E+07,float('nan'),		2.42E+07,float('nan'),		4.91E+06],
              [1.08E+07,	1.87E+07,float('nan'),		2.30E+07,float('nan'),		2.13E+07],
              [6.39E+06,	1.10E+07,float('nan'),		2.30E+07,float('nan'),8.98E+06]]

# Find global min and max for viability and live_cells
v_min = min([min(v) for v in viability if not all(np.isnan(v))])
v_max = max([max(v) for v in viability if not all(np.isnan(v))])
lc_min = min([min(lc) for lc in live_cells if not all(np.isnan(lc))])
lc_max = max([max(lc) for lc in live_cells if not all(np.isnan(lc))])
    

for i in range(len(viability)):
    days=np.array(days)
    
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Days')
    ax1.set_ylabel('Viability', color=color)
    

    
    y1 = np.array(viability[i]).astype(np.double)
    y1_mask= np.isfinite(y1)
    ax1.plot(days[y1_mask], y1[y1_mask], color=color, marker='o')
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  
    color = 'tab:blue'
  
    ax2.set_ylabel('Live Cells', color=color)  
    y2 = np.array(live_cells[i]).astype(np.double)
    y2_mask= np.isfinite(y2)
    ax2.plot(days[y2_mask], y2[y2_mask], color=color, marker='o')
    ax2.tick_params(axis='y', labelcolor=color)


    fig.tight_layout()  
    plt.title(f'Condition {i + 1}: Viability and Live Cells over Days Post EP')
    plt.grid(True)
 
    # Set the same y-axis limits for all plots
    ax1.set_ylim([v_min-10,v_max+10])
    ax2.set_ylim([lc_min-10**6 ,lc_max +10**6])
    
    
    if i >= 3 and i < 7:
        plt.axvline(x=3, color='k', linestyle='--') # Add a vertical dashed line after day 3
        plt.text(3, (0), '+Puro',ha='center') # Add text "+Puro" below the x-axis
    

#%%plot only viability for certain conditions:
    
# Define the conditions to plot
conditions_to_plot = [4, 5] 

# Define colors for each condition
colors = ['tab:red', 'tab:blue']
# Define labels for each condition
labels = ["Repair DNA", "Repair DNA + RNP"]

fig, ax1 = plt.subplots(figsize=(3.3, 2.5), dpi=1000)  # You can adjust the values as needed

ax1.set_xlabel('Days Post EP',fontsize=tickfsize)
ax1.set_ylabel('Viability (%)',fontsize=tickfsize)

for i, color, label in zip(conditions_to_plot, colors, labels):
    y1 = np.array(viability[i]).astype(np.double)
    y1_mask= np.isfinite(y1)
    ax1.plot(days[y1_mask], y1[y1_mask], color=color, marker='o', label=label)

ax1.tick_params(axis='y')
ax1.legend(fontsize=tickfsize,loc ='lower left')

#add line for puro
plt.axvline(x=3, color='k', linestyle='--') # Add a vertical dashed line after day 3
plt.text(4, 95, '+10 μg/mL Puro',ha='center', fontsize=tickfsize) # Add text "+Puro" below the x-axis

fig.tight_layout()  
plt.title('Cell Viability Post EP',fontsize=fsize)
plt.grid(True)

# Set the same y-axis limits for all plots
ax1.set_ylim([v_min-10,v_max+10])

plt.show()

savename = fig_output+'Supplementary/CRISPR Puro Viability.png'
fig.savefig(savename, bbox_inches = 'tight', dpi=1000)
#%%

data_relish = {}
data_relish_norm = {}

# Loop through the nested dictionary
for fname in dict_intensities.keys():
    enh = fname.split('_')[1]
    treat = fname.split('_')[2]
    rep =  fname.split('_')[3]
      
    cond = enh+'_'+treat
    celldf = pd.DataFrame({cell: [intensity for time, intensity in values] for cell, values in dict_intensities[fname]['relish_ratio'].items()})
    
    #normalize by average of all prestim values
    column_norm = pd.DataFrame() 
    for column in celldf.columns: 
        column_norm[column] = celldf[column] / np.mean(celldf[column][:stim])
    
    if cond not in data_relish.keys():
        data_relish[cond] = celldf
        data_relish_norm[cond] = column_norm
    else:
        data_relish[cond]= pd.concat([data_relish[cond], celldf], axis=1)
        data_relish_norm[cond]= pd.concat([data_relish_norm[cond], column_norm], axis=1)

# DELETE FOV20 MTK10X CELL 73
data_relish['Mtk_10X'] = data_relish['Mtk_10X'].drop(columns=['Cell 73'])
data_relish_norm['Mtk_10X'] = data_relish_norm['Mtk_10X'].drop(columns=['Cell 73'])


# Create lineplot ---------------------------------------------------------------------
all_conditions = list(data_relish.keys())
ordered_conditions = ['Dpt_NoInj', 'Dpt_-PGN', 'Dpt_10X', 'Dpt_100X', 'Mtk_NoInj', 'Mtk_-PGN', 'Mtk_10X', 'Mtk_100X']
#ctrl_conditions = ['Dpt_NoInj', 'Dpt_-PGN', 'Mtk_NoInj', 'Mtk_-PGN']
ctrl_conditions = ['Dpt_-PGN', 'Mtk_-PGN']


cond_toplot = ctrl_conditions


if cond_toplot == ctrl_conditions:
    fig, axes = plt.subplots(nrows=len(cond_toplot), ncols=2, figsize=(6.5, 5), sharex=True, sharey=False, dpi=1000)
else:
    fig, axes = plt.subplots(nrows=len(cond_toplot), ncols=2, figsize=(11, 20), sharex=True, sharey=False, dpi=700)

# Loop through each condition and plot
for i, condition in enumerate(cond_toplot):
    # Plot individual lines
    for column in range(data_relish[condition].shape[1]):
        sns.lineplot(x=interval_forplt, y=data_relish[condition].iloc[:, column], ax=axes[i, 0], legend=None, color='black', alpha=0.2, linewidth=0.7)
        sns.lineplot(x=interval_forplt, y=data_relish_norm[condition].iloc[:, column], ax=axes[i, 1], legend=None, color='black', alpha=0.2, linewidth=0.7)
    
    # Plot mean and standard deviation
    mean_original = data_relish[condition].mean(axis=1)
    std_original = data_relish[condition].std(axis=1)
    mean_normalized = data_relish_norm[condition].mean(axis=1)
    std_normalized = data_relish_norm[condition].std(axis=1)
    
    axes[i, 0].plot(interval_forplt, mean_original, color='blue', linewidth=2)
    axes[i, 0].fill_between(interval_forplt, mean_original - std_original, mean_original + std_original, color='blue', alpha=0.4)
    
    axes[i, 1].plot(interval_forplt, mean_normalized, color='blue', linewidth=2)
    axes[i, 1].fill_between(interval_forplt, mean_normalized - std_normalized, mean_normalized + std_normalized, color='blue', alpha=0.4)
    

    # Add titles and labels
    axes[i, 0].set_title(f'{condition}', fontsize= fsize)
    axes[i, 0].set_xlabel('Time (min)', fontsize= fsize )
    axes[i, 0].set_ylabel('Intensity (AU)', fontsize= fsize )
    axes[i, 0].set_ylim(0.3,0.8)

    
    axes[i, 1].set_title(f'{condition} (Normalized)', fontsize= fsize)
    axes[i, 1].set_xlabel('Time (min)', fontsize= fsize )
    axes[i, 1].set_ylabel('Fold-Change Intensity', fontsize= fsize )
    axes[i, 1].set_ylim(0.8, 1.6)

for ax in axes.flatten(): 
    ax.grid(True) 
    ax.set_xlim([min(interval_forplt), max(interval_forplt)]) # Ensure x-limits are consistent
    ax.axvline(x=interval_forplt[stim], color='red', linestyle='--', linewidth=1)

# Adjust layout
plt.tight_layout()
plt.show()

savename = fig_output+'Supplementary/noise20.png'
fig.savefig(savename, bbox_inches = 'tight', dpi=1000)

#%%
# Function to get the min and max values for normalization
def get_min_max(data_dict):
    all_values = np.concatenate([df.values.flatten() for df in data_dict.values()])
    return all_values.min(), all_values.max()

# Get min and max values for normalization
vmin_relish, vmax_relish = get_min_max(data_relish)
vmin_relish_norm, vmax_relish_norm = get_min_max(data_relish_norm)
new_cond = ['Dpt_-PGN',
 'Dpt_10X',
 'Dpt_100X',
 'Mtk_-PGN',
 'Mtk_10X',
 'Mtk_100X']
# Create subplots
fig, axes = plt.subplots(nrows=len(new_cond), ncols=2, figsize=(6.5, 8), sharex=True, sharey=False, dpi=700)


# Loop through each condition and plot
for i, condition in enumerate(new_cond):
    # Plot heatmap for original data
    sns.heatmap(data_relish[condition].T, cmap='viridis', ax=axes[i, 0], cbar=True, vmin=vmin_relish, vmax=vmax_relish, yticklabels=False)

    # Plot heatmap for normalized (capped) data
    sns.heatmap(data_relish_norm[condition].T, cmap='viridis', ax=axes[i, 1], cbar=True, vmin=vmin_relish_norm, vmax=vmax_relish_norm, yticklabels=False)
    
    # Add titles and labels
    axes[i, 0].set_title(f'{condition}', fontsize=fsize*mag)
    axes[i, 0].set_xlabel('Time (min)', fontsize=fsize)
    axes[i, 0].set_ylabel('Cells', fontsize=fsize)
    
    axes[i, 1].set_title(f'{condition} (Normalized)', fontsize=fsize*mag)
    axes[i, 1].set_xlabel('Time (min)', fontsize=fsize)
    axes[i, 1].set_ylabel('Cells', fontsize=fsize)

# Adjust layout
plt.tight_layout()
plt.show()


# Determine the global min and max values for the colormap range
vmin, vmax = df['Ratio'].min(), df['Ratio'].max()

# Create subplots
fig, axes = plt.subplots(nrows=len(ordered_conditions), ncols=2, figsize=(6, 20), sharex=True, sharey=False, dpi=700)

# Loop through each condition and plot
for i, condition in enumerate(ordered_conditions):
    # Sort data for heatmap rows in descending order
    data_sorted = data_relish[condition].T.loc[data_relish[condition].T.max(axis=1).sort_values(ascending=False).index]
    data_norm_sorted = data_relish_norm[condition].T.loc[data_relish_norm[condition].T.max(axis=1).sort_values(ascending=False).index]

    # Plot heatmap for original data
    sns.heatmap(data_sorted, cmap='viridis', ax=axes[i, 0], cbar=(i == 0), vmin=vmin_relish, vmax=vmax_relish, yticklabels=False,
                cbar_ax=fig.add_axes([0.93, 0.7 - i * 0.2, 0.02, 0.2]) if i == 0 else None)

    # Plot heatmap for normalized (capped) data
    sns.heatmap(data_norm_sorted, cmap='viridis', ax=axes[i, 1], cbar=(i == 1), vmin=vmin_relish_norm, vmax=vmax_relish_norm, yticklabels=False,
                cbar_ax=fig.add_axes([0.93, 0.7 - i * 0.2 - 0.2, 0.02, 0.2]) if i == 1 else None)
    
    # Add titles and labels
    axes[i, 0].set_title(f'{condition}', fontsize=fsize*mag)
    axes[i, 1].set_title(f'{condition} (Normalized)', fontsize=fsize*mag)

    if i== i[-1]:
        axes[i, 0].set_xlabel('Time (min)', fontsize=fsize)
        axes[i, 1].set_xlabel('Time (min)', fontsize=fsize)

    axes[i, 0].set_ylabel('Cells', fontsize=fsize)
    
    
# Adjust layout
plt.tight_layout()
plt.show()

#%% Predictive SVM histograms

predSVMfiles = gdrive+'Paper/Figures/Figure Codes/Supplementary Dicts/Predictive SVM/'

with open(predSVMfiles+"goodcomp3_location_averages_df_area.pkl", 'rb') as handle:
    averages_df = pickle.load(handle)

with open(predSVMfiles+"goodcomp3_locations_dict_area.pkl", 'rb') as handle:
    locations_dict = pickle.load(handle)
    
with open(predSVMfiles+"goodcomp3_predictor_SVM_results_df_area_imm.pkl", 'rb') as handle:
    predictor_SVM_results_df = pickle.load(handle)
    
with open(predSVMfiles+"goodcomp3_SVM_results_dict_noIc.pkl", 'rb') as handle:
    SVM_results_dict = pickle.load(handle)

num_categories = 'imm'
stim_time = 30
plot=True


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
    
#%%
#def plot_feature_traces(locations_dict, averages_df, SVM_results_dict, predictor_SVM_results_df, num_categories = 5, stim_time = 30):
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
    #axs_inset = inset_axes(axs[0, n], width = "30%", height = "15%", loc = "upper right", borderpad = 1.5)
    
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
        #axs[0, n].set_ylabel(trace_ylabel, fontsize = 14)
        
        axs[0, n].tick_params(axis='y', labelsize=20)
        axs[0, n].tick_params(axis='x', labelsize=20)
        
        # # plot traces in inset
        # axs_inset.plot(times, traces.T, linewidth = 0.5, color = color, alpha = 0.6)
        # axs_inset.set_xlim(ins_x1, ins_x2)
        # axs_inset.set_ylim(ins_y1, ins_y2)
        # axs_inset.set_xticks([ins_x1, ins_x2])
        # axs_inset.set_yticks([ins_y1, ins_y2])
        
        
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
        axs[i + 1, n].hist(array, bins = hist_bins, density = True, color = colors)
        
        # set axes ticks and labels
        axs[i + 1, n].set_xlabel(f"{trace_ylabel_dict[location]} ({data_type})", fontsize = 14)
        # axs[i + 1, n].set_ylim(0, 800)
        if n == 0:
            axs[i + 1, n].set_ylabel("Density (cells)", fontsize = 14)
        else:
            axs[i + 1, n].set_yticklabels([])
        
        axs[i + 1, n].tick_params(axis='y', labelsize=20)
        axs[i + 1, n].tick_params(axis='x', labelsize=20)
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
plt.savefig(fig_output+'Supplementary/predictor_hist.png', dpi = 1000)
#plt.show()