# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:58:21 2024

@author: noshin
"""

# packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import colorsys
import os
import pickle
import copy
import random
from scipy.stats import chi2_contingency
from itertools import combinations
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset, inset_axes
from statsmodels.sandbox.stats.multicomp import multipletests
from statannotations.Annotator import Annotator
import datetime
current_date   = datetime.date.today()
formatted_date = current_date.strftime("%Y-%m-%d")

#!!! Update path file!!!
gitdir = 'G:/path/' 
#!!! Update path file!!!

files_import = gitdir+'Figure 2 Files/'
fig_output = gitdir+'Temp Output/Fig 2/'

plt.rcParams['font.family'] = 'Arial'
labelfsize = 12
fsize = 10
tickfsize = 9
mag = 1.2
shrink = 0.78

def import_data(file_path, object_name, file_type = ".pkl"):   
    full_file_path = file_path + "/" + object_name + file_type
    # print(full_file_path)
    with open(full_file_path, "rb") as f:
        object_data = pickle.load(f)
        
    return object_data


#%% Import Data

#import variables needed from clustering code to run SVM:
subcluster_traces_smooth        = import_data(files_import, "goodcomp3_subcluster_traces_div_smooth")
results_dict                    = import_data(files_import+'Classifier Outputs 11.18/', "results_dict")
cell_categories_df              = import_data(files_import+'Classifier Outputs 11.18/', "cell_categories_df")
dict_trace_descriptors_SVM      = import_data(files_import+'Classifier Outputs 11.18/', "dict_trace_descriptors_SVM")
all_traces_df                   = import_data(files_import+'Classifier Outputs 11.18/', "all_traces_df")

goodcomp3_predictor_SVM_results_df_area_imm     = import_data(files_import+'Predictor Outputs 11.18/', "goodcomp3_predictor_SVM_results_df_area_imm")
goodcomp3_locations_dict_area                   = import_data(files_import+'Predictor Outputs 11.18/', "goodcomp3_locations_dict_area")

stim=4 #first slice after inject -1!!!
interval_forplt = np.concatenate([np.arange(0, 121, 15) , np.arange(150, 631, 30), np.arange(690, 971, 60)])
interval_forplt_adj = interval_forplt-interval_forplt[stim].tolist()

#%% Fig 2B Plot classification of traces
figsize1 = (6.3, 2.4)

#---------------------------------------- Select sample traces and parameters
random_cell = False
cell_names = {"N": "1X Cell 20240801-2-112",
                   "G": "1X Cell 20240410-2-81",
                   "I": "100X Cell 20240410-6-60",
                   "D": "10X Cell 20240801-3-75",
                   "Id": "100X Cell 20240410-6-67"
}
treatment = 'all' #'100X'
show_title = False
stim_time=30

colors          = ["#DC143C", "#FF6F61", "indigo", "dodgerblue", "grey"]
color_palette   = sns.color_palette(colors)
behavior_colors = {"I": color_palette[0], "Id": color_palette[1], "G": color_palette[2], "D": color_palette[3], "N": color_palette[4]}
behavior_keys   = {"I": "Immediate", "Id": "Immediate with decay", "G": "Gradual", "D": "Delayed", "N": "Nonresponsive"}
behavior_rev    = {v: k for k, v in behavior_keys.items()}

offset          = 30 #pre-stimulus time
times           = list(all_traces_df.columns-offset)

#----------------------------------------  plot figure
fig, axs    = plt.subplots(2, 5, figsize = figsize1, sharex = True, sharey = True)
    
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
    axs[1, n].plot(times, traces.T, linewidth = 0.5, color = color, alpha=0.4)
    axs[1, n].plot(times, traces.mean(axis=0), color = "black", linewidth = 1.5)
    
    axs[0, n].set_xticks(ticks = [-30, 0, 200,400,600,800,1000], labels=[])
    axs[1, n].set_xticks(ticks = [-30, 0, 200,400,600,800, 1000], labels=['-30','','','400','', '800', ''])
    # set axes labels and title
    title_color = behavior_color
    axs[1, n].tick_params(axis = "both", labelsize = tickfsize)
    axs[1, n].text(y = 1.75, x = 650-offset, s = f"n = {num_cells}", ha = "center", va = "center", fontsize = tickfsize)
    
    # highlight pre- and post-stim times
    y_min, y_max = axs[1, n].get_ylim()
    prestim  = patches.Rectangle((-offset, 0.6), 30, 0.05, edgecolor="black", facecolor="none", hatch='///')
    hatch = patches.Rectangle((0, 0.6), times[-1], 0.05, color='black')
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
    if SVM_behavior == 'Id':
        axs[0, n].set_title('Imm + Decay', fontsize = fsize, color = behavior_color)
    elif SVM_behavior == 'I':
        axs[0, n].set_title('Imm', fontsize = fsize, color = behavior_color)
    else:
        axs[0, n].set_title(f"{behavior_name}", fontsize = fsize, color = behavior_color)

    axs[0, n].tick_params(axis = "both", labelsize = tickfsize)
    
    # set y-axis limits and ticks
    axs[0, n].set_ylim(0.6, 2.0)
    axs[0, n].set_yticks([0.8, 1.2, 1.6, 2])
    
    # plot horizontal line at 1.2
    axs[0, n].axhline(y = 1.2, color = "grey", linestyle = "dashed", linewidth = 0.5)
    
    max_val = cell_data["Max Value"]
    max_time = cell_data["Max Time"]-offset
    
    if SVM_behavior != "N":
        # add vertical lines at t = 150-30
        axs[0, n].axvline(x = 150-offset, color = "grey", linestyle = "dashed", linewidth = 0.5)
        axs[0, n].text(y = 1.75, x = 140-offset*1.8, s = "t=120", ha ='center', va = 'center', fontsize = tickfsize*shrink, color = 'grey', fontstyle = "italic", rotation = 90)
        t150_val = cell_trace[150]
        
        # if n==0:
        # # add "Initial behavior" label
        #     axs[0, n].text(y = 0.8, x = 75-offset, s = "Initial\nBehavior", ha = 'center', va = 'center', fontsize = tickfsize*shrink, color = 'black')
    
    if cell_data["Local Max"] != []:
        peak1_time = cell_data["Local Max"][0][0]-offset
        peak1_val  = cell_data["Local Max"][0][1]
        # print(peak1_time)
    else:
        peak1_time = max_time-offset
        peak1_val = max_val
    
    if "I" in SVM_behavior:
        # plot max point
        axs[0, n].plot(max_time, max_val, "go", color = "black", markersize= 4)
        
        # add "Max value"
        axs[0, n].text(y = max_val + 0.03, x = max_time + 25, s = "Max", ha = 'left', va = 'center', fontsize = tickfsize*shrink, color = 'black')
        
        # plot first peak point
        if peak1_time != max_time and peak1_val != max_val:
            axs[0, n].axvline(x = peak1_time, color = "grey", linestyle = "dashed", linewidth = 0.5)
            axs[0, n].text(y = 1.85, x = peak1_time - 5, s = f"t = {peak1_time}", ha = 'center', va = 'center', fontsize = tickfsize*shrink, color = 'grey', fontstyle = "italic", rotation = 90)
            axs[0, n].plot(peak1_time, peak1_val, "go", color = "black")
            axs[0, n].text(y = peak1_val + 0.03, x = peak1_time - 5, s = "First peak", ha = 'right', va = 'center', fontsize = tickfsize*shrink, color = 'black')
        
        # shade -0.2 from first peak
        axs[0, n].fill_between(x = [peak1_time-offset, 860-offset], y1 = peak1_val, y2 = peak1_val - 0.2, color = "lightgrey", alpha = 0.5)
        # if n==0:
        #     axs[0, n].text(y = 0.8, x = ((860 - peak1_time-offset) / 2) + peak1_time, s = "Long-term\nBehavior", ha = 'center', va = 'center', fontsize = tickfsize*shrink, color = 'black')
        
        # add height arrow
        fin_val = cell_trace[-1]
        
        axs[0, n].annotate('', xy = (870-offset, fin_val-.02), xytext = (870-offset, max_val+.02),
            arrowprops = dict(
                arrowstyle     = '|-|',
                color          = behavior_color,
                linewidth      = 0.7,
                mutation_scale = 2.5)) #2.5
        
        # Add text annotation
        axs[0, n].text(885-offset, (fin_val + max_val) / 2, "Δ FC  ", color = behavior_color, ha = 'right', va = 'bottom', fontsize = tickfsize*shrink, rotation = 0)
    
    else:
        
        if SVM_behavior != "N":
            axs[0, n].axvline(x = 430-offset, color = "grey", linestyle = "dashed", linewidth = 0.5)
            axs[0, n].text(y = 1.75, x = 430-offset*1.8, s = "t=400", ha ='center', va = 'center', fontsize = tickfsize*shrink, color = 'grey', fontstyle = "italic", rotation = 90)
            t430_val = cell_trace[430-offset]
            
            # axs[0, n].plot(430, t430_val, "go", color = "black")
            # axs[0, n].text(y = t430_val + 0.03, x = 425, s = f"$t_{{430}}$", ha = "right", va = "center", fontsize = 10, color = "black")
            
            sig_filter = [(index, value) for index, value in enumerate(cell_trace) if value >= 1.2]
            sig_time, sig_val = sig_filter[0]
            # print(f"{behavior_name} ({cell_name}): [{sig_val}, {sig_time}]")
            axs[0, n].plot(sig_time-offset, sig_val, "go", color = "black", markersize= 4)
            axs[0, n].text(y = sig_val + 0.03, x = sig_time - 25-offset, s = "FC ≥ 1.2", ha = "right", va = "center", fontsize = tickfsize*shrink, color = "black")

fig.supxlabel("\nTime (min)", fontsize=fsize, y=0.05)
# fig.supylabel("Nuclear Relish fraction (fold change)", fontsize = 14)
fig.text(0, 0.6, r"Fold-Change $R_{nuc:tot}$", va='center', ha='right', fontsize=fsize, rotation='vertical')

if show_title:
    treatment_cells = len([cell for cell in list(results_df.index) if treatment in cell])
    total_cells     = all_traces_df.shape[0]
    if treatment == "all":
        plt.suptitle(f"SVM behavior classification (n = {total_cells})", fontsize = fsize)
    else:
        plt.suptitle(f"SVM behavior classification ({treatment}, n = {treatment_cells})", fontsize = fsize)      

plt.tight_layout(pad = .1)     

plt.show()

#---------------------------------------- Save
figname = 'Classified Traces_treatment'
savename = fig_output+'Fig2B_'+figname+'.png'
fig.savefig(savename, bbox_inches = 'tight', dpi=1000)

#%% Fig 2A visualize sample trace classification characteristics
figsize2 = (2.5, 1.9)  # Adjust the figure size to match a single plot

cell_name = '100X Cell 20240801-3-98'
# Flatten trace data
times = list(all_traces_df.columns-offset)

# Retrieve cell data
tmt, cell = cell_name.split(maxsplit=1)
cell_descriptors = dict_trace_descriptors_SVM[tmt].loc[cell, :]
cell_data = all_traces_df.loc[cell_name, :]

# Create a single plot
fig, ax = plt.subplots(figsize=figsize2, dpi=1000)

# Plot the trace
ax.plot(times, cell_data, linewidth=1, color="black")
ax.set_xlabel("Time (min)", fontsize=fsize)
ax.set_ylabel("Fold-Change " +r"$R_{nuc:tot}$", fontsize=fsize)
ax.tick_params(labelsize=tickfsize)

ax.set_ylim(0.6, 2.0)
ax.set_yticks([0.8, 1.2, 1.6, 2])
ax.set_xticks(ticks = [-30, 0, 200,400,600,800], labels=['-30','','200','400','600', '800'])
ax.tick_params(axis='both', labelsize=tickfsize)

# Plot max value/time
max_val = cell_descriptors["Max Value"]
max_time = cell_descriptors["Max Time"]-offset
ax.plot(max_time, max_val, "go", color="black", markersize=4)
ax.axvline(x=max_time, color="k", linestyle="dashed", linewidth=0.5)
ax.text(y=1.95, x=max_time - 30, s="t max", ha='center', va='top', fontsize=tickfsize*shrink, color='k', fontstyle="italic", rotation=90)
ax.axhline(y=max_val, color="k", linestyle="dashed", linewidth=0.5)
ax.text(y=max_val + 0.055, x=860-offset, s="value max", ha='right', va='center', fontsize=tickfsize*shrink, color='k', fontstyle="italic")

# Plot half max value/time
half_val = cell_descriptors["Half Max"]
half_time = cell_descriptors["Half Time"]-offset
ax.plot(half_time, half_val, "go", color="black", markersize=4)
ax.axvline(x=half_time, color="k", linestyle="dashed", linewidth=0.5)
ax.text(y=1.95, x=half_time - 30, s="t 1/2max", ha='center', va='top', fontsize=tickfsize*shrink, color='k', fontstyle="italic", rotation=90)
#ax.axhline(y=half_val, color="k", linestyle="dashed", linewidth=0.5)
#ax.text(y=half_val + 0.055, x=860-offset, s="Half Max Value", ha='right', va='center', fontsize=ticksize, color='k', fontstyle="italic")

# Shade area under curve
ax.fill_between(times, cell_data, color='lightgrey', alpha=0.5)
ax.text(y=0.8, x=times[-1] / 2, s="Area", ha="center", va="center", fontsize=fsize, color="k")

plt.tight_layout()
plt.show()

#---------------------------------------- Save
figname = 'Classification Labels'
savename = fig_output+'Fig2A_'+figname+'.png'
fig.savefig(savename, bbox_inches = 'tight', dpi=1000)

#%% Fig 2C Stacked bar plot
plot = "hor"
figsize3_hor =  (3.5,1.6)
figsize3_vert = (3,6)

#---------------------------------------- Compile data
dict_trace_descriptors = dict_trace_descriptors_SVM
treatments = [treatment for treatment in list(dict_trace_descriptors.keys())]
treatments = sorted(treatments, key = lambda x: (x != '-PGN', int(x[:-1]) if x != '-PGN' else -1))
PGN_concs  = [tmt.strip("X") if "X" in tmt else 0 for tmt in treatments]
behavior_dict = {
        "N": "Nonresponsive",
        "D": "Delayed",
        "G": "Gradual",
        "I": "Immediate",
        "Id": "Immediate with decrease"}
col_names = ["# cells"] + [behavior for behavior in behavior_dict]
percents_df = pd.DataFrame(index = treatments, columns = col_names)
celln_df = pd.DataFrame(index = treatments, columns = col_names)

for tmt, tmt_df in dict_trace_descriptors.items():
    total_cells = tmt_df.shape[0]    
    percents_df.loc[tmt, "# cells"] = total_cells
    
    for behavior in col_names:
        if behavior == "# cells":
            continue
        
        behavior_name = behavior_dict[behavior]
        filtered_data = tmt_df[tmt_df['Behavior'] == behavior]
        sum_cells = filtered_data.shape[0]
        celln_df.loc[tmt, behavior] = sum_cells
        percent_cells = (sum_cells / total_cells) * 100
        percents_df.loc[tmt, behavior] = percent_cells

##----------------------------------------Stats functions

#chi squared test on bars
def get_asterisks_for_pval(p_val):
    if p_val > 0.05:
        p_text = "ns"  # not significant
    # elif p_val < 1e-4:
    #     p_text = '****'
    # elif p_val < 1e-3:
    #     p_text = '***'
    elif p_val < 1e-2:
        p_text = '**'
    else:
        p_text = '*'
    return p_text

# Perform chi-squared test and post-hoc correction
def chisq_and_posthoc_corrected(df):
    # Start by running chi2 test on the matrix
    chi2, p, dof, ex = chi2_contingency(df, correction=True)
    print(f"Chi2 result of the contingency table: {chi2}, p-value: {p}")

    # Post-hoc
    all_combinations = list(combinations(df.index, 2))  # gathering all combinations for post-hoc chi2
    p_vals = []
    for comb in all_combinations:
        new_df = df[(df.index == comb[0]) | (df.index == comb[1])]
        chi2, p, dof, ex = chi2_contingency(new_df, correction=True)
        p_vals.append(p)

    # Correction for multiple testing
    reject_list, corrected_p_vals = multipletests(p_vals, method='fdr_bh')[:2]
    significant_pairs = [(comb, p_val, corr_p_val) for comb, p_val, corr_p_val, reject in zip(all_combinations, p_vals, corrected_p_vals, reject_list) if reject]
    
    return significant_pairs, p
   
# Adjust the plot to include bars and stars
def add_significance_bars_vert(ax, pairs, y_max, index_list):
    heights = iter(range(1, len(pairs) + 1))
    for (start, end), p_val, corr_p_val in pairs:
        start_idx = index_list.index(start)
        end_idx = index_list.index(end)
        x1, x2 = start_idx, end_idx
        y, h, col = y_max + next(heights) *5, 1.5, 'k'  # Adjust y and h for more space, staggered heights (0.1, 0.02 originally)
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c=col)
        ax.text((x1+x2)*.5, y+h*.6, get_asterisks_for_pval(corr_p_val), ha='center', va='bottom', color=col, fontsize= tickfsize)

def add_significance_bars_horiz(ax, pairs, x_max, index_list):
    heights = iter(range(1, len(pairs) + 1))
    for (start, end), p_val, corr_p_val in pairs:
        start_idx = index_list.index(start)
        end_idx = index_list.index(end)
        y1, y2 = start_idx, end_idx
        x, h, col = x_max + next(heights) *8, 2, 'k'  # Adjust x and h for more space, staggered heights (5, 2 originally)
        ax.plot([x, x+h, x+h, x], [y1, y1, y2, y2], lw=.5, c=col)
        ax.text((x + h), (y1 + y2)/2, get_asterisks_for_pval(corr_p_val), ha='center', va='center', color=col, fontsize=tickfsize, rotation=-90)

chiq_test_df =  celln_df.iloc[:, 1:]
sig_pairs, p  = chisq_and_posthoc_corrected(chiq_test_df.astype(float))

##---------------------------------------- plot horz or vertical barplot

colors          = ["#DC143C", "#FF6F61", "indigo", "green", "dodgerblue", "grey"]
color_palette   = sns.color_palette(colors)
treatment_color = {
        "Immediate": color_palette[0], 
        "Immediate with decrease": color_palette[1],
        "Gradual": color_palette[2],
        "Delayed": color_palette[4],
        "Nonresponsive": color_palette[5]}
color_list = [treatment_color[behavior_dict[behavior]] for behavior in col_names if behavior != "# cells"]
cols_to_plot = percents_df.columns[1:]

if plot == "vert":
    ax = percents_df[cols_to_plot].plot.bar(stacked=True, figsize= figsize3_vert, color = color_list)

    plt.xlabel(r'[PGN] ($\mu$g/mL)', fontsize = fsize)
    plt.ylabel('% cells', fontsize = fsize)
    # plt.xticks(rotation = 0)
    plt.xticks(np.arange(len(PGN_concs)), PGN_concs, rotation = 0)
    ax.tick_params(axis = "both", labelsize = tickfsize)
    # plt.title('Distribution of Relish dynamics', fontsize = 16)
    
    for i, treatment in enumerate(treatments):
        x = percents_df.columns[1:]
        y = percents_df.loc[treatment, x].values
        total_cells = percents_df.loc[treatment, "# cells"]
        
        ax.text(x = i, y = y.sum() + 1, s = f"({total_cells})", ha = 'center', va = 'bottom', fontsize = tickfsize, color = 'black', fontstyle = "italic")
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_yticks([], minor=True)
        ax.set_yticklabels([tick if tick <= 100 else '' for tick in ax.get_yticks()])

        y_max = 105
        add_significance_bars_vert(ax, sig_pairs, y_max, list(chiq_test_df.index))

    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.show()
    
    #---------------------------------------- Save
    figname = 'Barplot_vert'
    savename = fig_output+'Fig2C_'+figname+'.png'
    plt.savefig(savename, bbox_inches = 'tight', dpi=1000)

    
elif plot == "hor":
    ax = percents_df[cols_to_plot].plot.barh(stacked=True, figsize=figsize3_hor, color = color_list)

    plt.ylabel(r'[PGN] ($\mu$g/mL)', fontsize = fsize)
    plt.xlabel('% Cells', fontsize = fsize)
    plt.yticks(np.arange(len(PGN_concs)), PGN_concs, rotation = 0)
    ax.tick_params(axis = "both", labelsize = tickfsize)
    
    for i, treatment in enumerate(treatments):
        y = percents_df.columns[1:]
        x = percents_df.loc[treatment, y].values
        total_cells = percents_df.loc[treatment, "# cells"]
        
        ax.text(y = i - 0.23, x = x.sum() + 1, s=f"({total_cells})", ha='center', va='bottom', fontsize= tickfsize*shrink, color='black', fontstyle = "italic", rotation = -90)
        
        x_max = percents_df[cols_to_plot].values.max() + 20 
        add_significance_bars_horiz(ax, sig_pairs, x_max, list(chiq_test_df.index))
        ax.set_xticks([tick for tick in ax.get_xticks() if tick <= 100])

    plt.legend(bbox_to_anchor=(1, 1.02), loc='upper left', fontsize=tickfsize)
    #plt.show()
    
    #---------------------------------------- Save
    figname = 'Barplot_horz'
    savename = fig_output+'Fig2C_'+figname+'.png'
    plt.savefig(savename, bbox_inches = 'tight', dpi=1000)


#%% Fig 2E Predictors Output
figsize5 = (3.52,2.23)

num_categories = "imm"
locations_dict = goodcomp3_locations_dict_area
results_df = goodcomp3_predictor_SVM_results_df_area_imm
treatment = "all"
stim_time = 30
num_categories = 'imm'
offset=30

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
behavior_rev    = {v: k for k, v in behavior_keys.items()}

# flatten trace data
#all_traces_df   = flatten_trace_df(subcluster_traces_smooth)
times           = list(all_traces_df.columns-offset)

# initialize figure
if num_categories != "imm":
    fig, axs        = plt.subplots(1, num_categories, figsize = (5 + (5 * num_categories), 10), sharex = True, sharey = True)
else:
    fig, axs        = plt.subplots(1, 2, figsize = figsize5, sharex = True, sharey = True, dpi=1000)

for n, (SVM_behavior, behavior_name) in enumerate(behavior_keys.items()):
    behavior_traces = pd.DataFrame()
    num_cells       = 0
    
    # define inset limits
    ins_x1, ins_x2, ins_y1, ins_y2 = 0-offset, stim_time-offset, 0.5, 1.5
    
    # plot zoomed inset axis
    axs_inset = inset_axes(axs[n], width = "50%", height = "25%", loc = "upper right", borderpad = .5)
    
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
        axs[n].plot(times, traces.T, linewidth = 0.5, color = color, alpha=0.4)
        
        # plot traces in inset
        axs_inset.plot(times, traces.T, linewidth = 0.5, color = color, alpha=0.4)
        axs_inset.set_xlim(ins_x1, ins_x2)
        axs_inset.set_ylim(ins_y1, ins_y2)
        axs_inset.set_xticks([ins_x1, ins_x2])
        axs_inset.set_yticks([ins_y1, ins_y2])
        axs_inset.tick_params(axis = "both", labelsize = tickfsize)
    
    num_cells   = behavior_traces.shape[0]
    
    # plot averages
    axs[n].plot(times, behavior_traces.mean(axis = 0), color = "black", linewidth = 1.5)
    
    # set axes labels and title
    title_color = behavior_colors[SVM_behavior]
    axs[n].set_title(f"{behavior_name} (n={num_cells})", fontsize = tickfsize, color = title_color)
    axs[n].tick_params(axis = "both", labelsize = tickfsize)
    
    # set y-axis limits and ticks
    axs[n].set_ylim(0.82, 2.1)
    axs[n].set_yticks([0.8, 1.2, 1.6, 2.0])
    axs[n].set_xticks(ticks = [-30, 0, 200,400,600,800], labels=['-30','','','400','', '800'])
    axs[n].tick_params(axis='both', labelsize=tickfsize)

    # highlight pre- and post-stim times
    y_min, y_max = axs[n].get_ylim()
    prestim  = patches.Rectangle((-30, 0.6), 30, 0.05, edgecolor="black", facecolor="none", hatch='///')
    hatch = patches.Rectangle((0, 0.6), times[-1], 0.05, color= 'black')
    axs[n].add_patch(prestim)
    axs[n].add_patch(hatch)
    
fig.supxlabel("Time (min)", fontsize=fsize, y=0.05)
fig.text(0, 0.5, r"    Fold-Change $R_{nuc:tot}$", va = 'center', ha = 'center', fontsize = fsize, rotation = 'vertical')
treatment_cells = len([cell for cell in list(results_df.index) if treatment in cell])
total_cells     = results_df.shape[0]

fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.21,  wspace=0.08)
plt.tight_layout(pad=0.5)
plt.show()

#---------------------------------------- Save
figname = 'Predictor Outputs'
savename = fig_output+'Fig2E_'+figname+'.png'
fig.savefig(savename, bbox_inches = 'tight', dpi=1000)
