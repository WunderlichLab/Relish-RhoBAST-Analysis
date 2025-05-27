# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 14:48:07 2025

@author: noshin
"""
import pandas as pd
import seaborn as sns
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from scipy.stats import kstest, expon
from statannotations.Annotator import Annotator
import pickle
import time
from datetime import timedelta
datetime_str = time.strftime("%m%d%y_%H:%M")

#!!! Update path file!!!
gitdir = 'G:/path/' 
#!!! Update path file!!!

files_import = gitdir+'Figure 6 Files/'
fig_output = gitdir+'Temp Output/Fig 6/'

plt.rcParams['font.family'] = 'Arial'
labelfsize = 12
fsize = 10
tickfsize = 9
mag = 1.2
shrink = 0.78


#%% Fig 6A Import and plot sample dense data trace

with open(os.path.join(files_import, 'dataframe_peakproperties_alldays_prom200_relativheight0.9_moredetail_+-2min_042525.pkl'), 'rb') as handle:
    allpeaks_df = pickle.load(handle)
t = 'T1'
good_cell = 'Dpt-20250306-95'
df_entry            = allpeaks_df[ (allpeaks_df['ID'] == good_cell) & (allpeaks_df['Time Window'] == t)]

times               = df_entry['Time (min)'].tolist()[0]
rb_int_smooth       = df_entry['Smoothed RhoBAST Intensity'].tolist()[0]
rel_int_smooth      = df_entry['Smoothed Relish Ratio'].tolist()[0]
peaks_time          = df_entry['Peak Times (min)'].tolist()[0]
peak_max_heights    = df_entry['Prominences'].tolist()[0]
left_bases_time     = df_entry['Left Bases (min)'].tolist()[0]
left_base_heights   = df_entry["Left Base Heights"].tolist()[0]
right_bases_time    = df_entry['Right Bases (min)'].tolist()[0]
right_base_heights  = df_entry["Right Base Heights"].tolist()[0]
width_heights       = df_entry["Width Heights"].tolist()[0]
avg_width           = df_entry["Average Width (min)"].tolist()[0]
rising_slopes       = df_entry['Rising Slopes'].tolist()[0]
avg_slope           = np.nanmean(rising_slopes)
periods             = df_entry['Periods'].tolist()[0]
avg_period          = np.nanmean(periods)
off_times           = df_entry['Off Times (min)'].tolist()[0]
avg_off_time       = df_entry['Average Off Times (min)'].tolist()[0]


#% --------------------------Plot RhoBAST intensity on the left y-axis
fig, ax1 = plt.subplots(figsize=(3.8, 2))

# Plot the smoothed intensity and peaks
ax1.plot(times, rb_int_smooth, linewidth=1.2, color='black', alpha=.8 , zorder = 0)
ax1.plot(peaks_time, peak_max_heights, marker='.', markersize=4, linewidth=0, color='green', zorder = 2)
ax1.plot(left_bases_time, left_base_heights, marker='.', markersize=4, linewidth=0, color='green', zorder = 2)


# Dashed and slope lines
for i in range(len(peaks_time)):
    # Dashed line from left base to peak
    ax1.plot([peaks_time[i], peaks_time[i]],
              [width_heights[i], peak_max_heights[i]],
              linestyle="-", color='black', alpha=0.6, linewidth=0.5, zorder=1)
    
    #Dashed line along width # add 2 min before to account for lacz enlongation
    ax1.plot([left_bases_time[i], peaks_time[i]],
             [width_heights[i], width_heights[i]],
             linestyle="dashed", color='black', alpha=0.6, linewidth=0.5, zorder =3)
    
    # Slope line in green
    ax1.plot([left_bases_time[i], peaks_time[i]],
             [left_base_heights[i], peak_max_heights[i]],
             linestyle="-", linewidth=0.5, color='green', zorder = 2)

# Off Time annotation (red lines with markers)
if len(peaks_time) > 1:
    for i in range(len(peaks_time) - 1):
        height = -100
        ax1.plot([peaks_time[i], left_bases_time[i + 1]],
                 #[left_base_heights[i], left_base_heights[i + 1]],
                 [height, height],
                 linestyle="-", linewidth=0.5, color='red',
                 marker='.', markersize=4, zorder = 1)

   
#--------------------------Second y-axis for Relish signal
ax2 = ax1.twinx()
ax2.plot(times, rel_int_smooth, color='grey', alpha=0.3, linewidth=1.5, label='Relish', zorder = 0)


# Labeling
ax1.set_xlabel("Time (min)", fontsize= fsize)
ax1.set_ylabel("Int$_R$$_B$ (AU)", fontsize= fsize)
ax1.tick_params(labelsize=tickfsize)  # Set colorbar tick labels size

ax1.tick_params(axis='y', which='both', pad=.05)
ax1.tick_params(axis='x', which='both', pad=.1)
ax1.yaxis.set_label_coords(-.12, 0.5)

ax2.set_ylabel('$R_{nuc:tot}$', color='grey', fontsize= fsize)  # y-axis label in blue
ax2.tick_params(axis='y', labelcolor='grey')      # y-axis tick labels in blue
ax2.tick_params(labelsize=tickfsize)  
ax2.tick_params(axis='y', which='both', pad=.05)

ax1.set_yticks([0,500,1000,1800],labels=['0','500','1000',''], fontsize= tickfsize)  # y-axis label in blue
ax2.set_yticks([.4,.5,.6, .82],labels=['0.4','0.5','0.6',''], fontsize= tickfsize)  # y-axis label in blue

ax1.set_xlim(-5, 90)
ax1.set_xlim(-5,90)

#legend on top center
custom_handles = [
           Line2D([0], [0], color='green', lw=0.5, label= f'Avg Slope = {avg_slope:.1f} (AU/min)'),
           Line2D([0], [0], color='black', linestyle="dashed", lw=0.5, label=f'Avg On Time = {avg_width:.1f} (min)'),
           Line2D([0], [0], color='red', lw=0.5, label=f'Avg Off Time = {avg_off_time:.1f} (min)'),
                   ]

ax1.legend(handles=custom_handles, fontsize=tickfsize*shrink, loc = 'upper left' ,
            framealpha=1, handlelength=1, labelspacing= 0.2, handletextpad=0.5)

plotname = good_cell + " " + t 
fig.tight_layout()
#plt.show()

#---------------------------------------- Save
figname = f"CellProperties_Trace_2minadj_{good_cell}_{t}"
savename = fig_output+'Fig 6A_'+figname+'.png'
fig.savefig(savename, bbox_inches = 'tight', dpi=1000)

#%% Fig 6B-D Violin Plots

# Plot the violin plot
action = 'plot'
#action = 'save'

#---------------------------------------------------------
#Pick one

# cond = 'Rising Slopes'          #per cell
# cond_units = 'Avg Slope (int/min)'
# y_scale_top= .15
# y_scale_bottom = .2
# n_height = -.38
# first = True
# last = False
# cells = True

# cond= 'Off Times (min)'       #per peak
# cond_units = 'Off Time (min)'
# y_scale_top= 0
# y_scale_bottom = 0.25
# n_height = -.5
# first = False
# last = True
# cells = False

cond = 'Widths (min)'         #per peak
cond_units = 'On Time (min)'
y_scale_top = 0
y_scale_bottom = 0.25
n_height = -.32
first= False
last= False
cells = False

   
#---------------------------------------------------
peaks_df_exploded = allpeaks_df.explode(cond)
# Convert the exploded values to numeric (non-numeric values become NaN)
peaks_df_exploded[cond] = pd.to_numeric(peaks_df_exploded[cond], errors='coerce')
peaks_df_exploded = peaks_df_exploded.dropna(subset=[cond])

figname = cond

if cond== 'Rising Slopes':
    # Calculate the 90th percentile cutoff for your y-axis column
    cutoff = peaks_df_exploded[cond].quantile(0.90)
    # Create a filtered dataframe excluding the top 10% values
    peaks_df_exploded = peaks_df_exploded[peaks_df_exploded[cond] <= cutoff]
    #peaks_df_exploded = filtered_df

    
# ------------  Create the violin plot with an inner boxplot
my_pal = {"Dpt": "blue", "Mtk": "green"}
fig = plt.figure(figsize=(3.8, 1.7))

ax = fig.add_axes([0.1, 0.1, 0.7, 0.7])  
if first:
    sns.violinplot(x='Time Window', y=cond, hue='Enhancer', data=peaks_df_exploded, 
                   palette=my_pal, inner="box", linewidth=1.2, alpha=0.7, zorder=1, ax=ax)
                   #legend=False)
    plt.legend(fontsize =tickfsize*shrink, ncol=2, loc = 'upper left', columnspacing=0.5, handletextpad = 0.2, framealpha=0)
else:
    sns.violinplot(x='Time Window', y=cond, hue='Enhancer', data=peaks_df_exploded, 
                   palette=my_pal, inner="box", linewidth=1.2, alpha=0.7, zorder=1, ax=ax, legend=False)

# Formatting
#plt.title(figname, fontsize=14)
if last:    
    plt.xlabel("Time Window", fontsize=fsize)
    plt.xticks(fontsize = tickfsize)
else:
    plt.xlabel("", fontsize=fsize)
    ax.tick_params(labelbottom=False)    
plt.ylabel(cond_units, fontsize=fsize)
plt.yticks(fontsize = tickfsize)

# Expand the y-axis limits to add extra space at the top:
ymin, ymax = ax.get_ylim()
ax.set_ylim(ymin - y_scale_bottom * (ymax - ymin), ymax+ y_scale_top * (ymax - ymin))
    
# ------------ Add cell counts 
# Get unique enhancers and their offsets for annotations
enhancers = peaks_df_exploded['Enhancer'].unique()
offsets = np.linspace(-0.2, 0.2, num=len(enhancers))  # Small shifts to separate labels

# Compute unique cell counts per (TimeWindow, Enhancer)
if cells: #count per cell
    group_counts = peaks_df_exploded.groupby(['Time Window', 'Enhancer'])['Cell'].nunique() #cells
else:     #count per peak
    group_counts = peaks_df_exploded.groupby(['Time Window', 'Enhancer'])[cond].count()

# Annotate cell counts on violin plots with slight offsets
for i, (timewin, enhancer) in enumerate(group_counts.index):
    x_loc = list(peaks_df_exploded['Time Window'].unique()).index(timewin)  # Get x-axis position
    enhancer_idx = np.where(enhancers == enhancer)[0][0]  # Get enhancer index for offset
    plt.text(x=x_loc + offsets[enhancer_idx], 
             y=peaks_df_exploded[cond].max() *n_height,
             s= (f"{group_counts[(timewin, enhancer)]}"), 
             ha='center', va='bottom', fontsize=tickfsize, color='black')


# ------------  t-test testing
# Define pairs for statistical testing (within each TimeWindow)
pairs = []
time_windows = peaks_df_exploded['Time Window'].unique()
enhancers = peaks_df_exploded['Enhancer'].unique()

for timewin in time_windows:
    enhancer_pairs = [(enh1, enh2) for i, enh1 in enumerate(enhancers) for enh2 in enhancers[i+1:]]
    for pair in enhancer_pairs:
        pairs.append(((timewin, pair[0]), (timewin, pair[1])))  # Format as (x, hue) pairs
for enhan in enhancers:
    time_pairs = [(tim1, tim2) for i, tim1 in enumerate(time_windows) for tim2 in time_windows[i+1:]]
    for time in time_pairs:
        pairs.append(((time[0], enhan), (time[1], enhan)))  # Format as (x, hue) pairs


#with Bonferroni Correction
annotator = Annotator(ax, pairs, data=peaks_df_exploded, x='Time Window', y=cond, hue='Enhancer')
annotator.configure(
    hide_non_significant=True,
    test='Mann-Whitney', comparisons_correction="Bonferroni",
    text_format='star', loc='inside', verbose=1, 
    pvalue_thresholds=[[1e-4, "**"], [1e-3, "**"], [1e-2, "**"], [0.05, "*"]]
)
annotator.apply_and_annotate()

plt.tight_layout()

if action=='plot':
    plt.show()
elif action=='save':
#---------------------------------------- Save
    savename = fig_output+'Fig 6_'+figname+'.png'#
    plt.savefig(savename, bbox_inches = 'tight', dpi=1000)

#%% Figure S8: all time fit on and off durations to exponentials

fig = plt.figure(figsize=(7, 2))
gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1], wspace=0.2 ,figure=fig)
# Create nested GridSpecs for each pair
gs_pairs = []
for i in range(3):  # 3 time windows
    gs_pair = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[:, i], wspace=0.24)
    gs_pairs.append(gs_pair)

# Create axes using the nested GridSpecs
axs = np.empty((2, 6), dtype=object)
for row, enh in enumerate(['Dpt', 'Mtk']):
    for tw_idx, tw in enumerate(['T1', 'T2', 'T3']):
        for cond_idx, cond in enumerate(['Widths (min)', 'Off Times (min)']):
            col = tw_idx * 2 + cond_idx
            pair_idx = tw_idx
            subcol_idx = cond_idx
            
            # Create the axis
            axs[row, col] = fig.add_subplot(gs_pairs[pair_idx][row, subcol_idx])
            
            #make subdata
            sub_df = allpeaks_df[(allpeaks_df['Time Window']==tw) & (allpeaks_df['Enhancer'] == enh)]
            sub_df_exp = sub_df.explode(cond)
            sub_df_exp = sub_df_exp.dropna(subset=[cond])

            cond_series = pd.to_numeric(sub_df_exp[cond], errors='coerce').dropna()
            cond_values    = cond_series.values   # now guaranteed to be a float64 array
            
            # ——— Fit by MLE on the raw data ———
            # fix loc=0 if you really want an Exp(0, scale), otherwise drop floc=0 to estimate loc too
            loc_hat, scale_hat = expon.fit(cond_values, floc=0)
            
            # ——— Plot the fitted PDF ———
            x_fit = np.linspace(0, cond_values.max(), 100)
            # if you fixed loc=0, you can omit loc=loc_hat below; but itʼs harmless:
            y_fit = expon.pdf(x_fit, loc=loc_hat, scale=scale_hat)
            axs[row, col].hist(cond_values, bins=20, density=True,
                               color=('blue' if enh=='Dpt' else 'green'),
                               alpha=0.6)
            axs[row, col].plot(
                x_fit, y_fit, '-r',
                label=f'Exp fit\nloc={loc_hat:.2f}, scale={scale_hat:.2f}'
            )
            
            # ——— KS test with the same MLE parameters ———
            ks_stat, p_value = kstest(cond_values, 'expon',
                                      args=(loc_hat, scale_hat))
            
            if p_value>0.05:
                print(f"{enh}, {tw}, {cond} → KS={ks_stat:.3f}, p={p_value:.3f}")
            axs[row,col].set_xlim([0,30])
            
            
            #----------labels
            if row == 1:
                axs[row,col].set_xlabel('On (min)' if cond_idx==0 else 'Off (min)',
                              fontsize = fsize)
                axs[row,col].set_xticks([0, 10, 20, 30], labels=('0','','20',''),
                                        fontsize = tickfsize)
            else:
                axs[row,col].set_xticks([0,10, 20, 30], labels=())

            if col == 0:
                axs[row,col].set_ylabel(enh+'\n %Density', fontsize = fsize)
            
            if col in [4]:
                axs[row,col].set_yticks([0, .1, .2, .3], labels=('0','','','30'),
                                    fontsize = tickfsize, rotation= 90)
            else:
                axs[row,col].set_yticks([0, .1, .2], labels=('0','','20'),
                                    fontsize = tickfsize, rotation= 90)
            axs[row, col].tick_params(axis='y', which='major', pad=.01)

# Add custom centered titles above column pairs
for tw_idx, tw in enumerate(['T1', 'T2', 'T3']):
    # Calculate center position between two columns
    col_start = tw_idx * 2
    col_end = col_start + 1
    # Get positions of the two axes
    pos1 = axs[0, col_start].get_position()
    pos2 = axs[0, col_end].get_position()
    # Calculate center x position
    center_x = (pos1.x0 + pos2.x1) / 2
    # Place text above the top row
    fig.text(center_x, pos1.y1 + 0.05, tw, ha='center', va='bottom', fontsize=fsize+2, fontweight='bold')


plt.show()

#---------------------------------------- Save
figname = 'OnOffExpFit'
savename = fig_output+'Fig 6_'+figname+'.png'#
plt.savefig(savename, bbox_inches = 'tight', dpi=1000)
