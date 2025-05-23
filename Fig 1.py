# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
import tifffile as tf
import numpy as np
import matplotlib.patches as patches
import math
import os
from scipy import ndimage
from skimage import measure
from skimage.measure import label, regionprops
from scipy.ndimage import label
from scipy.ndimage import binary_dilation
import pandas as pd
import tifffile
from tifffile import imread
from tifffile import imshow
from skimage.segmentation import mark_boundaries
from skimage.feature.peak import peak_local_max
from skimage.feature import corner_fast, corner_peaks
import matplotlib.gridspec as gridspec
from PIL import Image
import pickle
import datetime
import time
from datetime import timedelta
import matplotlib.patches as patches
import skimage.io
import  microfilm.microplot
from microfilm import microplot
from microfilm.microplot import microshow, Micropanel
import seaborn as sns
import gc
from collections import Counter
import copy
from scipy.signal import savgol_filter


datetime_str = time.strftime("%m%d%y_%H:%M")

mac = '/Volumes/rkc_wunderlichLab/'
PC = 'R:/'
anacomp = 'Z:/'
resolution=  3.4756 #pixels per micron
units_per_pix = 1/resolution

computer = anacomp

plt.rcParams['font.family'] = 'Arial'
labelfsize = 12
fsize = 10
tickfsize = 9


fig_output = 'G:/Shared drives/Wunderlich Lab/People/Noshin/Paper/Figures/'


#%% Individual cell frames

curr_dataset = '2024-10-02 EnhLacZ_RhoBAST_31frames(ignore)/'
project = 'attB-LacZ-RhoBAST-Halotag-Relish/' 
rootdir = computer+'Imaging Data/Noshin Imaging Data/'+project+curr_dataset
mask_settings = '15link_nuc8_cyto40' #'15link_nuc6.5-(cyto)250_cyto40-950' 

#import dataset dict_intensities
intensities_df_import = rootdir+ 'Python/15link_nuc8_cyto40/Intensities DF/'
with open(os.path.join(intensities_df_import, 'dict_intensities_goodcells_peaks_size2-150_2400_ratio2,3'), 'rb') as handle:
    dict_intensities = pickle.load(handle)
fnames = list(dict_intensities.keys())
nfiles = len(fnames)

#set dataset parameters
stim=4 #first slice after inject -1!!!
interval_forplt = np.concatenate([np.arange(0, 121, 15) , np.arange(150, 631, 30), np.arange(690, 971, 60)])
interval_forplt_adj = interval_forplt-interval_forplt[stim].tolist()

interval_hrs = [str(timedelta(minutes=int(time.item())))[:-3] for time in interval_forplt]
nuc_channel = 0
rel_channel = 1

tifs =  rootdir+'TIF_Split_Series_MaxZ/'    
cyto_masks = rootdir+'Trackmate Files/'+mask_settings+'/Cyto Matched Masks/' #Updated masks with interpolated (Python Step 3-1) then matched (Fiji Step 3-2)
nuc_masks = rootdir+ 'Trackmate Files/'+mask_settings+'/Nuclei Matched Masks/'

fname=fnames[1]
day = curr_dataset.split(' ')[0]

time_series_tif = tf.imread(tifs+fname)
cyto_mask_tif = tf.imread(cyto_masks+fname)
nuc_mask_tif = tf.imread(nuc_masks+fname)
nframes = time_series_tif.shape[0]
    
f_dict_intensities = dict_intensities[fname]
df_relish_ratio = pd.DataFrame({cell: [intensity for time, intensity in values] for cell, values in f_dict_intensities['relish_ratio'].items()})



#%% time-course cell panel
figsize1 = [2.5,1.9]

#cell 04_Dpt_10X_1_Cell 76
celln= 76
cell='Cell '+str(celln)

figname = day+"_"+fname[3:-9]+"_"+cell
fig, ax = plt.subplots(dpi=1000)

#---------------------------------------- Set min and max coordinates for cell

xmin, xmax, ymin, ymax = float('inf'), float('-inf'), float('inf'), float('-inf')
for t in range(nframes):
    cyto_mask = (cyto_mask_tif[t] == int(celln)).astype(int)
    nuc_mask = (nuc_mask_tif[t] == int(celln)).astype(int)
    # Define the limits, crop image and contours
    if np.any(np.nonzero(cyto_mask)):  # Check if the cell is present in the current frame
        props_cyto = regionprops(cyto_mask)
        miny_c, minx_c, maxy_c, maxx_c = props_cyto[0].bbox
        props_nuc = regionprops(nuc_mask)
        miny_n, minx_n, maxy_n, maxx_n = props_nuc[0].bbox
        # Update the min and max coordinates only if the cell moves beyond the current range
        xmin = min(xmin, minx_c, minx_n)
        xmax = max(xmax, maxx_c, maxx_n)
        ymin = min(ymin, miny_c, miny_n)
        ymax = max(ymax, maxy_c, maxy_n)  
        
        xmin_n = min(xmin, minx_n)
        xmax_n = max(xmax, maxx_n)
        ymin_n = min(ymin, miny_n)
        ymax_n = max(ymax,maxy_n)  
    
#---------------------------------------- Set cell and outline parameters

times = [0,12,30]
cropped_cell = {}
cropped_cyto_mask = {}
cropped_nuc_mask = {}
mins = {}

for count,t in enumerate(times):
#    cropped_cell[count] = time_series_tif[t,[nuc_channel,rel_channel], ymin:ymax, xmin:xmax]
    cropped_cell[count] = time_series_tif[t, rel_channel, ymin:ymax, xmin:xmax]

    
    cyto_mask = (cyto_mask_tif[t] == int(celln)).astype(int)
    nuc_mask = (nuc_mask_tif[t] == int(celln)).astype(int) 
    
    cropped_cyto_mask[count] = cyto_mask[ymin:ymax, xmin:xmax]
    cropped_nuc_mask[count] = nuc_mask[ymin:ymax, xmin:xmax]
    
    mins[count] = str(interval_forplt_adj[t])+ ' min'

#---------------------------------------- Microplot

#colormaps = ['pure_blue', 'pure_red']
colormaps = ['magma']


minmax = [0,3700]
microim1 = microshow(cropped_cell[0], cmaps=colormaps, label_text=mins[0], label_font_size=fsize, rescale_type='limits', limits=minmax)
microim2 = microshow(cropped_cell[1], cmaps=colormaps, label_text=mins[1], label_font_size=fsize, rescale_type='limits', limits=minmax)
microim3 = microshow(cropped_cell[2], cmaps=colormaps, label_text=mins[2], label_font_size=fsize, rescale_type='limits', limits=minmax,
                     unit='um', scalebar_unit_per_pix=units_per_pix, scalebar_size_in_units=10, 
                     scalebar_color='white', scalebar_font_size=tickfsize, 
                     show_colorbar=True) 

panel = Micropanel(rows=1, cols=3, figsize=figsize1) #figscaling=2.5)  
panel.add_element([0, 0], microim1)
panel.add_element([0, 1], microim2)
panel.add_element([0, 2], microim3)

#----------------------------------------Channel labels

# channel_names = ['Hoechst', 'JFx650']
# channel_colors = ['blue','red']
# label_x, label_y = 0.01, 0.82
# for j, channel_name in enumerate(channel_names):
#     microim1.ax.text(label_x, label_y - j * 0.08, channel_name, 
#                      color= channel_colors[j], ha='left', transform=microim1.ax.transAxes,
#                      fontsize=fsize)  

channel_names = ['JFx650']
channel_colors = ['#db577b']
label_x, label_y = 0.01, 0.77
for j, channel_name in enumerate(channel_names):
    microim1.ax.text(label_x, label_y - j * 0.08, channel_name, 
                     color= channel_colors[j], ha='left', transform=microim1.ax.transAxes,
                     fontsize=tickfsize) 

#---------------------------------------- dashed contours
#---------------------------------------------------------- dilate masks
def dilate_masks(mask_list, iterations=1):
    return [binary_dilation(mask_list[mask], iterations=iterations) for mask in range(len(mask_list))]

# --- Process cyto masks -------------------------
original_cropped_cyto_mask = copy.deepcopy(cropped_cyto_mask)
cropped_cyto_mask_dil = dilate_masks(cropped_cyto_mask, iterations=1)

linstyl = "--"
color_cont = ['white', '#e69965']

lw = .6
alp= .9
lw_rb = .6
alp_rb = .9

# Draw cyto contours
cyto_contour1 = microim1.ax.contour(cropped_cyto_mask_dil[0], colors=color_cont[0], alpha=alp, linewidths=lw, linestyles=linstyl)
cyto_contour2 = microim2.ax.contour(cropped_cyto_mask_dil[1], colors=color_cont[0], alpha=alp, linewidths=lw, linestyles=linstyl)
cyto_contour3 = microim3.ax.contour(cropped_cyto_mask_dil[2], colors=color_cont[0], alpha=alp, linewidths=lw, linestyles=linstyl)

# original_cropped_cyto_mask = copy.deepcopy(cropped_cyto_mask)
# for masks in range(len(cropped_cyto_mask)):
#     cropped_cyto_mask[masks] = binary_dilation(cropped_cyto_mask[masks], iterations=1)

# # cyto outlines include inner where nucleus is removed
# cyto_contour1 = microim1.ax.contour(cropped_cyto_mask[0], colors=color_cont[0], alpha=alp, linewidths=lw, linestyle=linstyl)
# cyto_contour2 = microim2.ax.contour(cropped_cyto_mask[1], colors=color_cont[0], alpha=alp, linewidths=lw, linestyle=linstyl)
# cyto_contour3 = microim3.ax.contour(cropped_cyto_mask[2], colors=color_cont[0], alpha=alp, linewidths=lw, linestyle=linstyl)


#---------------------------------------- Save
plt.show()

#savename = fig_output+'Figure 1/'+'RelSignalTime_nocolorbar_Timeadj'+figname+'.png'
savename = fig_output+'Figure 1/'+'RelSignalTime_Timeadj'+figname+'.png'
#panel.savefig(savename, bbox_inches = 'tight', dpi=1000)

#%% Relish trace

# Figure setup
fig2size = [2.3,1]  # Figure size
fig, ax = plt.subplots(figsize=fig2size)  # Create a single plot

# Smooth relish data using Savitzky–Golay filter
sg_factor = 5  # Relish data smoothing factor
sg_order = 2  # Polynomial order for smoothing
y_values = df_relish_ratio[cell]  # Get y-values
window = int(len(y_values) / sg_factor)  # Define filter window size
y_sgsmooth = savgol_filter(y_values, window_length=window, polyorder=sg_order)  # Smooth data

# Plot smoothed data
ax.plot(interval_forplt_adj, y_sgsmooth, color='k', linewidth=1.3)

# Set labels and axes limits
ax.set_ylabel('$R_{nuc:tot}$', fontsize=fsize)
ax.set_ylim([0.40, 0.7])
ax.set_xlabel('Time (min)', fontsize=fsize)
ax.set_xticks(ticks =[-60,0, 200, 400, 600, 800], labels=['-60','', '200', '400', '600', '800'])
ax.set_xlim(-70, max(interval_forplt_adj) + 10)
ax.tick_params(axis='both', which='major', labelsize=tickfsize)

# Set x-axis ticks

# Define stimulation time bars
y_min, y_max = ax.get_ylim()
hatch_thick = (y_max - y_min) * 0.07  # Thickness of the hatch bar

# Pre-stimulation hatch (hatched pattern)
prestim_hatch = patches.Rectangle(
    (interval_forplt_adj[0], y_min), 
    60, 
    hatch_thick,
    edgecolor="black", 
    facecolor="none", 
    hatch='///'
)

# Post-stimulation solid bar (filled black)
poststim = patches.Rectangle(
    (interval_forplt_adj[stim], y_min), 
    interval_forplt_adj[-1] - interval_forplt_adj[stim],
    hatch_thick, 
    color="black"
)
# Add stimulation bars to axis
ax.add_patch(poststim)
ax.add_patch(prestim_hatch)

# Add faint vertical lines at specified times
for timemark in times:
    xval = interval_forplt_adj[timemark]
    ax.axvline(xval, color='#db577b', linewidth=0.8, linestyle='dotted')


# Show the plot
plt.show()

# # Save the figure
savename = fig_output + 'Figure 1/' + 'RelTrace_' + figname + '.png'
fig.savefig(savename, bbox_inches='tight', dpi=1000)


#%% Heatmap of cell traces
figsize3= (12, 14)
project = 'attB-Halotag-Relish + Hoechst/' 
rootdir = computer+'Imaging Data/Noshin Imaging Data/'+project

cmap = 'magma'

#import dataset dict_intensities
intensities_df_import = rootdir+ 'AllDatasets_IntensitiesDict/'
with open(os.path.join(intensities_df_import, 'dict_intensities_alldatasets_goodcells_areas_101024'), 'rb') as handle:
    dict_intensities_all = pickle.load(handle)
 

data = []
for date in list(dict_intensities_all.keys()):
    for fname in dict_intensities_all[date].keys():
        for cell, lists in dict_intensities_all[date][fname]['ratio'].items():
            for idx,vals in enumerate(lists):
                condition = fname.split('_')[1]
                rep = fname.split('_')[2]
                time_min = int(vals[0])
                ratio = vals[1]
                data.append([date, condition, rep, cell, time_min, ratio])
df = pd.DataFrame(data, columns=['Date', 'Condition', 'Condition Rep', 'Cell', 'Time', 'Ratio'])

#---------------------------------------- plot heatmap grid

conditions = list(df['Condition'].unique())
cond_titles = ['0 µg/mL', '1 µg/mL', '10 µg/mL', '100 µg/mL']
dates = list(df['Date'].unique())

# Determine the global min and max values for the colormap range
vmin, vmax = df['Ratio'].min(), df['Ratio'].max()

fig, axes = plt.subplots(4, 4, figsize=figsize3, dpi=1000, sharex=True, sharey=False)

# Create an extra axis for the colorbar
cbar_ax = fig.add_axes([0.93, 0.3, 0.02, 0.4])  # [left, bottom, width, height]

for row, date in enumerate(dates):
    for col, condition in enumerate(conditions):
        # Filter data for each date and condition
        df_date = df[df['Date'] == date].copy()
        df_datecond = df_date[df_date['Condition'] == condition].copy()
        
        # Pivot the data for heatmap
        df_plot = df_datecond.pivot_table(index="Cell", columns="Time", values="Ratio")
        df_plot = df_plot.loc[df_plot.max(axis=1).sort_values(ascending=False).index]  # Sort by max "Ratio" per "Cell"

        # Plot the heatmap in the respective subplot
        ax = axes[row, col]
        sns.heatmap(df_plot, cmap=cmap, ax=ax, vmin=vmin, vmax=vmax, 
                    cbar=(row == 0 and col == len(conditions) - 1),
                    cbar_ax=cbar_ax if (row == 0 and col == len(conditions) - 1) else None,
                    cbar_kws={'label': ' r"$R_{nuc:tot}$"'} if (row == 0 and col == len(conditions) - 1) else None)
        df_plot.loc[df_plot.max(axis=1).sort_values(ascending=False).index]  # Sort by max "Ratio" per "Cell"

        # Set titles and labels
        if col == 0:
            ax.set_ylabel('Dataset '+str(row) ,fontsize=fsize)
            ax.set_yticks([])  # Remove y-axis labels for cleaner appearance
        else:
            ax.set(ylabel=None)  # This removes the 'Cell' label from the y-axis
            ax.set_yticks([])  # Remove y-axis labels for cleaner appearance
        
        if row == 0:
            ax.set_title(cond_titles[col],fontsize=fsize)
        
        # After sns.heatmap() for the last row
        if row == len(dates) - 1:  # Check if it's the last row
            # Set tick positions for every 4 ticks
            xticks_positions = range(0, len(interval_forplt_adj), 4)
            ax.set_xticks(xticks_positions)  # Set tick positions
            ax.set_xticklabels([interval_forplt_adj[i] for i in xticks_positions], fontsize=fsize)  # Set tick labels

            ax.set_xlabel("Time (min)", fontsize=fsize)
        else:
            ax.set_xlabel(None)
# Adjust layout for better spacing
#plt.adjust(right=0.9, hspace=0.2, wspace=0.1)


plt.show()
#fig.savefig(fig_output+'Figure 1/DatasetRelishHeatmaps.png', dpi=1200, bbox_inches='tight')

#%% Heatmap just dataset 2 condition 10x

interval_forplt = list(df['Time'].unique())
interval_forplt = interval_forplt_adj


cmap = 'magma'# 'viridis'

# Filter data for the specified date and condition
date = dates[2]
df_date = df[df['Date'] == date].copy()
df_datecond = df_date[df_date['Condition'] == '10X'].copy()

# Pivot the data for heatmap plotting
df_plot = df_datecond.pivot_table(index="Cell", columns="Time", values="Ratio")
df_plot = df_plot.loc[df_plot.max(axis=1).sort_values(ascending=False).index]  # Sort by max "Ratio" per "Cell"

fig, ax = plt.subplots(figsize=(3, 1.35), dpi=1000)

# Plot the heatmap with customized colorbar label
ax= sns.heatmap(df_plot, 
            cmap=cmap, ax=ax, cbar=True, 
            cbar_kws={'label': 'Ratio', 'shrink': .8})

# Access and set colorbar tick label size
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=tickfsize)  # Set colorbar tick labels size
cbar.set_label('$R_{nuc:tot}$', fontsize=tickfsize)  # Set colorbar label font size

# Set labels and title
ax.set_ylabel('Single Cells', fontsize=fsize)
ax.set_yticks([])  # Remove y-axis labels for cleaner appearance
#ax.set_title('+ 10 µg/mL', fontsize=fsize)
ax.set_xlabel("Time (min)", fontsize=fsize)
ax.tick_params(axis='x', labelsize=tickfsize)

# Plot a white vertical line at the stimulation point with annotation

ax.axvline(stim, color='white', linestyle='-', linewidth=1, alpha= 0.5)  
ax.text(2.6,-2.2,  # Adjusted text placement
        '+ (10 µg/mL PGN)', color='k', ha='left', va='center', fontsize=tickfsize)

# Adjust layout for better spacing
plt.subplots_adjust(right=0.92)
# Adjust x-ticks to map to new time labels
xticks_positions = range(0, len(interval_forplt_adj), 4)
ax.set_xticks(xticks_positions)  # Set tick positions
ax.set_xticklabels([interval_forplt_adj[i] for i in xticks_positions], fontsize=tickfsize)  # Set tick labels

plt.show()

fig.savefig(fig_output + 'Figure 1/0410_10x_RelishHeatmap_2_onetone_colorbar.png', dpi=1000, bbox_inches='tight')


#%% Heatmap NONLINEAR TIME just dataset 2 condition 10x
time_points = np.array(interval_forplt_adj)
time_edges = np.zeros(len(time_points) + 1)
time_edges[1:-1] = (time_points[:-1] + time_points[1:]) / 2
time_edges[0] = time_points[0] - (time_points[1] - time_points[0]) / 2
time_edges[-1] = time_points[-1] + (time_points[-1] - time_points[-2]) / 2
time_edges = time_edges[:-1]


df_plot = df_plot.loc[df_plot.max(axis=1).sort_values(ascending=True).index]  # Sort by max "Ratio" per "Cell"
data_matrix = np.array(df_plot)
num_cells = len(data_matrix)

fig, ax = plt.subplots(figsize=(2.7,1.35), dpi=1000)

# Create the heatmap using pcolormesh
c = ax.pcolormesh(time_edges, np.arange(num_cells + 1), data_matrix, cmap='magma', shading='flat')

# # Access and set colorbar tick label size
# cbar = ax.collections[0].colorbar
# cbar.ax.tick_params(labelsize=tickfsize)  # Set colorbar tick labels size
# cbar.set_label('$R_{nuc:tot}$', fontsize=fsize)  # Set colorbar label font size

# Set labels and title
ax.set_ylabel('Single Cells', fontsize=fsize)
ax.set_yticks([])  # Remove y-axis labels for cleaner appearance
#ax.set_title('+ 10 µg/mL', fontsize=fsize)
ax.set_xlabel("Time (min)", fontsize=fsize)
ax.tick_params(axis='x', labelsize=tickfsize)

ax.set_xticks(ticks =[-60,0, 200, 400, 600, 800], labels=['-60','', '200', '400', '600', '800'])

# Plot a white vertical line at the stimulation point with annotation

ax.axvline(0, color='white', linestyle='-', linewidth=1, alpha= 0.5)  
ax.text(-16, 135,  # Adjusted text placement
        '+ (10 µg/mL PGN)', color='k', ha='left', va='center', fontsize=tickfsize)

# Adjust layout for better spacing
plt.subplots_adjust(right=0.92)
plt.show()

#fig.savefig(fig_output + 'Figure 1/0410_10x_RelishHeatmap_nonlintime_onetone.png', dpi=1000, bbox_inches='tight')



