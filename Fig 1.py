# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
import tifffile as tf
import numpy as np
import matplotlib.patches as patches
import os
from skimage.measure import label, regionprops
from scipy.ndimage import binary_dilation
import pandas as pd
import tifffile
import pickle
import time
from datetime import timedelta
import  microfilm.microplot
from microfilm import microplot
from microfilm.microplot import microshow, Micropanel
import seaborn as sns
import copy
from scipy.signal import savgol_filter

#!!! Update path file!!!
gitdir = 'G:/path/' 
#!!! Update path file!!!

files_import = gitdir+'Figure 1 Files/'
fig_output = gitdir+'Temp Output/Fig 1/'

datetime_str = time.strftime("%m%d%y_%H:%M")
resolution=  3.4756 #pixels per micron
units_per_pix = 1/resolution

plt.rcParams['font.family'] = 'Arial'
labelfsize = 12
fsize = 10
tickfsize = 9

#%% Import data for #cell 04_Dpt_10X_1_Cell 76------------------------------------------------------------------------------------------------------------------------------------

#set dataset parameters
stim=4 #first slice after inject -1!!!
nuc_channel = 0
rel_channel = 1
interval_forplt = np.concatenate([np.arange(0, 121, 15) , np.arange(150, 631, 30), np.arange(690, 971, 60)])
interval_forplt_adj = interval_forplt-interval_forplt[stim].tolist()
interval_hrs = [str(timedelta(minutes=int(time.item())))[:-3] for time in interval_forplt]

day = '2024-10-02'
fname='04_Dpt_10X_1_maxZ'
celln= 76
cell = 'Cell 76'

with open(os.path.join(files_import, 'dict_intensities_goodcells_peaks_size2-150_2400_ratio2,3'), 'rb') as handle:
    dict_intensities = pickle.load(handle)
f_dict_intensities = dict_intensities[fname+'.tif']
df_relish_ratio = pd.DataFrame({cell: [intensity for time, intensity in values] for cell, values in f_dict_intensities['relish_ratio'].items()})

time_series_tif = tf.imread(files_import+fname+'.tif')
cyto_mask_tif = tf.imread(files_import+fname+'_Cyto Matched Mask.tif')
nuc_mask_tif = tf.imread(files_import+fname+'_Nuc Matched Mask.tif')
nframes = time_series_tif.shape[0]
    
#%% Fig 1C time-course cell panel------------------------------------------------------------------------------------------------------------------------------------

figsize1 = [2.5,1.9]
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

channel_names = ['JFx650']
channel_colors = ['#db577b']
label_x, label_y = 0.01, 0.77
for j, channel_name in enumerate(channel_names):
    microim1.ax.text(label_x, label_y - j * 0.08, channel_name, 
                     color= channel_colors[j], ha='left', transform=microim1.ax.transAxes,
                     fontsize=tickfsize) 

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

#---------------------------------------- Save
plt.show()

figname = day+"_"+fname[3:-9]+"_"+cell
savename = fig_output+'Fig1C_'+figname+'.png'
panel.savefig(savename, bbox_inches = 'tight', dpi=1000)

#%% Fig 1D Relish trace------------------------------------------------------------------------------------------------------------------------------------

figsize2 = [2.3,1]  # Figure size
fig, ax = plt.subplots(figsize=figsize2)  # Create a single plot

#---------------------------------------- Smooth relish data using Savitzky–Golay filter
sg_factor = 5  # Relish data smoothing factor
sg_order = 2  # Polynomial order for smoothing
y_values = df_relish_ratio[cell]  # Get y-values
window = int(len(y_values) / sg_factor)  # Define filter window size
y_sgsmooth = savgol_filter(y_values, window_length=window, polyorder=sg_order)  # Smooth data

#----------------------------------------  Plot smoothed data
ax.plot(interval_forplt_adj, y_sgsmooth, color='k', linewidth=1.3)
# Set labels and axes limits
ax.set_ylabel('$R_{nuc:tot}$', fontsize=fsize)
ax.set_ylim([0.40, 0.7])
ax.set_xlabel('Time (min)', fontsize=fsize)
ax.set_xticks(ticks =[-60,0, 200, 400, 600, 800], labels=['-60','', '200', '400', '600', '800'])
ax.set_xlim(-70, max(interval_forplt_adj) + 10)
ax.tick_params(axis='both', which='major', labelsize=tickfsize)


#----------------------------------------  Define stimulation time bars
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

#----------------------------------------  Add faint vertical lines at specified times
for timemark in times:
    xval = interval_forplt_adj[timemark]
    ax.axvline(xval, color='#db577b', linewidth=0.8, linestyle='dotted')


#---------------------------------------- Save
plt.show()

figname = day+"_"+fname[3:-9]+"_"+cell+'_Trace'
savename = fig_output+'Fig1D_'+figname+'.png'
fig.savefig(savename, bbox_inches = 'tight', dpi=1000)


#%% Fig 1E Heatmap of cell traces------------------------------------------------------------------------------------------------------------------------------------
figsize3= (12, 14)

#----------------------------------------  import and sort dataset dict_intensities
with open(os.path.join(files_import, 'dict_intensities_alldatasets_goodcells_areas_101024'), 'rb') as handle:
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

#---------------------------------------- plot heatmap grid only for one dataset

conditions = list(df['Condition'].unique())
cond_titles = ['0 µg/mL', '1 µg/mL', '10 µg/mL', '100 µg/mL']
dates = list(df['Date'].unique())

# Filter data for the specified date and condition
date = dates[2]
df_date = df[df['Date'] == date].copy()
df_datecond = df_date[df_date['Condition'] == '10X'].copy()

# Pivot the data for heatmap plotting
df_plot = df_datecond.pivot_table(index="Cell", columns="Time", values="Ratio")
df_plot = df_plot.loc[df_plot.max(axis=1).sort_values(ascending=True).index]  # Sort by max "Ratio" per "Cell"
data_matrix = np.array(df_plot)
num_cells = len(data_matrix)

# Determine the global min and max values for the colormap range
vmin, vmax = df['Ratio'].min(), df['Ratio'].max()

# Set up non-linear time edges
time_points = np.array(interval_forplt_adj)
time_edges = np.zeros(len(time_points) + 1)
time_edges[1:-1] = (time_points[:-1] + time_points[1:]) / 2
time_edges[0] = time_points[0] - (time_points[1] - time_points[0]) / 2
time_edges[-1] = time_points[-1] + (time_points[-1] - time_points[-2]) / 2
time_edges = time_edges[:-1]

cmap = 'magma'
fig, ax = plt.subplots(figsize=(2.7,1.35), dpi=1000)
# Create the heatmap using pcolormesh
c = ax.pcolormesh(time_edges, np.arange(num_cells + 1), data_matrix, cmap='magma', shading='flat')

# Set labels and title
ax.set_ylabel('Single Cells', fontsize=fsize)
ax.set_yticks([])
ax.set_xlabel("Time (min)", fontsize=fsize)
ax.tick_params(axis='x', labelsize=tickfsize)
ax.set_xticks(ticks =[-60,0, 200, 400, 600, 800], labels=['-60','', '200', '400', '600', '800'])

# Plot a white vertical line at the stimulation point with annotation
ax.axvline(0, color='white', linestyle='-', linewidth=1, alpha= 0.5)  
ax.text(-16, 85,  # Adjusted text placement
        '+ (10 µg/mL PGN)', color='k', ha='left', va='center', fontsize=tickfsize)

# Adjust layout for better spacing
plt.subplots_adjust(right=0.92)
plt.show()

#---------------------------------------- Save
figname = '0410_10X_RelishHeatmap_nonlintime'
savename = fig_output+'Fig1E_'+figname+'.png'
fig.savefig(savename, bbox_inches = 'tight', dpi=1000)
