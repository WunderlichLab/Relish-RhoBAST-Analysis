# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 15:59:30 2025

@author: noshin
"""

import pandas as pd
import tifffile as tf
from tifffile import imread
from tifffile import imshow
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from scipy.signal import savgol_filter
from skimage.measure import label, regionprops
from scipy.ndimage import binary_dilation
from scipy.signal import savgol_filter
from skimage.segmentation import mark_boundaries
from skimage.feature.peak import peak_local_max
from PIL import Image
import pickle
import  microfilm.microplot
from microfilm import microplot
from microfilm.microplot import microshow, Micropanel
import gc
import copy
import time
from datetime import timedelta
datetime_str = time.strftime("%m%d%y_%H:%M")

#!!! Update path file!!!
gitdir = 'G:/path/' 
#!!! Update path file!!!

files_import = gitdir+'Figure 3 Files/'
fig_output = gitdir+'Temp Output/Fig 3/'

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

#%% Fig 3B Time diagram

# Full imaging period in minutes
full_start = -60
full_end = 930
# Original time points (in minutes)
interval_forplt = np.concatenate([
    np.arange(0, 121, 15),
    np.arange(150, 631, 30),
    np.arange(690, 991, 60)
])
offset = 60
interval_forplt_offset = interval_forplt-offset

time_points = list(interval_forplt_offset)
stim = 4

# Define the three high-resolution windows (each 2 hours = 120 min)
# Window 1: from 60 to 180 min
window1_start, window1_end = 0, 120
center1 = (window1_start + window1_end) / 2  # 120
# Window 2: from 465 to 585 min
window2_start, window2_end = 405, 525
center2 = (window2_start + window2_end) / 2  # 525
# Window 3: from 870 to 990 min
window3_start, window3_end = 810, 930
center3 = (window3_start + window3_end) / 2  # 930

fig, ax = plt.subplots(figsize=(2.25, 1.2),dpi=1000)

# Set axis limits and labels
y_min, y_max = 0, 0.8
ax.set_ylim(y_min, y_max)
ax.set_xlim(full_start, full_end)
ax.set_xlabel('Time (min)', fontsize=fsize)
ax.set_yticks([])
ax.set_xticks([-60, 0, 200, 400, 600, 800], labels=['-60','','200','','600',''], fontsize=tickfsize)

ax.tick_params(axis='x', which='both', pad=.01)

# Calculate center of y-axis
center_y = (y_min + y_max) / 2

# Draw vertical lines at every original time point (background lines)
for t in time_points:
    ax.plot([t, t], [y_min, y_max], color='lightgray', linestyle='-', 
            linewidth=0.9, alpha=1, zorder=0)

# --- X-axis hatch (drawn at the bottom)
hatch_thickness = 0.07 * (y_max - y_min)
prestim  = patches.Rectangle((-60, 0), 60, hatch_thickness, edgecolor="black", facecolor="none", hatch='///')
hatch = patches.Rectangle((0, 0), interval_forplt_offset[-1], hatch_thickness, color= 'black')
ax.add_patch(prestim)
ax.add_patch(hatch)

# --- Shaded rectangles for high-resolution windows
patch_height = 0.4                   # height of the rectangle
patch_y = center_y - patch_height/2  # y coordinate for bottom of rectangle

rect1 = patches.Rectangle((window1_start, patch_y), window1_end - window1_start, patch_height,
                           color='gray', alpha=1, zorder=3)
rect2 = patches.Rectangle((window2_start, patch_y), window2_end - window2_start, patch_height,
                           color='gray', alpha=1, zorder=3)
rect3 = patches.Rectangle((window3_start, patch_y), window3_end - window3_start, patch_height,
                           color='gray', alpha=1, zorder=3)
ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)

# --- 30-second (0.5 min) lines drawn inside each rectangle
# These lines span from the rectangle's bottom (patch_y) to its top (patch_y+patch_height)
for t in np.arange(0, 120, 3):
    ax.plot([window1_start + t, window1_start + t], [patch_y, patch_y + patch_height],
            color='white', linestyle='-', linewidth=0.2, alpha=1, zorder=4)
    ax.plot([window2_start + t, window2_start + t], [patch_y, patch_y + patch_height],
            color='white', linestyle='-', linewidth=0.2, alpha=1, zorder=4)
    ax.plot([window3_start + t, window3_start + t], [patch_y, patch_y + patch_height],
            color='white', linestyle='-', linewidth=0.2, alpha=1, zorder=4)

# --- Place centered text labels in the middle of each rectangle
ax.text(center1, center_y, 'W1', ha='center', va='center', 
        color='k', fontsize=tickfsize, zorder=5)
ax.text(center2, center_y, 'W2', ha='center', va='center', 
        color='k', fontsize=tickfsize, zorder=5)
ax.text(center3, center_y, 'W3', ha='center', va='center', 
        color='k', fontsize=tickfsize, zorder=5)

# --- Stimulus injection line and label (placed near the center of y)
stim_time = interval_forplt_offset[stim]
ax.axvline(stim_time, color='k', linestyle='--', linewidth=1, zorder=4)
ax.text(stim_time+78, center_y+.5, '+ PGN', ha='center', va='center', 
        color='k', fontsize=tickfsize, zorder=5)

plt.tight_layout()
plt.show()

#---------------------------------------- Save
figname = 'TimeDiagram'
savename = fig_output+'Fig 3B_'+figname+'.png'
fig.savefig(savename, bbox_inches = 'tight', dpi=1000)


#%% Fig 3C: Sparse imaging (Import data for 07_Dpt_100X_1_Cell 92)

stim        = 4 #first slice after inject -1!!!
nuc_channel = 0
rel_channel = 1
rb_channel  = 2
interval_forplt = np.concatenate([np.arange(0, 121, 15) , np.arange(150, 631, 30), np.arange(690, 991, 60)])
interval_hrs = [str(timedelta(minutes=int(time.item())))[:-3] for time in interval_forplt]

day = '2024-10-23'
fname='07_Dpt_100X_1_maxZ'
celln= 92
cell = 'Cell 92'

with open(os.path.join(files_import, 'Sparse Imaging', 'dict_intensities_ilastikpeaks_meansum_alldatasets_goodcells_nomasks_010925'), 'rb') as handle:
    dict_intensities = pickle.load(handle)
    
f_dict_intensities = dict_intensities[day][fname+'.tif']
df_relish_ratio = pd.DataFrame({cell: [intensity for time, intensity in values] for cell, values in f_dict_intensities['relish_ratio'].items()})
df_rb = pd.DataFrame({cell: [intensity for time, intensity in values] for cell, values in f_dict_intensities['peak_intensities_sum'].items()})

time_series_tif     = tf.imread(files_import + 'Sparse Imaging/' + fname+'.tif')
cyto_mask_tif       = tf.imread(files_import + 'Sparse Imaging/' + fname+'_Cyto Matched Mask.tif')
nuc_mask_tif        = tf.imread(files_import + 'Sparse Imaging/' + fname+'_Nuc Matched Mask.tif')
rb_mask_tif         = tf.imread(files_import + 'Sparse Imaging/' + fname+'_Object Identities.tiff')

nframes = time_series_tif.shape[0]
    
    
#%% Fig 3C: Sparse imaging (time-course cell panels)
figsize1 = [2.8,2]

times = [0,8,16,29]
colorb = False #toggle whether to display colorbar
figname_file = f"{day}_{fname}_{cell}"

#---------------------------------------------------------- Set min and max coordinates for cell
xmin, xmax, ymin, ymax = float('inf'), float('-inf'), float('inf'), float('-inf')
xmin_n, xmax_n, ymin_n, ymax_n = float('inf'), float('-inf'), float('inf'), float('-inf')

bufferwhole=1
for t in times:
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
        
        xmin_n = min(xmin_n, minx_n)
        xmax_n = max(xmax_n, maxx_n)
        ymin_n = min(ymin_n, miny_n)
        ymax_n = max(ymax_n,maxy_n)  
    
#---------------------------------------------------------- Set cell and outline parameters
xmin = xmin+5
xmax = xmax-8

ymin = ymin-30
ymax = ymax+15

cropped_cell_rel,       cropped_cell_rb = {},{}
cropped_cyto_mask,      cropped_nuc_mask,       cropped_rb_mask = {},{},{}
cropped_cell_rb_zoom,   cropped_nuc_mask_zoom,  cropped_rb_mask_zoom = {},{},{}
mins = {}

buffer = 22

for count,t in enumerate(times):
#   cropped_cell[count] = time_series_tif[t,[nuc_channel,rel_channel], ymin:ymax, xmin:xmax]
    cropped_cell_rel[count] = time_series_tif[t, rel_channel, ymin:ymax, xmin:xmax]
    cropped_cell_rb[count] = time_series_tif[t, rb_channel, ymin:ymax, xmin:xmax]

    cyto_mask = (cyto_mask_tif[t] == int(celln)).astype(int)
    nuc_mask = (nuc_mask_tif[t] == int(celln)).astype(int) 
    rb_mask = (rb_mask_tif[t]) 

    cropped_cyto_mask[count] = cyto_mask[ymin:ymax, xmin:xmax]
    cropped_nuc_mask[count] = nuc_mask[ymin:ymax, xmin:xmax]
    cropped_rb_mask[count] = rb_mask[ymin:ymax, xmin:xmax]
 
    
    buff_scale_max = int(ymax-buffer*1.6)
    buff_scale_min = int(ymin+buffer*1.7)
    cropped_cell_rb_zoom[count] = time_series_tif[t, rb_channel, buff_scale_min:buff_scale_max, xmin+buffer:xmax-buffer]
    cropped_nuc_mask_zoom[count] = nuc_mask[buff_scale_min:buff_scale_max, xmin+buffer:xmax-buffer]
    cropped_rb_mask_zoom[count] = rb_mask[buff_scale_min:buff_scale_max, xmin+buffer:xmax-buffer]

    mins[count] = str(interval_forplt[t]-60)# + ' min'

#---------------------------------------------------------- Microplot 
fig, ax = plt.subplots(dpi=1000)

colormaps_rel = ['magma']
minmax = [0,4500]

microim1 = microshow(cropped_cell_rel[0], cmaps=colormaps_rel, label_text=mins[0], label_font_size=fsize, rescale_type='limits', limits= minmax)
microim2 = microshow(cropped_cell_rel[1], cmaps=colormaps_rel, label_text=mins[1], label_font_size=fsize, rescale_type='limits', limits= minmax)
microim3 = microshow(cropped_cell_rel[2], cmaps=colormaps_rel, label_text=mins[2], label_font_size=fsize, rescale_type='limits', limits= minmax)
microim4 = microshow(cropped_cell_rel[3], cmaps=colormaps_rel, label_text=mins[3], label_font_size=fsize, rescale_type='limits', limits= minmax,
                     unit='um', scalebar_unit_per_pix=units_per_pix, scalebar_size_in_units=5, 
                    scalebar_color='white',scalebar_font_size= None, show_colorbar=colorb)
cmap_rb = ['magma']#['copper']
minmax = [0, 4500]
microim5 = microshow(cropped_cell_rb_zoom[0], cmaps=cmap_rb, rescale_type='limits', limits= minmax)
microim6 = microshow(cropped_cell_rb_zoom[1], cmaps=cmap_rb, rescale_type='limits', limits= minmax)
microim7 = microshow(cropped_cell_rb_zoom[2], cmaps=cmap_rb, rescale_type='limits', limits= minmax)
microim8 = microshow(cropped_cell_rb_zoom[3], cmaps=cmap_rb, rescale_type='limits', limits= minmax,
                                         unit='um', scalebar_unit_per_pix=units_per_pix, scalebar_size_in_units=5, 
                                        scalebar_color='white', scalebar_font_size= None, show_colorbar=colorb)

panel = Micropanel(rows=2, cols=4, figsize=figsize1) 
panel.add_element([0, 0], microim1)
panel.add_element([0, 1], microim2)
panel.add_element([0, 2], microim3)
panel.add_element([0, 3], microim4)

panel.add_element([1, 0], microim5)
panel.add_element([1, 1], microim6)
panel.add_element([1, 2], microim7)
panel.add_element([1, 3], microim8)

#---------------------------------------------------------- Channel labels
# channel_names = ['JFx650']
# channel_colors = ['#db577b']
# label_x, label_y = 0.01, 0.05
# for j, channel_name in enumerate(channel_names):
#     microim1.ax.text(label_x, label_y - j * 0.08, channel_name, 
#                      color= channel_colors[j], ha='left', transform=microim1.ax.transAxes,
#                      fontsize=tickfsize)  
# channel_names = ['SpyRHO555']
# channel_colors = ['#e69965']
# label_x, label_y = 0.01, 0.05
# for j, channel_name in enumerate(channel_names):
#     microim5.ax.text(label_x, label_y - j * 0.08, channel_name, 
#                      color= channel_colors[j], ha='left', transform=microim5.ax.transAxes,
#                      fontsize=tickfsize) 

#---------------------------------------------------------- dilate masks
def dilate_masks(mask_list, iterations=1):
    return [binary_dilation(mask_list[mask], iterations=iterations) for mask in range(len(mask_list))]

# --- Process cyto masks -------------------------
original_cropped_cyto_mask = copy.deepcopy(cropped_cyto_mask)
cropped_cyto_mask_dil = dilate_masks(cropped_cyto_mask, iterations=1)
# --- Process rb and nuc masks -------------------
original_cropped_rb_mask = copy.deepcopy(cropped_rb_mask_zoom)
cropped_rb_mask_zoom_dil = dilate_masks(cropped_rb_mask_zoom, iterations=1)

original_cropped_nuc_mask = copy.deepcopy(cropped_nuc_mask_zoom)
cropped_nuc_mask_zoom_dil = dilate_masks(cropped_nuc_mask_zoom, iterations=1)

#---------------------------------------------------------- dashed contours
linstyl = "--"
color_cont = ['white', '#e69965']

lw = .4
alp= .9
lw_rb = .4
alp_rb = .9

# Draw cyto contours
cyto_contour1 = microim1.ax.contour(cropped_cyto_mask_dil[0], colors=color_cont[0], alpha=alp, linewidths=lw, linestyles=linstyl)
cyto_contour2 = microim2.ax.contour(cropped_cyto_mask_dil[1], colors=color_cont[0], alpha=alp, linewidths=lw, linestyles=linstyl)
cyto_contour3 = microim3.ax.contour(cropped_cyto_mask_dil[2], colors=color_cont[0], alpha=alp, linewidths=lw, linestyles=linstyl)
cyto_contour4 = microim4.ax.contour(cropped_cyto_mask_dil[3], colors=color_cont[0], alpha=alp, linewidths=lw, linestyles=linstyl)

# Draw rb and nuc contours 

rb_contour2 = microim6.ax.contour(cropped_rb_mask_zoom_dil[1], colors=color_cont[1], alpha=alp_rb, linewidths=lw_rb, linestyles=linstyl)
rb_contour3 = microim7.ax.contour(cropped_rb_mask_zoom_dil[2], colors=color_cont[1], alpha=alp_rb, linewidths=lw_rb, linestyles=linstyl)
rb_contour4 = microim8.ax.contour(cropped_rb_mask_zoom_dil[3], colors=color_cont[1], alpha=alp_rb, linewidths=lw_rb, linestyles=linstyl)

nuc_contour5= microim5.ax.contour(cropped_nuc_mask_zoom_dil[0], colors=color_cont[0], alpha=alp, linewidths=lw, linestyles=linstyl)
nuc_contour6= microim6.ax.contour(cropped_nuc_mask_zoom_dil[1], colors=color_cont[0], alpha=alp, linewidths=lw, linestyles=linstyl)
nuc_contour7 = microim7.ax.contour(cropped_nuc_mask_zoom_dil[2], colors=color_cont[0], alpha=alp, linewidths=lw, linestyles=linstyl)
nuc_contour8 = microim8.ax.contour(cropped_nuc_mask_zoom_dil[3], colors=color_cont[0], alpha=alp, linewidths=lw, linestyles=linstyl)

#plt.show()
#---------------------------------------------------------- Save
savename = ['Fig 3C_colorbar_'+figname_file+'.png' if colorb else 'Fig 3C_'+figname_file+'.png']
savename = fig_output+savename[0]
panel.savefig(savename, bbox_inches = 'tight', dpi=1000)

#%% Fig 3D  Sparse imaging (Relish and rhobast trace)
figsize2 = [3, 1.2]  # Adjusted the figure size for two panels
fig, axs = plt.subplots(2, 1, figsize=figsize2, sharex=True)  # Create a 2-panel plot

#---------------------------------------------------------- smooth relish data
y_values = df_relish_ratio[cell]
window   = int(len(y_values)/sg_factor_rel) #length of the filter window
y_sgsmooth = savgol_filter(y_values, window_length = window, polyorder = sg_order_rel)
    
#---------------------------------------------------------- Top panel: df_relish_ratio[cell] vs. time
axs[0].plot(interval_forplt_offset, y_sgsmooth, marker=None, color='k',linewidth=1)

axs[0].set_ylabel('$R_{nuc:tot}$', fontsize=fsize)
axs[0].set_ylim([.55, .75])
axs[0].set_xlim([interval_forplt_offset[0], interval_forplt_offset[-1]])
axs[0].set_xticks(ticks = [-30, 0, 200,400,600,800], labels=[])

axs[0].tick_params(axis='both', which='major', labelsize=tickfsize)
#set stim time bar
y_min, y_max = axs[0].get_ylim()
hatchthick = (y_max-y_min)*0.07
prestim  = patches.Rectangle((-60, y_min), 60, hatchthick, edgecolor="black", facecolor="none", hatch='///')
poststim_hatch = patches.Rectangle((interval_forplt_offset[stim], y_min), interval_forplt_offset[-1] - interval_forplt_offset[stim], 
                                   hatchthick, color = "black")
axs[0].add_patch(prestim)
axs[0].add_patch(poststim_hatch)

#---------------------------------------------------------- Bottom panel: df_rb[cell] vs. time with markers for positive intensity
y_values_rb = df_rb[cell]
positive_intensity = y_values_rb > 0

# Plot positive intensity values
axs[1].scatter(interval_forplt_offset[positive_intensity], (y_values_rb[positive_intensity])/1000, color='k', marker='o', s=8)  
axs[1].set_ylabel('Int$_R$$_B$ (AU)', fontsize=fsize)
axs[1].text(-165, 27, 'x10$^3$', color='black', ha='left', va='top', fontsize=tickfsize)

#set stim time bar
axs[1].set_ylim([-1, max(y_values_rb)/1000 * 1.1])  # Add some padding to avoid clipping
axs[1].set_xlim([interval_forplt_offset[0]-5, interval_forplt_offset[-1]])
axs[1].set_yticks([0, 15])  # Add some padding to avoid clipping
axs[1].set_xticks(ticks = [-60, 0, 200,400,600,800], labels=['-60','','','400','', '800'])
axs[1].tick_params(axis = "both", labelsize = tickfsize)

y_min, y_max = axs[1].get_ylim()
hatchthick = (y_max-y_min)*0.07
prestim  = patches.Rectangle((-60, y_min), 60, hatchthick, edgecolor="black", facecolor="none", hatch='///')
poststim_hatch = patches.Rectangle((interval_forplt_offset[stim], y_min), interval_forplt_offset[-1] - interval_forplt_offset[stim], 
                                   hatchthick, color = "black",)
axs[1].add_patch(prestim)
axs[1].add_patch(poststim_hatch)

axs[1].set_xlabel('Time (min)', fontsize=fsize)

#----------------------------------------------------------  Add timemarks
# Adding selected time marks
for timemark in times:
    xval = interval_forplt[timemark]-offset
    axs[0].plot(xval, y_sgsmooth[timemark], marker='o', markersize=4, color='#db577b', alpha=0.9)
    axs[1].plot(xval, (y_values_rb[timemark]/1000), marker='o', markersize=4, color='#c25a13', alpha=0.9, mfc='none')
    #e69965

# Adding time marks
for time in time_points:
    axs[0].axvline(x=time, color='gray', linestyle='-', linewidth = 0.4, alpha=0.4, zorder=0)
    axs[1].axvline(x=time, color='gray', linestyle='-', linewidth = 0.4, alpha=0.4, zorder=0)
    #e69965
    
for ax in axs.flat:  # Works for both 1D and 2D subplot arrays
    ax.yaxis.set_label_coords(-.1, 0.5)
# Adjust layout to prevent overlap
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.01,  hspace=0.12)

plt.show()

#---------------------------------------------------------- Save
savename = fig_output+'Fig 3D_SparseTraces_'+figname_file+'.png' 
fig.savefig(savename, bbox_inches = 'tight', dpi=1000)

#%% Fig 3E: Dense imaging (import data for 2025-02-26 DptMtk_Dense_3time_Cell 147)
stim        = 4 #first slice after inject -1!!!
nuc_channel = 0
rel_channel = 1
rb_channel  = 2

full_time = np.concatenate([
    np.arange(0, 120.1, 0.5),
    np.arange(405,525.1,0.5),
    np.arange(810,930.1,0.5)
]).tolist()

day = '2025-02-26'
celln= 147
cell = 'Cell 147'

fname_root = 'Dpt_100X_T'
# Create dictionaries for each type of image
time_series_tifs = {}
cyto_mask_tifs = {}
nuc_mask_tifs = {}
rb_mask_tifs = {}

for timewin in [1, 2, 3]:
    fname = fname_root + str(timewin)
    time_series_tifs[f'T{timewin}']     = tf.imread(files_import + 'Dense Imaging/' + fname+'.tif')
    cyto_mask_tifs[f'T{timewin}']       = tf.imread(files_import + 'Dense Imaging/' + fname+'_Cyto Matched Mask.tif')
    nuc_mask_tifs[f'T{timewin}']        = tf.imread(files_import + 'Dense Imaging/' + fname+'_Nuc Matched Mask.tif')
    rb_mask_tifs[f'T{timewin}']         = tf.imread(files_import + 'Dense Imaging/' + fname+'_Object Identities.tif')

nframes = time_series_tifs['T1'].shape[0]    
    
with open(os.path.join(files_import, 'Dense Imaging', 'cellsmultiwindows_dataframe_peakproperties_alldays_prom300_relativheight0.9_032025.pkl'), 'rb') as handle:
    peaks_df_cellsmultiplewindows = pickle.load(handle)
    
cell_id = 'Dpt-20250226-147'
cell147_multitime = peaks_df_cellsmultiplewindows[peaks_df_cellsmultiplewindows['ID']==cell_id]

df_relish_ratio = pd.DataFrame()
df_rb = pd.DataFrame()
df_times = pd.DataFrame()

for index, row in cell147_multitime.iterrows():
    column_name = row['Time Window']
    column_rel_values = row['Smoothed Relish Ratio']
    column_rb_values = row['Smoothed RhoBAST Intensity']
    column_times_values = row['Time (min)']
    
    # Add a new column to df_relish_ratio_
    df_relish_ratio[column_name] = column_rel_values
    df_rb[column_name] = column_rb_values
    df_times[column_name] = column_times_values

#%% Fig 3E: Dense imaging (time-course cell panels)
figsize3 =  [2.65,1.9]
colorb= True
figname_file = f"{day}_{fname}_{cell}"

#---------------------------------------------------------- Set min and max coordinates for cell
xmin, xmax, ymin, ymax = float('inf'), float('-inf'), float('inf'), float('-inf')

for timewin in [1, 2, 3]:
    for t in range(nframes):
        cyto_mask = (cyto_mask_tifs[f'T{timewin}'][t] == int(celln)).astype(int)
        nuc_mask = (nuc_mask_tifs[f'T{timewin}'][t] == int(celln)).astype(int)
        # Define the limits, crop image and contours

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
#---------------------------------------------------------- Set cell and outline parameters
xmin=0
xmin_n=0
xmax = xmax 

times = [0, 154, 155, 72]
timewin = ['T1', 'T1', 'T2', 'T3']
cropped_cell_rel,       cropped_cell_rb = {},{}
cropped_cyto_mask,      cropped_nuc_mask,       cropped_rb_mask = {},{},{}
cropped_cell_rb_zoom,   cropped_nuc_mask_zoom,  cropped_rb_mask_zoom = {},{},{}
mins = {}

buffer = 10
scale= int(buffer*2.7) 
buffer_x= 20 #increase to zoom more in x
for count, frame in enumerate(times):
    timewin_time = timewin[count]

    cropped_cell_rel[count] =       time_series_tifs[timewin_time][frame, rel_channel, ymin:ymax, xmin:xmax]
    cropped_cell_rb[count] =        time_series_tifs[timewin_time][frame, rb_channel, ymin:ymax, xmin:xmax]
    cropped_cell_rb_zoom[count] =   time_series_tifs[timewin_time][frame, rb_channel, ymin+scale:ymax-scale, xmin+buffer_x:xmax-buffer_x]

    cyto_mask = (cyto_mask_tifs[timewin_time][frame] == int(celln)).astype(int)
    nuc_mask = (nuc_mask_tifs[timewin_time][frame] == int(celln)).astype(int) 
    rb_mask = (rb_mask_tifs[timewin_time][frame]) 

    cropped_cyto_mask[count] = cyto_mask[ymin:ymax, xmin:xmax]
    cropped_nuc_mask[count] = nuc_mask[ymin:ymax, xmin:xmax]
    cropped_rb_mask[count] = rb_mask[ymin:ymax, xmin:xmax]
    
    cropped_nuc_mask_zoom[count] = nuc_mask[ymin+scale :ymax-scale, xmin+buffer_x:xmax-buffer_x]
    cropped_rb_mask_zoom[count] = rb_mask[ymin+scale:ymax-scale, xmin+buffer_x:xmax-buffer_x]

    mins[count] = str(df_times.loc[frame][timewin_time])#+ ' min'

#---------------------------------------------------------- Microplot -----------START RUN HERE AFTER FIRST TIME

fig, ax = plt.subplots(dpi=1000)

colormaps_rel = ['magma']
minmax = [40, 1600]
microim1 = microshow(cropped_cell_rel[0], cmaps=colormaps_rel, label_text=mins[0], label_font_size=fsize, rescale_type='limits', limits= minmax)
microim2 = microshow(cropped_cell_rel[1], cmaps=colormaps_rel, label_text=mins[1], label_font_size=fsize, rescale_type='limits', limits= minmax)
microim3 = microshow(cropped_cell_rel[2], cmaps=colormaps_rel, label_text=mins[2], label_font_size=fsize, rescale_type='limits', limits= minmax)
microim4 = microshow(cropped_cell_rel[3], cmaps=colormaps_rel, label_text=mins[3], label_font_size=fsize, rescale_type='limits', limits= minmax,
                     unit='um', scalebar_unit_per_pix=units_per_pix, scalebar_size_in_units=5, 
                    scalebar_color='white',scalebar_font_size=None,
                    show_colorbar=colorb)
cmap_rb = ['magma']
minmax = [20, 3000]
microim5 = microshow(cropped_cell_rb_zoom[0], cmaps=cmap_rb, rescale_type='limits', limits= minmax)
microim6 = microshow(cropped_cell_rb_zoom[1], cmaps=cmap_rb, rescale_type='limits', limits= minmax)
microim7 = microshow(cropped_cell_rb_zoom[2], cmaps=cmap_rb, rescale_type='limits', limits= minmax)
microim8 = microshow(cropped_cell_rb_zoom[3], cmaps=cmap_rb, rescale_type='limits', limits= minmax,
                                         unit='um', scalebar_unit_per_pix=units_per_pix, scalebar_size_in_units=5,
                                        scalebar_color='white', scalebar_font_size=tickfsize,
                                        show_colorbar=colorb)
 
panel = Micropanel(rows=2, cols=4, figsize=figsize3) #figscaling=2.5)  
panel.add_element([0, 0], microim1)
panel.add_element([0, 1], microim2)
panel.add_element([0, 2], microim3)
panel.add_element([0, 3], microim4)

panel.add_element([1, 0], microim5)
panel.add_element([1, 1], microim6)
panel.add_element([1, 2], microim7)
panel.add_element([1, 3], microim8)

#---------------------------------------------------------- Channel labels
# channel_names = ['JFx650']
# channel_colors = ['#db577b']
# label_x, label_y = 0.01, 0.05
# for j, channel_name in enumerate(channel_names):
#     microim1.ax.text(label_x, label_y - j * 0.08, channel_name, 
#                      color= channel_colors[j], ha='left', transform=microim1.ax.transAxes,
#                      fontsize=tickfsize)  
# channel_names = ['SpyRHO555']
# channel_colors = ['#e69965']
# label_x, label_y = 0.01, 0.05
# for j, channel_name in enumerate(channel_names):
#     microim5.ax.text(label_x, label_y - j * 0.08, channel_name, 
#                      color= channel_colors[j], ha='left', transform=microim5.ax.transAxes,
#                      fontsize=tickfsize) 
#---------------------------------------------------------- dilate masks
# --- Process cyto masks -------------------------
original_cropped_cyto_mask = copy.deepcopy(cropped_cyto_mask)
cropped_cyto_mask_dil = dilate_masks(cropped_cyto_mask, iterations=1)
# --- Process rb and nuc masks -------------------
original_cropped_rb_mask = copy.deepcopy(cropped_rb_mask_zoom)
cropped_rb_mask_zoom_dil = dilate_masks(cropped_rb_mask_zoom, iterations=1)

original_cropped_nuc_mask = copy.deepcopy(cropped_nuc_mask_zoom)
cropped_nuc_mask_zoom_dil = dilate_masks(cropped_nuc_mask_zoom, iterations=1)

#---------------------------------------------------------- dashed contours
linstyl = "--"
color_cont = ['white', '#e69965']

lw = .4
alp= .9
lw_rb = .4
alp_rb = .9

# Draw cyto contours
cyto_contour1 = microim1.ax.contour(cropped_cyto_mask_dil[0], colors=color_cont[0], alpha=alp, linewidths=lw, linestyles=linstyl)
cyto_contour2 = microim2.ax.contour(cropped_cyto_mask_dil[1], colors=color_cont[0], alpha=alp, linewidths=lw, linestyles=linstyl)
cyto_contour3 = microim3.ax.contour(cropped_cyto_mask_dil[2], colors=color_cont[0], alpha=alp, linewidths=lw, linestyles=linstyl)
cyto_contour4 = microim4.ax.contour(cropped_cyto_mask_dil[3], colors=color_cont[0], alpha=alp, linewidths=lw, linestyles=linstyl)

# Draw rb and nuc contours 
rb_contour1 = microim5.ax.contour(cropped_rb_mask_zoom_dil[0], colors=color_cont[1], alpha=alp_rb, linewidths=lw_rb, linestyles=linstyl)
rb_contour2 = microim6.ax.contour(cropped_rb_mask_zoom_dil[1], colors=color_cont[1], alpha=alp_rb, linewidths=lw_rb, linestyles=linstyl)
rb_contour3 = microim7.ax.contour(cropped_rb_mask_zoom_dil[2], colors=color_cont[1], alpha=alp_rb, linewidths=lw_rb, linestyles=linstyl)
rb_contour4 = microim8.ax.contour(cropped_rb_mask_zoom_dil[3], colors=color_cont[1], alpha=alp_rb, linewidths=lw_rb, linestyles=linstyl)

nuc_contour5 = microim5.ax.contour(cropped_nuc_mask_zoom_dil[0], colors=color_cont[0], alpha=alp, linewidths=lw, linestyles=linstyl)
nuc_contour6= microim6.ax.contour(cropped_nuc_mask_zoom_dil[1], colors=color_cont[0], alpha=alp, linewidths=lw, linestyles=linstyl)
nuc_contour7 = microim7.ax.contour(cropped_nuc_mask_zoom_dil[2], colors=color_cont[0], alpha=alp, linewidths=lw, linestyles=linstyl)
nuc_contour8 = microim8.ax.contour(cropped_nuc_mask_zoom_dil[3], colors=color_cont[0], alpha=alp, linewidths=lw, linestyles=linstyl)

#plt.show()

#---------------------------------------- Save
savename = ['Fig 3E_colorbar_'+figname_file+'.png' if colorb else 'Fig 3E_'+figname_file+'.png']
savename = fig_output+savename[0]
panel.savefig(savename, bbox_inches = 'tight', dpi=1000)


#%% Fig 3F  Dense imaging (Relish and rhobast trace)
figsize3 = [2.8, 1.25] 

#---------------------------------------------------------- Create contiguous (compressed) time arrays
t1 = df_times['T1'].to_numpy()
t2 = df_times['T2'].to_numpy()
t3 = df_times['T3'].to_numpy()

# Assume uniform time spacing in each segment.
gap = 20
delta1 = t1[1]-t1[0]
new_t1 = t1.copy()
offset2 = new_t1[-1] + gap + delta1
new_t2 = t2 - t2[0] + offset2
delta2 = t2[1]-t2[0]
offset3 = new_t2[-1] + gap + delta2
new_t3 = t3 - t3[0] + offset3
# Bundle them in a dictionary for ease of use:
new_times = {'T1': new_t1, 'T2': new_t2, 'T3': new_t3}

# ---------------------------------------------------------- Plot using the new, compressed time arrays 
fig, axs = plt.subplots(2, 1, figsize=figsize3, sharex=True)
lines = []

for win in [1, 2, 3]:
    timewin = f'T{win}'
    is_first = (win == 1)
    y_rel = df_relish_ratio[timewin]
    y_rb = df_rb[timewin] / 100
    current_times = new_times[timewin]  # Use the compressed times for plotting

    color = 'k' 
    # Top Panel: Relish Ratio
    line, = axs[0].plot(current_times, y_rel, linewidth=1, label=str(win),
                          color=color, zorder=3)
    lines.append(line)
    # Bottom Panel: Rhobast
    axs[1].plot(current_times, y_rb, linewidth=1, color=color,
                zorder=3)
    
    if is_first:
        axs[0].set_ylim([.45, .7])
        axs[1].set_ylim([-2, 15])
        # Place the text relative to the compressed time axis:
        axs[1].text(new_t1[0] - 45, 15, 'x10$^2$', ha='left', va='top', fontsize=tickfsize)
        axs[1].set_yticks([0, 5])
        axs[0].set_yticks([0.6, 0.7])

        #axs[0].set_ylabel('$R_{nuc:tot}$', fontsize=fsize)
        #axs[1].set_ylabel('∑ Foci', fontsize=fsize)
        axs[1].set_xlabel('Time (min)', fontsize= fsize)

        # Pre- and post-stim hatching bars (adjust indices as needed)
        for ax in axs:
            y_min, y_max = ax.get_ylim()
            hatch_height = (y_max - y_min) * 0.07
            # ax.add_patch(patches.Rectangle((current_times[0], y_min), 0, hatch_height,
            #                                edgecolor="black", facecolor="none", hatch='///', zorder=1))
            ax.add_patch(patches.Rectangle((current_times[0], y_min), 930,
                                           hatch_height, color="black", zorder=1))

    # Time point markers (using indices from your variable "times"; adjust as needed)
    mark_indices = times[0:2] if is_first else [times[win]]
    for idx in mark_indices:
        x_val = current_times[idx]
        axs[0].plot(x_val, y_rel[idx], 'o', markersize=4, color='#db577b', alpha=0.9, zorder=4)
        axs[1].plot(x_val, y_rb[idx], 'o', markersize=4, color='#c25a13', alpha=0.9, zorder=4)

# ---------------------------------------------------------- Set custom x-ticks with original time labels
# For each block, generate tick positions (using the new times) and labels (from original times)
ticks1 = np.arange(new_t1[0], new_t1[-1] + 1, 40)
labels1 = np.arange(t1[0], t1[-1] + 1, 40)
ticks2 = np.arange(new_t2[0], new_t2[-1] + 1, 40)
labels2 = np.arange(t2[0], t2[-1] + 1, 40)
ticks3 = np.arange(new_t3[0], new_t3[-1] + 1, 40)
labels3 = np.arange(t3[0], t3[-1] + 1, 40)

# Combine the ticks and labels from all three segments:
xticks_combined = np.concatenate([ticks1, ticks2, ticks3])
xtick_labels_combined = np.concatenate([labels1, labels2, labels3])

# Apply these to the bottom axis (they will be shared):
axs[1].set_xticks(xticks_combined)
axs[1].set_xticklabels([f"{label:.0f}" for label in xtick_labels_combined], fontsize=tickfsize, rotation=45)
axs[0].set_ylabel('$R_{nuc:tot}$', fontsize=fsize)
axs[1].set_ylabel('Int$_R$$_B$ (AU)', fontsize=fsize)
for ax in axs.flat:  # Works for both 1D and 2D subplot arrays
    ax.yaxis.set_label_coords(-.12, 0.5)
    ax.tick_params(axis='x', which='both', pad=.3)

    
# Set the x-axis limits to span the new contiguous time values:
axs[1].set_xlim([new_t1[0], new_t3[-1]])

# Define the interval in minutes (30 seconds = 0.5 minutes)
line_interval = 0.5  # minutes

# For each window, generate tick positions every 30 seconds:
ticks_t1 = np.arange(new_t1[0], new_t1[-1] + line_interval, line_interval)
ticks_t2 = np.arange(new_t2[0], new_t2[-1] + line_interval, line_interval)
ticks_t3 = np.arange(new_t3[0], new_t3[-1] + line_interval, line_interval)

# Combine all tick positions:
all_ticks = np.concatenate([ticks_t1, ticks_t2, ticks_t3])

# Draw vertical dashed lines at each 30-second interval within each window
for tick in all_ticks:
    for ax in axs:
        ax.axvline(x=tick, color='gray', linestyle='-', linewidth=0.05, 
                   alpha=0.4, zorder=0)


# Add legend and label the x-axis:
# axs[0].legend(handles=lines, loc='upper right', fontsize=tickfsize * shrink, 
#               title='Time Window', handlelength=1.2,
#               ncol = 3, labelspacing=0.2, columnspacing=0.5)

plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.01,  hspace=0.12)
#plt.show()

#---------------------------------------- Save
savename = 'Fig 3F_DenseTraces_'+figname_file+'.png' 
savename = fig_output+savename
fig.savefig(savename, bbox_inches = 'tight', dpi=1000)
