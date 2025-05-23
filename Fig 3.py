# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 15:59:30 2025

@author: noshin
"""

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
mask_settings = '15link_nuc8_cyto40' #'15link_nuc6.5-(cyto)250_cyto40-950' 

resolution=  3.4756 #pixels per micron
units_per_pix = 1/resolution
nuc_channel = 0
rel_channel = 1
rb_channel = 2

sg_factor_rel = 5   #5 for relish data
sg_factor_rb = 8   #5 for relish data
sg_order_rel  = 2   #Polynomial order for Savitzky–Golay smoothing


plt.rcParams['font.family'] = 'Arial'

fsize = 10
tickfsize = 9
mag = 1.2
shrink = 0.78

fig_output = 'G:/Shared drives/Wunderlich Lab/People/Noshin/Paper/Figures/'

#%% Time diagram
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

savename = fig_output+'Figure 3/'+'TimeDiagram_'+'.png'
fig.savefig(savename, dpi=1000, bbox_inches='tight')

#%% Import cell info for long imaging (Dpt)

project = 'attB-LacZ-RhoBAST-Halotag-Relish/' 
rootdir = computer+'Imaging Data/Noshin Imaging Data/'+project

#import dataset dict_intensities (full dataset)
intensities_df_import = rootdir+ 'Python/'+mask_settings+'/Intensities DF/'
# with open(os.path.join(intensities_df_import, 'dict_intensities_ilastikpeaks_meansum_goodcells_010925'), 'rb') as handle:
#     dict_intensities = pickle.load(handle)
intensities_df_import = computer+'Imaging Data/Noshin Imaging Data/'+project+ 'AllDatasets_IntensitiesDict/'
with open(os.path.join(intensities_df_import, 'dict_intensities_ilastikpeaks_meansum_alldatasets_goodcells_nomasks_010925'), 'rb') as handle:
    dict_intensities = pickle.load(handle)
dict_intensities = dict_intensities['2024-10-23']

fname = '07_Dpt_100X_1_maxZ.tif'#'09_Dpt_100X_3_maxZ.tif'
f_dict_intensities = dict_intensities[fname]
df_relish_ratio = pd.DataFrame({cell: [intensity for time, intensity in values] for cell, values in f_dict_intensities['relish_ratio'].items()})
df_rb = pd.DataFrame({cell: [intensity for time, intensity in values] for cell, values in f_dict_intensities['peak_intensities_sum'].items()})

#set dataset parameters
interval_forplt = np.concatenate([np.arange(0, 121, 15) , np.arange(150, 631, 30), np.arange(690, 991, 60)])
interval_hrs = [str(timedelta(minutes=int(time.item())))[:-3] for time in interval_forplt]
stim=4 #first slice after inject -1!!!

curr_dataset= '2024-10-23 DptMtk_LacZ_RhoBAST/'
rootdir_day = rootdir+curr_dataset

tifs =  rootdir_day+'TIF_Split_Series_MaxZ/'    
cyto_masks = rootdir_day+'Trackmate Files/'+mask_settings+'/Cyto Matched Masks/' #Updated masks with interpolated (Python Step 3-1) then matched (Fiji Step 3-2)
nuc_masks = rootdir_day+ 'Trackmate Files/'+mask_settings+'/Nuclei Matched Masks/'
rb_masks = rootdir_day+ 'ilastik Outputs/Aptamer Masks/'
time_series_tif = tf.imread(tifs+fname)
cyto_mask_tif = tf.imread(cyto_masks+fname)
nuc_mask_tif = tf.imread(nuc_masks+fname)
rb_mask_tif = tf.imread(rb_masks+fname[:-4]+'_Object Identities.tiff')

nframes = time_series_tif.shape[0]
    
#%% FIG B long imaging time-course cell panels

#cell 09_Dpt_100X_3_Cell 56
celln= 92
cell='Cell '+str(celln)

times = [0,8,16,29]

colorb = False
figname = curr_dataset.split(' ')[0]+fname[3:-9]+"_"+cell
figsize1 = [2.8,2]#[2,1.2]

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

 
    # cropped_cell_rb_zoom[count] = time_series_tif[t, rb_channel, ymin+buffer:ymax-(int(buffer*.2)), xmin+buffer:xmax-buffer]
    # cropped_nuc_mask_zoom[count] = nuc_mask[ymin+buffer:ymax-(int(buffer*.2)), xmin+buffer:xmax-buffer]
    # cropped_rb_mask_zoom[count] = rb_mask[ymin+buffer:ymax-(int(buffer*.2)), xmin+buffer:xmax-buffer]

    mins[count] = str(interval_forplt[t]-60)# + ' min'

#---------------------------------------------------------- Microplot -----------START RUN HERE AFTER FIRST TIME
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

#---------------------------------------------------------- Save
plt.show()

if colorb:
    savename = fig_output+'Figure 3/'+'RelRhobastSignalTime_'+figname+'.png'
else:
    savename = fig_output+'Figure 3/'+'RelRhobastSignalTime_nocolorbar_'+figname+'.png'
panel.savefig(savename, bbox_inches = 'tight', dpi=1000)

#%% FIG B long imaging time-course cell panels

#cell 09_Dpt_100X_3_Cell 56
celln= 92
cell='Cell '+str(celln)

times = [0,8,16,29]


figname = curr_dataset.split(' ')[0]+fname[3:-9]+"_"+cell
figsize1 = [5,2]

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


cropped_cell_rel,       cropped_cell_rb = {},{}
cropped_cyto_mask,      cropped_nuc_mask,       cropped_rb_mask = {},{},{}
cropped_cell_rb_zoom,   cropped_nuc_mask_zoom,  cropped_rb_mask_zoom = {},{},{}
mins = {}

buffer = 20

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
 
    
    buff_scale_max = int(ymax-buffer*1.2)
    buff_scale_min = int(ymin+buffer*.42)
    cropped_cell_rb_zoom[count] = time_series_tif[t, rb_channel, buff_scale_min:buff_scale_max, xmin+buffer:xmax-buffer]
    cropped_nuc_mask_zoom[count] = nuc_mask[buff_scale_min:buff_scale_max, xmin+buffer:xmax-buffer]
    cropped_rb_mask_zoom[count] = rb_mask[buff_scale_min:buff_scale_max, xmin+buffer:xmax-buffer]

 
    # cropped_cell_rb_zoom[count] = time_series_tif[t, rb_channel, ymin+buffer:ymax-(int(buffer*.2)), xmin+buffer:xmax-buffer]
    # cropped_nuc_mask_zoom[count] = nuc_mask[ymin+buffer:ymax-(int(buffer*.2)), xmin+buffer:xmax-buffer]
    # cropped_rb_mask_zoom[count] = rb_mask[ymin+buffer:ymax-(int(buffer*.2)), xmin+buffer:xmax-buffer]

    mins[count] = str(interval_forplt[t]-60)+ ' min'

#---------------------------------------------------------- Microplot -----------START RUN HERE AFTER FIRST TIME
fig, ax = plt.subplots(dpi=1000)

colormaps_rel = ['magma']
minmax = [0,4500]

microim1 = microshow(cropped_cell_rel[0], cmaps=colormaps_rel, label_text=mins[0], label_font_size=fsize, rescale_type='limits', limits= minmax)
microim2 = microshow(cropped_cell_rel[1], cmaps=colormaps_rel, label_text=mins[1], label_font_size=fsize, rescale_type='limits', limits= minmax)
microim3 = microshow(cropped_cell_rel[2], cmaps=colormaps_rel, label_text=mins[2], label_font_size=fsize, rescale_type='limits', limits= minmax)
microim4 = microshow(cropped_cell_rel[3], cmaps=colormaps_rel, label_text=mins[3], label_font_size=fsize, rescale_type='limits', limits= minmax,
                     unit='um', scalebar_unit_per_pix=units_per_pix, scalebar_size_in_units=10, 
                    scalebar_color='white',scalebar_font_size=None,
                    show_colorbar=True)
cmap_rb = ['magma']#['copper']
minmax = [0, 4500]
microim5 = microshow(cropped_cell_rb_zoom[0], cmaps=cmap_rb, rescale_type='limits', limits= minmax)
microim6 = microshow(cropped_cell_rb_zoom[1], cmaps=cmap_rb, rescale_type='limits', limits= minmax)
microim7 = microshow(cropped_cell_rb_zoom[2], cmaps=cmap_rb, rescale_type='limits', limits= minmax)
microim8 = microshow(cropped_cell_rb_zoom[3], cmaps=cmap_rb, rescale_type='limits', limits= minmax,
                                         unit='um', scalebar_unit_per_pix=units_per_pix, scalebar_size_in_units=10, 
                                        scalebar_color='white', scalebar_font_size=tickfsize,
                                        show_colorbar=True)

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

lw = .6
alp= .9
lw_rb = .6
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

#---------------------------------------------------------- Save
plt.show()

savename = fig_output+'Figure 3/'+'RelRhobastSignalTime_colorbar_'+figname+'.png'
#panel.savefig(savename, bbox_inches = 'tight', dpi=1000)

#%% FIG C long imaging Relish and rhobast trace

fig2size = [3, 1.2]  # Adjusted the figure size for two panels
fig, axs = plt.subplots(2, 1, figsize=fig2size, sharex=True)  # Create a 2-panel plot

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

savename = fig_output+'Figure 3/'+'RelRhobastTrace_'+figname+'.png'
fig.savefig(savename, dpi=1000, bbox_inches='tight')

#%% FIG D long imaging heatplot of Dpt100X spikers with sorted behaviors
figsize3 = (6, 4.5)

heat_color = 'magma'

def divide_by_thousand(x, pos):
    """Custom formatter to divide colorbar values by 1,000."""
    return f'{x / 1000:.0f}'

#import full dataset
intensities_df_import = computer+'Imaging Data/Noshin Imaging Data/attB-LacZ-RhoBAST-Halotag-Relish/AllDatasets_IntensitiesDict/'
with open(os.path.join(intensities_df_import, 'dict_intensities_ilastikpeaks_meansum_alldatasets_goodcells_nomasks_010925'), 'rb') as handle:
    dict_intensities_all = pickle.load(handle)
#----------------------------------------------------------  Make df of all cells that spike
spikers = []
for date, fnames in dict_intensities_all.items():
    for fname, data in fnames.items():
        if 'Dpt_100X' in fname:
            for cell, t_int_list in data['peak_intensities_sum'].items():
                rep = fname.split('_')[3]
                # Check if any intensity value is non-zero
                if any(intensity != 0 for _, intensity in t_int_list):
                    spikers.append([date, rep, cell, t_int_list])

spikers_df = pd.DataFrame(spikers, columns=['Date', 'Rep', 'Cell', 'Time_Intensity'])
time_points = [time for time, _ in spikers_df.iloc[0]['Time_Intensity']]

spikers_df['Average_Intensity'] = spikers_df['Time_Intensity'].apply(
    lambda t_int_list: np.mean([intensity for _, intensity in t_int_list]))

#----------------------------------------------------------  Order by Relish behavior category
with open(os.path.join(fig_output, 'Figure 3/goodcomp7_rel_AMP_SVM_results_dict_noIc.pkl'), 'rb') as handle:
    rel_behav_dict = pickle.load(handle)

dpt100X_behav = rel_behav_dict['All cells']
dpt100X_behav = dpt100X_behav[['100X Cell Dpt' in s for s in dpt100X_behav.index]]
    
dpt100X_behav = dpt100X_behav.reset_index()
dpt100X_behav['Date'] = dpt100X_behav['index'].apply(lambda x: '2024-10-' + x.split(' ')[2][10:12])
dpt100X_behav['Cell'] = dpt100X_behav['index'].apply(lambda x: 'Cell ' + x.split('-')[-1])
dpt100X_behav['Rep'] = dpt100X_behav['index'].apply(lambda x: x.split('-')[-2])

#Filter dpt100X_behav to include only matching 'Date' and 'Cell' pairs
matching_pairs = pd.merge(spikers_df[['Date', 'Cell','Rep']], dpt100X_behav[['Date', 'Cell','Rep']], on=['Date', 'Cell', 'Rep'])
dpt100X_behav_filtered = dpt100X_behav.merge(matching_pairs, on=['Date', 'Cell','Rep'])

# Set 'Date' and 'Cell' as the index for both DataFrames
spikers_df.set_index(['Date', 'Cell','Rep'], inplace=True)
dpt100X_behav_filtered.set_index(['Date', 'Cell','Rep'], inplace=True)

# Assign 'Behavior' values to spikers_df
spikers_df['Behavior'] = dpt100X_behav_filtered['Predicted']


#----------------------------------------------------------  Make sorted matrix of all cells that spike
color_palette  = ["#DC143C", "#FF6F61", "indigo", "dodgerblue", "grey"]
behavior_colors = {"I": color_palette[0], "Id": color_palette[1], "G": color_palette[2], "D": color_palette[3], "N": color_palette[4]}


behavior_order = ['I', 'Id', 'G', 'D', 'N']  # Replace with your actual behavior group names
behavior_labels = {
    'I': {'display_name': 'I', 'color': color_palette[0]},
    'Id': {'display_name': 'Id', 'color': color_palette[1]},
    'G': {'display_name': 'G', 'color': color_palette[2]},
    'D': {'display_name': 'D', 'color': color_palette[3]},
    'N': {'display_name': 'N', 'color': color_palette[4]}
}

spikers_df['Behavior'] = pd.Categorical(spikers_df['Behavior'], categories=behavior_order, ordered=True)
spikers_df_sorted = spikers_df.sort_values(by=['Behavior', 'Average_Intensity'], ascending=[True, False]).reset_index(drop=True)

data_matrix = []
for _, row in spikers_df_sorted.iterrows():
    intensities = [intensity for _, intensity in row['Time_Intensity']]
    data_matrix.append(intensities)
data_matrix = np.array(data_matrix)
num_cells = len(data_matrix)
#---------------------------------------------------------- # Plot the heatmap 

# fig, ax = plt.subplots(figsize=(figsize3), dpi= 1000)

# sns.heatmap(data_matrix, xticklabels=time_points, cmap=heat_color, ax=ax, cbar=True, cbar_kws={'label': 'Foci Sum Intensity (×10³)', 'shrink': 0.8})

# cbar = ax.collections[0].colorbar
# cbar.formatter = ticker.FuncFormatter(divide_by_thousand)
# cbar.update_ticks()
# cbar.ax.tick_params(labelsize=tickfsize)  # Set colorbar tick labels size

# # Set labels and title
# ax.set_ylabel('Single Cells\n', fontsize=fsize)
# ax.set_xlabel('Time (min)', fontsize=fsize)
# ax.set_yticks([]) 

# # Adjust x-axis tick labels to show every other label
# for index, label in enumerate(ax.get_xticklabels()):
#     if index % 2 != 0:  # Hide every other label
#         label.set_visible(False)
#     label.set_fontsize(tickfsize)

# # Determine the start and end indices for each behavior group
# behavior_groups = spikers_df_sorted.groupby('Behavior').size().cumsum()
# previous_index = 0

# for behavior, end_index in behavior_groups.items():
#     # Calculate the midpoint for placing the text label
#     midpoint = (previous_index + end_index) / 2
#     # Add the behavior label
#     display_name = behavior_labels[behavior]['display_name']
#     color = behavior_labels[behavior]['color']
#     ax.text(-0.5, midpoint, display_name, va='center', ha='right', fontsize=tickfsize, fontweight='bold', color=color)    # Add the brace
#     # Add the bracket
#     ax.plot([-0.3, -0.3], [previous_index, end_index], color= color, linewidth=1.5, clip_on=False)
#     ax.plot([-0.3, -0.2], [previous_index, previous_index], color= color, linewidth=1.5, clip_on=False)
#     ax.plot([-0.3, -0.2], [end_index, end_index], color=color, linewidth=1.5, clip_on=False)
#     previous_index = end_index


# # Display the plot
# plt.show()

# savename = fig_output+'Figure 3/'+'Heatmap_Dpt100X_Spikers_colordiff.png'
# #fig.savefig(savename, dpi=1000, bbox_inches='tight')

#%%
#---------------------------------------------------------- # Plot the heatmap with nonlinear time

# Calculate the edges of the time bins
time_points = np.array(time_points)
time_edges = np.zeros(len(time_points) + 1)
time_edges[1:-1] = (time_points[:-1] + time_points[1:]) / 2
time_edges[0] = time_points[0] - (time_points[1] - time_points[0]) / 2
time_edges[-1] = time_points[-1] + (time_points[-1] - time_points[-2]) / 2


fig, ax = plt.subplots(figsize=(figsize3), dpi=1000)

# Create the heatmap using pcolormesh
c = ax.pcolormesh(time_edges, np.arange(num_cells + 1), data_matrix, cmap=heat_color, shading='flat')

cbar = fig.colorbar(c, ax=ax, label='Foci Sum Intensity (×10³)', shrink=0.8)
cbar.formatter = ticker.FuncFormatter(divide_by_thousand)
cbar.outline.set_visible(False)
cbar.update_ticks()
cbar.ax.tick_params(labelsize=tickfsize)  # Set colorbar tick labels size

# Set labels and title
ax.set_ylabel('Single Cells', fontsize= fsize)
ax.set_yticks([]) 
ax.invert_yaxis()


ax.set_xlabel('Time (min)', fontsize= fsize)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)


behavior_groups = spikers_df_sorted.groupby('Behavior').size().cumsum()
previous_index = 0

bracketx = -80

for behavior, end_index in behavior_groups.items():
    # Calculate the midpoint for placing the text label
    midpoint = (previous_index + end_index) / 2
    # Add the behavior label
    display_name = behavior_labels[behavior]['display_name']
    color = behavior_labels[behavior]['color']
    ax.text(-88, midpoint, display_name, va='center', ha='right', fontsize=tickfsize, fontweight='bold', color=color)
    # Add the curly brace
    ax.plot([bracketx, bracketx], [previous_index, end_index], color= color, linewidth=1.5, clip_on=True)
    ax.plot([bracketx, bracketx+8], [previous_index, previous_index], color= color, linewidth=1.5, clip_on=True)
    ax.plot([bracketx, bracketx+8], [end_index, end_index], color=color, linewidth=1.5, clip_on=True)
    previous_index = end_index

# Adjust x-axis tick labels to show every other label
ax.set_xticks(time_points)
ax.set_xticklabels([f'{int(t)}' if i % 2 == 0 else '' for i, t in enumerate(time_points)], fontsize=tickfsize, rotation=90)


stim_frame = time_points[stim]
ax.axvline(stim_frame, color='white', linestyle='-', linewidth=1, alpha= 0.5)  # Red dashed line
ax.text(4,-9.5,  # Adjusted text placement
        '+ (100 µg/mL PGN) / (water)', color='k', ha='left', va='center', fontsize=fsize)


# Display the plot
plt.show()

savename = fig_output+'Figure 3/'+'Heatmap_Nonlintime_Dpt100X_Spikers_colordiff.png'
fig.savefig(savename, dpi=1000, bbox_inches='tight')
#----------------------------------------------------------


#%% Import cell info for dense imaging (Dpt)

# Import cell info for long imaging, 02/26 cell 147
curr_dataset = '2025-02-26 DptMtk_Dense_3time/'

project = 'Dense RhoBAST/' 
rootdir = computer+'Imaging Data/Noshin Imaging Data/'+project+curr_dataset

tifs =  rootdir+'TIF_Split_Series_MaxZ/'    
cyto_masks = rootdir+'Trackmate Files/'+mask_settings+'/Cyto Matched Masks/' 
nuc_masks = rootdir+ 'Trackmate Files/'+mask_settings+'/Nuclei Matched Masks/'
rb_masks = rootdir+ 'ilastik Outputs/Aptamer Masks/'

fname_root = 'Dpt_100X_T'
# Create dictionaries for each type of image
time_series_tifs = {}
cyto_mask_tifs = {}
nuc_mask_tifs = {}
rb_mask_tifs = {}

for timewin in [1, 2, 3]:
    fname = fname_root + str(timewin)
    time_series_tifs[f'T{timewin}'] = tf.imread(tifs + fname+'.tif')
    cyto_mask_tifs[f'T{timewin}'] = tf.imread(cyto_masks + fname+'.tif')
    nuc_mask_tifs[f'T{timewin}'] = tf.imread(nuc_masks + fname+'.tif')
    rb_mask_tifs[f'T{timewin}'] = tf.imread(rb_masks + fname + '_Object Identities.tif')

nframes = time_series_tifs['T1'].shape[0]    

#import dataset dict_intensities (full dataset for multitime cells)
intensities_df_import = 'Z:/Imaging Data/Noshin Imaging Data/Dense RhoBAST/AllDatasets_IntensitiesDict/'
with open(os.path.join(intensities_df_import, 'cellsmultiwindows_dataframe_peakproperties_alldays_prom300_relativheight0.9_032025.pkl'), 'rb') as handle:
    peaks_df_cellsmultiplewindows = pickle.load(handle)

#concatenate dataset for 02/26 cell 147
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

full_time = np.concatenate([
    np.arange(0, 120.1, 0.5),
    np.arange(405,525.1,0.5),
    np.arange(810,930.1,0.5)
]).tolist()


#%% FIG E dense imaging time-course cell panels
figsize4 =  [2.65,1.9]
colorb= True
#cells with great short dynamics:
#    1_maxZ_Cell 31 156 
#    2_maxZ_cell 15
# 02/26 cell 147 (has all 3 time windows)

celln = 147
cell='Cell '+str(celln)
figname = fname[:-3]+"_"+cell

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
    #cropped_cell[count] = time_series_tif[t,[nuc_channel,rel_channel], ymin:ymax, xmin:xmax]
    #scale along y differently
    

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
 
panel = Micropanel(rows=2, cols=4, figsize=figsize4) #figscaling=2.5)  
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

#---------------------------------------- Save
plt.show()
if colorb:
    savename = fig_output+'Figure 3/'+'RelRhobastDENSESignalTime_'+figname+'.png'
else:
    savename = fig_output+'Figure 3/'+'RelRhobastDENSESignalTime_nocolorbar_'+figname+'.png'

panel.savefig(savename, bbox_inches = 'tight', dpi=1000)

#%% FIG F dense imaging Relish and rhobast trace (3 times, cell 147)
#other_cols = ['black','cornflowerblue','gray']
other_cols = ['black','black','black']

fig5size = [2.8, 1.25] 

# Assume df_times, df_relish_ratio, df_rb, other_cols, fig5size, tickfsize, shrink, 
# fig_output, figname, and times are already defined.

# === Step 1. Create contiguous (compressed) time arrays ===
# Get the original time arrays from your DataFrame
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

# === Step 2. Plot using the new, compressed time arrays ===
fig, axs = plt.subplots(2, 1, figsize=fig5size, sharex=True)
lines = []

for win in [1, 2, 3]:
    timewin = f'T{win}'
    is_first = (win == 1)
    y_rel = df_relish_ratio[timewin]
    y_rb = df_rb[timewin] / 100
    current_times = new_times[timewin]  # Use the compressed times for plotting

    color = 'k' if is_first else other_cols[win - 1]
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

# === Step 3. Set custom x-ticks with original time labels ===
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
plt.show()

# Save the figure
savename = fig_output + 'Figure 3/RelRhobastDENSETraces_3windows_' + figname + '.png'
fig.savefig(savename, dpi=1000, bbox_inches='tight')

#%% FIG G dense imaging heatplot of Dpt100X spikers

#%% heatmap of dense spikers
figsize6 = (6, 4.5)

#import dict
intensities_df_import = 'Z:/Imaging Data/Noshin Imaging Data/Dense RhoBAST/AllDatasets_IntensitiesDict/'
with open(os.path.join(intensities_df_import, 'cellsmultiwindows_dataframe_peakproperties_alldays_prom300_relativheight0.9_032025'), 'rb') as handle:
    peaks_df_allcells = pickle.load(handle)
    
interval_forplt = np.concatenate([np.arange(0, 3630, 30)])

#----------------------------------------------------------  Make sorted matrix of all cells that spike
spikers = []
for fnames, data in dict_intensities.items():
    for cell, t_int_list in data['peak_intensities_sum'].items():
        # Check if any intensity value is non-zero
        if any(intensity != 0 for _, intensity in t_int_list):
            spikers.append([fnames, cell, t_int_list])

spikers_df = pd.DataFrame(spikers, columns=['Fname', 'Cell', 'Time_Intensity'])
spikers_df['Average_Intensity'] = spikers_df['Time_Intensity'].apply(
    lambda t_int_list: np.mean([intensity for _, intensity in t_int_list]))
spikers_df_sorted = spikers_df.sort_values(by='Average_Intensity', ascending=False).reset_index(drop=True)

data_matrix = []
for _, row in spikers_df_sorted.iterrows():
    intensities = [intensity for _, intensity in row['Time_Intensity']]
    data_matrix.append(intensities)

# Convert to a NumPy array for easier manipulation
data_matrix = np.array(data_matrix)

#----------------------------------------------------------  Make sorted matrix of all cells that spike
def divide_by_thousand(x, pos):
    return f'{x / 1000:.0f}'

fig, ax = plt.subplots(figsize=(figsize6), dpi=1000)
# Plot the heatmap with customized colorbar label
ax= sns.heatmap(data_matrix, xticklabels=interval_forplt,
            cmap=heat_color, ax=ax, cbar=True, 
            cbar_kws={'label': 'Ratio', 'shrink': 0.8})

cbar = ax.collections[0].colorbar
# Set the formatter for the colorbar
cbar.formatter = ticker.FuncFormatter(divide_by_thousand)
cbar.update_ticks()
# Set the colorbar label with the desired font size and multiplier notation
cbar.set_label('Foci Sum Intensity (×10³)', fontsize=tickfsize)
cbar.ax.tick_params(labelsize=tickfsize)  # Set colorbar tick labels size

# Set labels and title
ax.set_ylabel('Single Cells', fontsize=fsize)
ax.set_yticks([]) 

ax.set_xlabel("Time (sec)", fontsize=fsize)


# Set the x-axis tick labels' font size
for index, label in enumerate(ax.get_xticklabels()):
    if index % 10 != 0:  # Hide every other label
        label.set_visible(False)
    label.set_fontsize(fsize * shrink)
plt.show()

savename = fig_output+'Figure 3/'+'Heatmap_Dense_Dpt100X_Spikers_diffcolor.png'
plt.savefig(savename, dpi=1000, bbox_inches='tight')
