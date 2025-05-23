# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 13:47:54 2025

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
from matplotlib.lines import Line2D
from matplotlib.cm import get_cmap
import matplotlib.ticker as ticker
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import matplotlib.gridspec as gridspec
import os
from scipy import ndimage
from scipy.signal import savgol_filter
from skimage import measure
import skimage.io
from skimage.measure import label, regionprops
from scipy.ndimage import label
from scipy.ndimage import binary_dilation
from scipy import stats
from scipy.stats import ks_2samp
from scipy.signal import savgol_filter
from skimage.segmentation import mark_boundaries
from skimage.feature.peak import peak_local_max
from scipy.stats import ks_2samp
from sklearn.metrics import roc_curve, roc_auc_score
from MLstatkit.stats import Delong_test
from statsmodels.stats.multitest import multipletests
from PIL import Image
import itertools
import pickle
import  microfilm.microplot
from microfilm import microplot
from microfilm.microplot import microshow, Micropanel
import gc
import copy
import datetime
import time
import random
from datetime import timedelta
datetime_str = time.strftime("%m%d%y_%H:%M")

mac = '/Volumes/rkc_wunderlichLab/'
PC = 'R:/'
anacomp = 'Z:/'

computer = anacomp

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

if computer == mac:
    gdrive = '/Users/noshin/Library/CloudStorage/GoogleDrive-noshin@bu.edu/Shared drives/Wunderlich Lab/People/Noshin/'
else:
    gdrive = 'G:/Shared drives/Wunderlich Lab/People/Noshin/'

fig_output = gdrive+'Paper/Figures/'


#%% Histogram of Relish ratios by spikers
intensities_df_import = gdrive+"Paper/Figures/Figure Codes/Figure 5 Dicts/Histogram Dicts/Sparse Imaging/"
with open(os.path.join(intensities_df_import, 'goodcomp7_times.pkl'), 'rb') as handle:
    goodcomp7_times = pickle.load(handle)
times = goodcomp7_times

#Relish values
with open(os.path.join(intensities_df_import, 'goodcomp7_rbpeaks_rel_dict.pkl'), 'rb') as handle:
    peaksovertime_rel = pickle.load(handle)
#AUCs
with open(os.path.join(intensities_df_import, 'goodcomp7_rbpeaks_AUCs_dict.pkl'), 'rb') as handle:
    peaksovertime_auc = pickle.load(handle)

#--------------------------------------
#make datafarmes for each condition to count (same across normalizations)
conc= '100X'

rel_peaks = peaksovertime_rel['Norm']['Peaks']
rel_nopeaks = peaksovertime_rel['Norm']['No peaks']
total_cells_allconc = rel_nopeaks.shape[0]


rel_peaks_conc = rel_peaks[[conc in s for s in rel_peaks.index]]
rel_nopeaks_conc = rel_nopeaks[[conc in s for s in rel_nopeaks.index]]

#break up by enh
rel_peaks_conc_im2 = rel_peaks_conc[['IM2' in s for s in rel_peaks_conc.index]]
rel_nopeaks_conc_im2 = rel_nopeaks_conc[['IM2' in s for s in rel_nopeaks_conc.index]]

rel_peaks_conc_dpt = rel_peaks_conc[['Dpt' in s for s in rel_peaks_conc.index]]
rel_nopeaks_conc_dpt = rel_nopeaks_conc[['Dpt' in s for s in rel_nopeaks_conc.index]]

rel_peaks_conc_mtk = rel_peaks_conc[['Mtk' in s for s in rel_peaks_conc.index]]
rel_nopeaks_conc_mtk = rel_nopeaks_conc[['Mtk' in s for s in rel_nopeaks_conc.index]]

#--------------------------------------
# count of how many spikers are across the whole dataset, just for 100X
total_cells = rel_nopeaks_conc.shape[0]

total_time = total_cells*28
ncells_nonepeakers = rel_nopeaks_conc.dropna().shape[0]
ncells_atleastone_peakers = rel_peaks_conc.apply(lambda row: row.notna().any(), axis=1).sum()
#total: 494/1184 (42%) cells peak at least once post-stim(100X)

total_im2 = rel_peaks_conc_im2.shape[0]
ncells_im2_nonepeakers = rel_nopeaks_conc_im2.dropna().shape[0]
ncells_im2_atleastone_peakers = rel_peaks_conc_im2.apply(lambda row: row.notna().any(), axis=1).sum()
#im2: 0/84 (0%) cells peak at least once post-stim  (100X)
perc_im2 = (ncells_im2_atleastone_peakers/total_im2 ) *100

total_dpt = rel_peaks_conc_dpt.shape[0]
ncells_dpt_nonepeakers = rel_nopeaks_conc_dpt.dropna().shape[0]
ncells_dpt_atleastone_peakers = rel_peaks_conc_dpt.apply(lambda row: row.notna().any(), axis=1).sum()
#dpt 365/653 (56%)
perc_dpt = (ncells_dpt_atleastone_peakers/total_dpt ) *100

total_mtk = rel_peaks_conc_mtk.shape[0]
ncells_mtk_nonepeakers = rel_nopeaks_conc_mtk.dropna().shape[0]
ncells_mtk_atleastone_peakers = rel_peaks_conc_mtk.apply(lambda row: row.notna().any(), axis=1).sum()
#mtk 129/447 (29%)
perc_mtk = (ncells_mtk_atleastone_peakers/total_mtk ) *100

 

#%%
def add_ks_significance(ax, sample1, sample2, tickfsize, alternative='less'):
    """
    Performs a KS test on sample1 and sample2 and, if significant (p < 0.05),
    plots a horizontal line with vertical ticks and a star annotation on the provided axis.
    
    Parameters:
      ax         : matplotlib.axes.Axes object on which to plot the significance markers.
      sample1    : 1D array-like data (e.g., peaks for one condition).
      sample2    : 1D array-like data (e.g., no peaks for one condition).
      tickfsize  : Font size for the annotation text.
      alternative: Alternative hypothesis for the KS test (default 'less').
    """
    
    stat, p_val = ks_2samp(sample1, sample2, alternative=alternative)
    if p_val < 0.05:
        marker = '*'
        if p_val < 0.01:
            marker = '**'
            
        # Determine the maximum y-value across lines and collections on the axis.
        max_y = 0
        for line in ax.get_lines():
            x_line, y_line = line.get_data()
            if len(y_line) > 0:
                max_y = max(max_y, np.max(y_line))
        for coll in ax.collections:
            for path in coll.get_paths():
                vertices = path.vertices
                max_y = max(max_y, np.max(vertices[:, 1]))
        
        # Position of the significance marker.
        y_bar = max_y * 1.07
        center_sample1 = np.mean(sample1)
        center_sample2 = np.mean(sample2)
        
        # Plot horizontal significance line.
        ax.plot([center_sample1, center_sample2], [y_bar, y_bar],
                color='black', linewidth=1, zorder=2)
        # Plot vertical ticks.
        tick_height = 0.02 * y_bar
        ax.plot([center_sample1, center_sample1], [y_bar, y_bar - tick_height],
                color='black', linewidth=1, zorder=2)
        ax.plot([center_sample2, center_sample2], [y_bar, y_bar - tick_height],
                color='black', linewidth=1, zorder=2)
        # Place significance star.
        ax.text((center_sample1 + center_sample2) / 2, y_bar -.4,
                marker, ha='center', va='bottom', fontsize=tickfsize, zorder=2)

    return p_val

#%%  3 column version (combined TOC plots)
conc    = '100X'
AMPs    = ['Dpt', 'Mtk']  # Dpt will be the first (top) row
norm = False

# List of characteristics and manual labels (modify as needed)
characteristics = ["$R_{nuc:tot}$", "Fold-Change $R_{nuc:tot}$", 
                   "AUC $R_{nuc:tot} T= -30$", "AUC $R_{nuc:tot} T= -60$", 
                   "AUC $R_{nuc:tot} T= 0$"]
# If you want different text than the characteristic names, change char_labels accordingly:
char_labels = ["$R_{nuc:tot}$", "FC $R_{nuc:tot}$", 
                   "T[ -30:Peak ]", "T[ -60:Peak ]", 
                   "T[ 0:Peak ]"]

# Create a figure with 5 rows and 3 columns:
# Column 0 = Dpt KDE, Column 1 = Mtk KDE, Column 2 = Combined ROC for Dpt and Mtk.


fig, axs = plt.subplots(nrows=5, ncols=3, figsize=(4.5, 5), 
                        sharex=False, sharey=False)
scores_dpt   = {}
scores_mtk   = {}
y_true_dpt   = {}
y_true_mtk   = {}


for i, char in enumerate(characteristics):
    # ----------------- Data extraction (Rel or AUC branch) -----------------
    if i in [0,1]:
        if i == 0:  # Rnuc:tot
            rel_peaks = peaksovertime_rel['No norm']['Peaks']
            rel_nopeaks = peaksovertime_rel['No norm']['No peaks']
            xlabel = "$R_{nuc:tot}$"
        elif i == 1:  # FC Rnuc:tot
            rel_peaks = peaksovertime_rel['Norm']['Peaks']
            rel_nopeaks = peaksovertime_rel['Norm']['No peaks']
            xlabel = "FC $R_{nuc:tot}$"
            
        # Filter for the specific concentration:
        rel_peaks_conc = rel_peaks[[conc in s for s in rel_peaks.index]]
        rel_nopeaks_conc = rel_nopeaks[[conc in s for s in rel_nopeaks.index]]
        
        for AMP in AMPs:
            rel_peaks_conc_AMP = rel_peaks_conc[[AMP in s for s in rel_peaks_conc.index]]
            rel_nopeaks_conc_AMP = rel_nopeaks_conc[[AMP in s for s in rel_nopeaks_conc.index]]
            
            if AMP == 'Dpt':
                dpt_peaks = rel_peaks_conc_AMP.stack().values
                dpt_nopeaks = rel_nopeaks_conc_AMP.stack().values
                num_peaks_dpt = len(dpt_peaks)
                num_nopeaks_dpt = len(dpt_nopeaks)
            elif AMP == 'Mtk':
                mtk_peaks = rel_peaks_conc_AMP.stack().values
                mtk_nopeaks = rel_nopeaks_conc_AMP.stack().values
                num_peaks_mtk = len(mtk_peaks)
                num_nopeaks_mtk = len(mtk_nopeaks)
    else:
        if norm:
            AUC_norm_nonorm = peaksovertime_auc['Norm']
            xlabel = "AUC FC $R_{nuc:tot}$"
        else:
            AUC_norm_nonorm = peaksovertime_auc['No norm']
            xlabel = "AUC $R_{nuc:tot}$"
            
        if i == 2:
            auc_time = 30
        elif i == 3:
            auc_time = 60
        elif i == 4:
            auc_time = None
            
        AUC_norm_nonorm_time = AUC_norm_nonorm[auc_time]
        AUC_peaks = AUC_norm_nonorm_time['Peaks']
        AUC_nopeaks = AUC_norm_nonorm_time['No peaks']
        
        AUC_peaks_conc = AUC_peaks[[conc in s for s in AUC_peaks.index]]
        AUC_nopeaks_conc = AUC_nopeaks[[conc in s for s in AUC_nopeaks.index]]
        
        for AMP in AMPs:
            AUC_peaks_conc_AMP = AUC_peaks_conc[[AMP in s for s in AUC_peaks_conc.index]]
            AUC_nopeaks_conc_AMP = AUC_nopeaks_conc[[AMP in s for s in AUC_nopeaks_conc.index]]
            
            if AMP == 'Dpt':
                dpt_peaks = AUC_peaks_conc_AMP.stack().values
                dpt_nopeaks = AUC_nopeaks_conc_AMP.stack().values
            elif AMP == 'Mtk':
                mtk_peaks = AUC_peaks_conc_AMP.stack().values
                mtk_nopeaks = AUC_nopeaks_conc_AMP.stack().values
    
    
    # y_true_dpt = np.concatenate([np.ones(len(dpt_peaks)), np.zeros(len(dpt_nopeaks))])
    # y_true_mtk = np.concatenate([np.ones(len(mtk_peaks)), np.zeros(len(mtk_nopeaks))])
    # scores_dpt[char]= np.concatenate([dpt_peaks, dpt_nopeaks]) 
    # scores_mtk[char] = np.concatenate([mtk_peaks, mtk_nopeaks]) 
    
    # store Dpt
    scores_dpt[char] = np.concatenate([dpt_peaks,   dpt_nopeaks])
    y_true_dpt[char] = np.concatenate([np.ones(len(dpt_peaks)), np.zeros(len(dpt_nopeaks))])
    # store Mtk
    scores_mtk[char] = np.concatenate([mtk_peaks,   mtk_nopeaks])
    y_true_mtk[char] = np.concatenate([np.ones(len(mtk_peaks)), np.zeros(len(mtk_nopeaks))])

    # print('Dpt Peaks'+char+' '+str(np.median(dpt_peaks)))
    # print('Dpt No Peaks'+char+' '+str(np.median(dpt_nopeaks)))
    # print('Mtk Peaks '+char+' '+str(np.median(mtk_peaks)))
    # print('Mtk No Peaks '+char+' '+str(np.median(mtk_nopeaks)))
    
    
    # ----------------- Plotting -----------------
    # ----- Dpt KDE (Column 0) -----
    ax_kde_dpt = axs[i, 0]
    sns.kdeplot(dpt_peaks, ax=ax_kde_dpt, label="Foci+", fill=True, 
                color="green", alpha=0.6, zorder=1)
    sns.kdeplot(dpt_nopeaks, ax=ax_kde_dpt, label="Foci-", fill=True, 
                color="lightgreen", alpha=0.6, zorder=2)
        
    #KS Test
    p_val = add_ks_significance(ax_kde_dpt, dpt_peaks, dpt_nopeaks, tickfsize)
    print(f"Dpt {char} KS p-val = {p_val}")
    
    # ----- Mtk KDE (Column 1) -----
    ax_kde_mtk = axs[i, 1]
    sns.kdeplot(mtk_peaks, ax=ax_kde_mtk, label="Foci+", fill=True, 
                color="blue", alpha=0.6, zorder=1)
    sns.kdeplot(mtk_nopeaks, ax=ax_kde_mtk, label="Foci-", fill=True, 
                color="lightblue", alpha=0.6, zorder=2)
    ax_kde_mtk.set_ylabel(None)

    #KS Test
    p_val = add_ks_significance(ax_kde_mtk, mtk_peaks, mtk_nopeaks, tickfsize)
    print(f"Mtk {char} KS p-val = {p_val}")
    
    # Add characteristic label
    # ax_kde_dpt.text(-0.5, 0.5, char_labels[i], transform=ax_kde_dpt.transAxes,
    #                 rotation=90, va='center', ha='center', fontsize=fsize)
    if i != 1:
        if i ==0:
            ax_kde_mtk.set_title('Mtk', fontsize=fsize*mag)        
            ax_kde_dpt.set_title('Dpt', fontsize=fsize*mag)

            ax_kde_mtk.legend(fontsize=tickfsize*shrink, loc='upper right', 
                              handlelength=0.2, labelspacing=0.2,handletextpad=0.2)
            ax_kde_dpt.legend(fontsize=tickfsize*shrink, loc='upper right', 
                              handlelength=0.2, labelspacing=0.2,handletextpad=0.2)
            ax_kde_mtk.set_xlim([.35, .9])
            ax_kde_mtk.set_xticks(ticks = [.5, .75], labels=['0.5','0.75'], fontsize=tickfsize)
            ax_kde_dpt.set_xlim([.35, .9])
            ax_kde_dpt.set_xticks(ticks = [.5, .75], labels=['0.5','0.75'], fontsize=tickfsize)
            
            ax_kde_mtk.set_ylim([0, 10])
            ax_kde_dpt.set_ylim([0, 10])
        else:
            if i ==4:
                ax_kde_mtk.set_xlim([.2, .85])
                ax_kde_mtk.set_xticks(ticks = [.25, .5, .75], labels=['0.25','0.5','0.75'], fontsize=tickfsize)
                ax_kde_dpt.set_xlim([.2, .85])
                ax_kde_dpt.set_xticks(ticks = [.25, .5, .75], labels=['0.25','0.5','0.75'], fontsize=tickfsize)
            else:
                ax_kde_mtk.set_xlim([.2, .85])
                ax_kde_mtk.set_xticks(ticks = [.25, .5, .75], labels=[], fontsize=tickfsize)
                ax_kde_dpt.set_xlim([.2, .85])
                ax_kde_dpt.set_xticks(ticks = [.25, .5, .75], labels=[], fontsize=tickfsize)
            
            ax_kde_mtk.set_ylim([0, 10])
            ax_kde_dpt.set_ylim([0, 10])
        

    else:
        ax_kde_dpt.set_xlim([.5, 2])
        ax_kde_dpt.set_ylim([0, 4])
        ax_kde_mtk.set_xlim([.5, 2])
        ax_kde_mtk.set_ylim([0, 4])
    
    ax_kde_dpt.tick_params(axis='both', which='major', labelsize=tickfsize)
    ax_kde_mtk.tick_params(axis='both', which='major', labelsize=tickfsize)

   

        # ----- Combined ROC Panel (Column 2) -----
    ax_roc = axs[i, 2]
    # Dpt ROC
    #y_true_dpt = np.concatenate([np.ones(len(dpt_peaks)), np.zeros(len(dpt_nopeaks))])
    # scores_dpt = np.concatenate([dpt_peaks, dpt_nopeaks])
    # fpr_dpt, tpr_dpt, _ = roc_curve(y_true_dpt, scores_dpt)
    
    dpt_scores = scores_dpt[char]
    dpt_labels = y_true_dpt[char]
    fpr_dpt, tpr_dpt, _ = roc_curve(dpt_labels, dpt_scores)
    
    auc_val_dpt = roc_auc_score(dpt_labels, dpt_scores)
    ax_roc.plot(fpr_dpt, tpr_dpt, label=f"Dpt: {auc_val_dpt:.2f}", color="green", lw=1)
    
    # Mtk ROC
    if i ==0:
        ax_roc.set_title('ROC Curves', fontsize=fsize*mag)    
        
    #y_true_mtk = np.concatenate([np.ones(len(mtk_peaks)), np.zeros(len(mtk_nopeaks))])
    # scores_mtk = np.concatenate([mtk_peaks, mtk_nopeaks])
    # fpr_mtk, tpr_mtk, _ = roc_curve(y_true_mtk, scores_mtk)
    
    mtk_scores = scores_mtk[char]
    mtk_labels = y_true_mtk[char]
    fpr_mtk, tpr_mtk, _ = roc_curve(mtk_labels, mtk_scores)
    
    auc_val_mtk = roc_auc_score(mtk_labels, mtk_scores)
    ax_roc.plot(fpr_mtk, tpr_mtk, label=f"Mtk: {auc_val_mtk:.2f}", color="blue", lw=1)
    
    ax_roc.plot([0, 1], [0, 1], 'k--')
    ax_roc.set_xlim([0, 1])
    ax_roc.set_ylim([0, 1])
    
    ax_roc.set_xticks([0, 0.5, 1], labels =['0', '', '1'],  fontsize=tickfsize)
    ax_roc.set_yticks([0, 0.5, 1], labels =['0', '', '1'],  fontsize=tickfsize)
    

    if i==4:
        ax_roc.set_xticklabels(['0', '', '1'],fontsize=tickfsize)
        ax_roc.set_xlabel('FPR', fontsize=fsize)
    else:
        ax_roc.set_xticklabels(['', '', ''],fontsize=tickfsize)
    
    
    # Create the legend with a title "AUC"
    # if i ==0:
    #     leg = ax_roc.legend(title="AUC", fontsize=tickfsize*shrink, loc="lower right", handlelength=0.2)
    #     plt.setp(leg.get_title(), fontsize=tickfsize)
    # else:
    leg = ax_roc.legend(fontsize=tickfsize*shrink, loc="lower right",
                        handlelength=0.2, labelspacing=0.2,handletextpad=0.2)
# #Add AUC suby label
# auc_x= -0.024
# line_auc_right = Line2D([auc_x, auc_x], [0.1, 0.57], transform=fig.transFigure, color='black', linewidth=2)
# fig.add_artist(line_auc_right)
# if norm==True:
#     fig.text(auc_x, 0.335, "FC AUC $R_{nuc:tot}$", ha="center", va="center", rotation=90,
#              fontsize=fsize*mag,
#              bbox=dict(facecolor='white', edgecolor='none', pad=0))
# else:
#     fig.text(auc_x, 0.335, "AUC $R_{nuc:tot}$", ha="center", va="center", rotation=90,
#              fontsize=fsize*mag,
#              bbox=dict(facecolor='white', edgecolor='none', pad=0))

# ----------------- Global Adjustments -----------------
plt.subplots_adjust(wspace=0.3, hspace=0.3)
for ax in axs.flat:  # Works for both 1D and 2D subplot arrays
    ax.tick_params(axis='y', which='both', pad=.01)
    ax.tick_params(axis='x', which='both', pad=.3)
    ax.yaxis.set_label_coords(-.2, 0.5)
    ax.yaxis.label.set_horizontalalignment('center')

# axs[:,2].set_ylabel('TPR', fontsize=fsize)
# axs[:,2].yaxis.set_label_coords(-.2, 0.5)

plt.show()

savename = fig_output + '/Figure 5/' + f'peak_KDE-ROC_3column_{conc}'
#fig.savefig(savename, dpi=1000, bbox_inches='tight')


#%% DeLong test with multiple comparison corrction
# Get all unique pairs of models
pairs = [
    ('$R_{nuc:tot}$', 'Fold-Change $R_{nuc:tot}$'),
    ('$R_{nuc:tot}$', 'AUC $R_{nuc:tot} T= -30$'),
    ('$R_{nuc:tot}$', 'AUC $R_{nuc:tot} T= -60$'),
    ('$R_{nuc:tot}$', 'AUC $R_{nuc:tot} T= 0$'),
    ('AUC $R_{nuc:tot} T= -30$', 'AUC $R_{nuc:tot} T= -60$'),
    ('AUC $R_{nuc:tot} T= -30$', 'AUC $R_{nuc:tot} T= 0$'),
    ('AUC $R_{nuc:tot} T= -60$', 'AUC $R_{nuc:tot} T= 0$'),
]

amps = ['Dpt','Mtk']
all_pvals = []
comparisons = []  # to remember which AMP & pair each p belongs to

for enhan in amps:
    if enhan=='Dpt':
        scores, labels = scores_dpt, y_true_dpt
    else:
        scores, labels = scores_mtk, y_true_mtk

    for model1, model2 in pairs:
        s1 = scores[model1]; s2 = scores[model2]
        y  = labels[model1]
        _, p = Delong_test(y, s1, s2)
        all_pvals.append(p)
        comparisons.append((enhan, model1, model2))

    
from statsmodels.stats.multitest import multipletests

# choose your method: 'bonferroni', 'holm', or 'fdr_bh'
reject, adj_pvals, _, _ = multipletests(all_pvals, alpha=0.05, method='bonferroni')

for (enhan, m1, m2), raw_p, p_corr in zip(comparisons, all_pvals, adj_pvals):
    print(f"{enhan}: {m1} vs {m2}  raw p = {raw_p:.3e},  corrected p = {p_corr:.3e}")

#%% Time trace for diagram, Run after sections for Fig3

fig2size = [1, 5.]  # Adjusted the figure size for two panels

npanels= 6
fig, axs = plt.subplots(npanels, 1, figsize=fig2size, sharex=True,
                       height_ratios = [.8,1,1,1,1,1])  # Create a 2-panel plot

#---------------------------------------------------------- smooth relish data
y_values = df_relish_ratio[cell]
window   = int(len(y_values)/sg_factor_rel) #length of the filter window
y_sgsmooth = savgol_filter(y_values, window_length = window, polyorder = sg_order_rel)
norm_prestim  = sum(y_sgsmooth[0:4])/4
y_sgsmoot_norm = y_sgsmooth/norm_prestim

    
#---------------------------------------------------------- Top panel: sum foci [cell] vs. time
y_values_rb = df_rb[cell]
positive_intensity = y_values_rb > 0

# Plot positive intensity values
axs[0].scatter(interval_forplt_offset[positive_intensity], (y_values_rb[positive_intensity])/1000, color='k', marker='o', s=8)  
axs[0].set_ylabel('Int$_R$$_B$ (AU)', fontsize=fsize)

#set stim time bar
ylim_rb = [-1, max(y_values_rb)/1000 * 1.1]
axs[0].set_ylim(ylim_rb)  # Add some padding to avoid clipping
axs[0].set_xlim([interval_forplt_offset[0]-5, interval_forplt_offset[-20]])
axs[0].set_yticks([10], labels= ['$10^4$'],fontsize=tickfsize)  # Add some padding to avoid clipping
axs[0].set_xticks(ticks = [-60, 0, 200], labels=[])

axs[0].annotate('t Foci', xy=(.35 ,1.05), xycoords='axes fraction', fontsize=tickfsize, color='black', ha='center')
axs[0].annotate('+', xy=(.69 ,1.05), xycoords='axes fraction', fontsize= fsize, color='green', ha='center')
axs[0].annotate('-', xy=(.81 ,1.05), xycoords='axes fraction', fontsize= fsize, color='lightgreen', ha='center')

#----------------------------------------------------------  panel 1: rnuc:tot

axs[1].plot(interval_forplt_offset, y_sgsmooth, marker=None, color='k',linewidth=1)

axs[1].set_ylabel('$R_{nuc:tot}$', fontsize=fsize )
ylim_nonnorm = [.55, .75]
axs[1].set_ylim(ylim_nonnorm)
axs[1].set_yticks(ticks = [.6], labels=['0.6'], fontsize=tickfsize)
axs[1].set_xlim([interval_forplt_offset[0], interval_forplt_offset[-20]])
axs[1].set_xticks(ticks = [-60, 0, 200], labels=[])


#----------------------------------------------------------  panel 2: FCrnuc:tot

axs[2].plot(interval_forplt_offset, y_sgsmoot_norm, marker=None, color='k',linewidth=1)

axs[2].set_ylabel('FC $R_{nuc:tot}$', fontsize=fsize)
ylim_norm = [.95, 1.25]

axs[2].set_ylim(ylim_norm)
axs[2].set_yticks(ticks = [1], labels=['1'], fontsize=tickfsize)
axs[2].set_xlim([interval_forplt_offset[0], interval_forplt_offset[-20]])
axs[2].set_xticks(ticks = [-60, 0, 200], labels=[])


#----------------------------------------------------------  Add timemarks
timemark=10
xval = interval_forplt[timemark]-offset

timemark_neg = 11
xval_neg = interval_forplt[timemark_neg]-offset

axs[0].plot(xval, (y_values_rb[timemark]/1000), marker='o', markersize=4, color='green', alpha=0.9, mfc='none')
axs[0].plot([xval, xval], ylim_rb, color='green', linestyle='-', 
        linewidth=1, alpha=1, zorder=2)
axs[0].plot([xval_neg, xval_neg], ylim_rb, color='lightgreen', linestyle='-', 
        linewidth=1, alpha=1, zorder=2)

axs[1].plot(xval, y_sgsmooth[timemark], marker='o', markersize=4, color='green', alpha=0.9)
axs[1].plot(xval_neg, y_sgsmooth[timemark_neg], marker='o', markersize=4, color='lightgreen', alpha=0.9)

axs[2].plot(xval, y_sgsmoot_norm[timemark], marker='o', markersize=4, color='green', alpha=0.9)
axs[2].plot(xval_neg, y_sgsmoot_norm[timemark_neg], marker='o', markersize=4, color='lightgreen', alpha=0.9)


#----------------------------------------------------------  panel 3: AUC30

axs[3].plot(interval_forplt_offset, y_sgsmooth, marker=None, color='k',linewidth=1)

axs[3].set_ylabel('T[-30:t]' ,fontsize=fsize)

axs[3].set_ylim([.55, .75])
axs[3].set_yticks(ticks = [.6], labels=['0.6'],fontsize=tickfsize)
axs[3].set_xlim([interval_forplt_offset[0], interval_forplt_offset[-20]])
axs[3].set_xticks(ticks = [-60, 0, 200], labels=[])

auc_back= 30 #30min
idx_back= 1 
start_idx = timemark - idx_back
end_idx = timemark + 1  # +1 if you want to include timemark
axs[3].fill_between(interval_forplt_offset[start_idx:end_idx],
                    y_sgsmooth[start_idx:end_idx],
                    color='green', alpha=0.5, hatch='///')


start_idx = timemark_neg - idx_back
end_idx = timemark_neg + 1  # +1 if you want to include timemark
axs[3].fill_between(interval_forplt_offset[start_idx:end_idx],
                    y_sgsmooth[start_idx:end_idx],
                    color='lightgreen', alpha=0.5, hatch='\\\\\\')

#----------------------------------------------------------  panel 4: AUC60

axs[4].plot(interval_forplt_offset, y_sgsmooth, marker=None, color='k',linewidth=1)

axs[4].set_ylabel('T[-60:t]', fontsize=fsize)
axs[4].set_ylim([.55, .75])
axs[4].set_yticks(ticks = [.6], labels=['0.6'], fontsize=tickfsize)
axs[4].set_xlim([interval_forplt_offset[0], interval_forplt_offset[-20]])
axs[4].set_xticks(ticks = [-60, 0, 200], labels=[])

auc_back=60 #60min
idx_back = 2
start_idx = timemark - idx_back
end_idx = timemark + 1  # +1 if you want to include timemark
axs[4].fill_between(interval_forplt_offset[start_idx:end_idx],
                    y_sgsmooth[start_idx:end_idx],
                    color='green', alpha=0.5, hatch='///')

start_idx = timemark_neg - idx_back
end_idx = timemark_neg + 1  # +1 if you want to include timemark
axs[4].fill_between(interval_forplt_offset[start_idx:end_idx],
                    y_sgsmooth[start_idx:end_idx],
                    color='lightgreen', alpha=0.5, hatch='\\\\\\')


#----------------------------------------------------------  panel 5: AUCtotal

axs[5].plot(interval_forplt_offset, y_sgsmooth, marker=None, color='k',linewidth=1)

axs[5].set_ylabel('T[0:t]', fontsize=fsize)
axs[5].set_ylim([.55, .75])
axs[5].set_yticks(ticks = [.6], labels=['0.6'],fontsize=tickfsize)
axs[5].set_xlim([interval_forplt_offset[0], interval_forplt_offset[-20]])
axs[5].set_xticks(ticks = [-60, 0, 200], labels=['-60','','200'],fontsize=tickfsize)

auc_back=120 #60min
idx_back = 6
start_idx = timemark - idx_back
end_idx = timemark + 1  # +1 if you want to include timemark
axs[5].fill_between(interval_forplt_offset[start_idx:end_idx],
                    y_sgsmooth[start_idx:end_idx],
                    color='green', alpha=0.5, hatch='///')

start_idx = timemark_neg - (idx_back+1)
end_idx = timemark_neg + 1  # +1 if you want to include timemark
axs[5].fill_between(interval_forplt_offset[start_idx:end_idx],
                    y_sgsmooth[start_idx:end_idx],
                    color='lightgreen', alpha=0.5, hatch='\\\\\\')


axs[5].set_xlabel('Time (min)', fontsize=fsize)


#add hatch
y_min, y_max = axs[5].get_ylim()
hatchthick = (y_max-y_min)*0.07
prestim  = patches.Rectangle((-60, y_min), 60, hatchthick, edgecolor="black", facecolor="none", hatch='///')
poststim_hatch = patches.Rectangle((interval_forplt_offset[stim], y_min), interval_forplt_offset[-1] - interval_forplt_offset[stim], 
                                   hatchthick, color = "black",)
axs[5].add_patch(prestim)
axs[5].add_patch(poststim_hatch)

# Adding time marks
for time in time_points:
    for axisn in range(npanels):
        axs[axisn].axvline(x=time, color='lightgray', linestyle='-', linewidth = 0.4, alpha=0.5, zorder=0)


for ax in axs.flat:  # Works for both 1D and 2D subplot arrays
    ax.tick_params(axis='y', which='both', pad=.01)
    ax.yaxis.set_label_coords(-.3, 0.5)
    ax.tick_params(axis='x', which='both', pad=.3)
    ax.yaxis.label.set_horizontalalignment('center')
    
#Add AUC suby label
auc_x= -0.4
line_auc_right = Line2D([auc_x, auc_x], [0.12, 0.48], transform=fig.transFigure, color='black', linewidth=1.5)
fig.add_artist(line_auc_right)

fig.text(auc_x, 0.3, " Avg $R_{nuc:tot}$ ", ha="center", va="center", rotation=90,
          fontsize=fsize,
          bbox=dict(facecolor='white', edgecolor='none', pad=0))

plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.show()

savename = fig_output+'Figure 5/'+'HistogramDiagram.png'
fig.savefig(savename, dpi=1000, bbox_inches='tight')

