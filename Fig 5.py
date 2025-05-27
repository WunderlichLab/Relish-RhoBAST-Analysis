# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 13:47:54 2025

@author: noshin
"""
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import os
from scipy.signal import savgol_filter
from scipy.stats import ks_2samp
from sklearn.metrics import roc_curve, roc_auc_score
from MLstatkit.stats import Delong_test
from statsmodels.stats.multitest import multipletests
import pickle
import time
from datetime import timedelta
datetime_str = time.strftime("%m%d%y_%H:%M")

#!!! Update path file!!!
gitdir = 'G:/path/' 
#!!! Update path file!!!

files_import = gitdir+'Figure 5 Files/'
fig_output = gitdir+'Temp Output/Fig 5/'

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

nuc_channel = 0
rel_channel = 1
rb_channel = 2

def import_data(file_path, object_name, file_type = ".pkl"):   
    full_file_path = file_path + "/" + object_name + file_type
    # print(full_file_path)
    with open(full_file_path, "rb") as f:
        object_data = pickle.load(f)
        
    return object_data

#%% Import Data
times                    = import_data(files_import, 'goodcomp7_times')
peaksovertime_rel        = import_data(files_import, 'goodcomp7_rbpeaks_rel_dict')
peaksovertime_auc        = import_data(files_import, 'goodcomp7_rbpeaks_AUCs_dict')


#%% KS Test function
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

#%% Fig 5C/D (combined KDE ROC plots)
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

fig, axs = plt.subplots(nrows=5, ncols=3, figsize=(4.5, 5), 
                        sharex=False, sharey=False)

#Initalize for DeLong's test
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

    # store Dpt
    scores_dpt[char] = np.concatenate([dpt_peaks,   dpt_nopeaks])
    y_true_dpt[char] = np.concatenate([np.ones(len(dpt_peaks)), np.zeros(len(dpt_nopeaks))])
    # store Mtk
    scores_mtk[char] = np.concatenate([mtk_peaks,   mtk_nopeaks])
    y_true_mtk[char] = np.concatenate([np.ones(len(mtk_peaks)), np.zeros(len(mtk_nopeaks))])

    # Option: print median values of distributions
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
    dpt_scores = scores_dpt[char]
    dpt_labels = y_true_dpt[char]
    fpr_dpt, tpr_dpt, _ = roc_curve(dpt_labels, dpt_scores)
    
    auc_val_dpt = roc_auc_score(dpt_labels, dpt_scores)
    ax_roc.plot(fpr_dpt, tpr_dpt, label=f"Dpt: {auc_val_dpt:.2f}", color="green", lw=1)
    
    # Mtk ROC
    if i ==0:
        ax_roc.set_title('ROC Curves', fontsize=fsize*mag)    
        
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
    
    leg = ax_roc.legend(fontsize=tickfsize*shrink, loc="lower right",
                        handlelength=0.2, labelspacing=0.2,handletextpad=0.2)


# ----------------- Global Adjustments -----------------
plt.subplots_adjust(wspace=0.3, hspace=0.3)
for ax in axs.flat:  # Works for both 1D and 2D subplot arrays
    ax.tick_params(axis='y', which='both', pad=.01)
    ax.tick_params(axis='x', which='both', pad=.3)
    ax.yaxis.set_label_coords(-.2, 0.5)
    ax.yaxis.label.set_horizontalalignment('center')

#axs[:,2].set_ylabel('TPR', fontsize=fsize)
#axs[:,2].yaxis.set_label_coords(-.2, 0.5)

plt.show()

#---------------------------------------- Save
figname = 'Peak KDE ROC'
savename = fig_output+'Fig 5CD_'+figname+'.png'#
#fig.savefig(savename, bbox_inches = 'tight', dpi=1000)

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

#%% Fig 5A Import data
#Import dataframe
with open(os.path.join(files_import, 'dict_intensities_ilastikpeaks_meansum_alldatasets_goodcells_nomasks_010925'), 'rb') as handle:
    dict_intensities = pickle.load(handle)

f_dict_intensities = dict_intensities['2024-10-23']['07_Dpt_100X_1_maxZ.tif']
df_relish_ratio = pd.DataFrame({cell: [intensity for time, intensity in values] for cell, values in f_dict_intensities['relish_ratio'].items()})
df_rb = pd.DataFrame({cell: [intensity for time, intensity in values] for cell, values in f_dict_intensities['peak_intensities_sum'].items()})

stim = 4
# Original time points (in minutes)
interval_forplt = np.concatenate([
    np.arange(0, 121, 15),
    np.arange(150, 631, 30),
    np.arange(690, 991, 60)
])
offset = 60
interval_forplt_offset = interval_forplt-offset

celln= 92
cell='Cell '+str(celln)
npanels= 6

#%% Fig 5A category schematics (Run after sections for Fig3)
figsize2 = [1, 5]  # Adjusted the figure size for two panels

fig, axs = plt.subplots(npanels, 1, figsize=figsize2, sharex=True,
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
        linewidth=1, alpha=1, zorder=0)
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
for time in interval_forplt_offset:
    for axisn in range(npanels):
        axs[axisn].axvline(x=time, color='lightgray', linestyle='-', linewidth = 0.4, alpha=0.5, zorder=0)


for ax in axs.flat:  
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

#---------------------------------------- Save
figname = 'Category Schematics'
savename = fig_output+'Fig 5A_'+figname+'.png'#
fig.savefig(savename, bbox_inches = 'tight', dpi=1000)
