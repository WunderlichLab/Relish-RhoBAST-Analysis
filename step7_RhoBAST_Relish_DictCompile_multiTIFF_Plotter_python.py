# ------------------------------------------------------------------------------
# Intensity Analysis & Plotting Script
#  * Author: Noshin Nawar, Boston University (noshin@bu.edu)
#
# Description:
#   - Computes aptamer foci (RhoBAST) and Relish intensities per cell over time.
#   - Saves intensity dictionaries to pickle files.
#   - Generates per‐cell multi‐page TIFFs showing segmentation, Relish ratio, and peak signals.
#
# Usage:
#   1. Set `maskSettings` and directory paths (lines 31-33).
#   2. Adjust calibration (`resolution`, channels, `stimFrame`, lines 36-42) for your data.
#   3. Set `useSavedDict = True` to load existing pickles instead of recomputing.
# ------------------------------------------------------------------------------


import os
import time
import pickle
import gc

import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
import tifffile as tiff
from microfilm.microplot import microshow
import matplotlib.pyplot as plt

# === CONFIGURATION ===
maskSettings    = "15link_nuc8_cyto40"
allData      	= "/path/to/your/data/"          		# Base directory for all datasets
datasetName 	= "2025-01-01_DatasetName/"				# Name of the dataset folder
baseDir       	= os.path.join(allData, datasetName) 	# Base directory for current dataset

resolution     = 3.4756            # pixels per µm
unitsPerPix    = 1.0 / resolution

nucChannel     = 0
relChannel     = 1
aptamerChannel = 2
stimFrame      = 4                 # injection frame index

intervalForPlot = np.concatenate([
    np.arange(0, 121, 15),
    np.arange(150, 631, 30),
    np.arange(690, 991, 60),
])
intervalHrs     = [
    str(pd.Timedelta(minutes=int(t))).split()[-1][:-3]
    for t in intervalForPlot
]

useSavedDict   = False
datetimeStr    = time.strftime("%y%m%d_%H%M")
dictFilename   = f'dictIntensities_{datetimeStr}.pkl'
dictNomaskFilename = f'dictIntensitiesNomask_{datetimeStr}.pkl'

tifsDir       = os.path.join(baseDir, 'TIF_Split_Series_MaxZ')
cytoMasksDir  = os.path.join(baseDir, 'Trackmate Files', maskSettings, 'Cyto Matched Masks')
nucMasksDir   = os.path.join(baseDir, 'Trackmate Files', maskSettings, 'Nuclei Matched Masks')

#%% 
def calculateMeanIntensity(mask, intensityImage):
    props = regionprops(mask.astype(int), intensity_image=intensityImage)
    return props[0].mean_intensity if props else 0


def getLastValueOrDefault(intensities, key, default=0):
    return intensities[key][-1][1] if intensities[key] else default


def makeIntensitiesDict(rootDir):
    """
    Computes and saves intensity dictionaries for all .tif in rootDir.
    """
    exportDir = os.path.join(rootDir, 'Python', maskSettings, 'IntensitiesDF')
    os.makedirs(exportDir, exist_ok=True)

    tifsDir       = os.path.join(rootDir, 'TIF_Split_Series_MaxZ')
    cytoMasksDir  = os.path.join(rootDir, 'Trackmate Files', maskSettings, 'Cyto Matched Masks')
    nucMasksDir   = os.path.join(rootDir, 'Trackmate Files', maskSettings, 'Nuclei Matched Masks')
    peakMasksDir  = os.path.join(rootDir, 'ilastik Outputs', 'Aptamer Masks')

    fnames = [f for f in os.listdir(tifsDir) if f.lower().endswith('.tif')]

    dictInt       = {}
    dictIntNomask = {}
    multiFoci     = {}
    missingCells  = {}

    for fname in fnames:
        tsStack      = tiff.imread(os.path.join(tifsDir, fname))
        cytoStack    = tiff.imread(os.path.join(cytoMasksDir, fname))
        nucStack     = tiff.imread(os.path.join(nucMasksDir, fname))
        peakStack    = tiff.imread(os.path.join(peakMasksDir, fname[:-4] + '_Object Identities.tiff'))

        nFrames = tsStack.shape[0]
        dictInt[fname] = {}
        dictIntNomask[fname] = {}

        # Initialize peak storage
        peaksMask4D = np.zeros_like(nucStack)
        peakInts    = {}
        peakSums    = {}

        # --- Compute RhoBAST peaks per frame/cell ---
        for t in range(nFrames):
            frame       = tsStack[t]
            cytoMask    = (cytoStack[t] > 0).astype(np.uint16)
            nucMask     = (nucStack[t] > 0).astype(np.uint16)
            rawPeakMask = peakStack[t]

            # create binary peak mask per nucleus
            binPeak     = (rawPeakMask > 0).astype(np.uint16)
            peaksCell   = binPeak * nucMask
            peaksMask4D[t] = peaksCell

            # loop cells
            maxCell = int(cytoStack[t].max())
            for cellIdx in range(1, maxCell + 1):
                key = f'Cell {cellIdx}'
                thisPeakMask = (peaksCell == cellIdx).astype(np.uint16)
                if thisPeakMask.any():
                    lbl, num_feats = label(thisPeakMask, return_num=True)
                    if num_feats == 1:
                        props = regionprops(thisPeakMask, intensity_image=frame[aptamerChannel] * (cytoMask | nucMask))
                        meanInt = props[0].mean_intensity
                        totInt  = props[0].intensity_image.sum()
                        peakInts.setdefault(key, []).append([intervalForPlot[t], meanInt])
                        peakSums.setdefault(key, []).append([intervalForPlot[t], totInt])
                    else:
                        multiFoci.setdefault(fname, {}).setdefault(key, []).append(t)

        # --- Fill missing frames with zeros ---
        for key in list(peakInts) + [f'Cell {i}' for i in range(1, int(cytoStack.max())+1)]:
            times = [pt[0] for pt in peakInts.get(key, [])]
            for tmin in intervalForPlot:
                if tmin not in times:
                    peakInts.setdefault(key, []).append([tmin, 0])
                    peakSums.setdefault(key, []).append([tmin, 0])
            peakInts[key].sort(key=lambda x: x[0])
            peakSums[key].sort(key=lambda x: x[0])

        # Store
        dictInt[fname]['peak_intensities_sum'] = peakSums
        dictInt[fname]['peaks_mask']          = peaksMask4D
        if fname in multiFoci:
            dictInt[fname]['cells_multinuc'] = multiFoci[fname]

        # --- Relish quantification ---
        relishVals = {'relish_nuclear': {}, 'relish_cyto': {}, 'relish_ratio': {}}
        for t in range(nFrames):
            frameRel = tsStack[t, relChannel]
            for cellIdx in range(1, int(cytoStack.max())+1):
                key = f'Cell {cellIdx}'
                cytoMask = (cytoStack[t] == cellIdx).astype(np.uint16)
                nucMask  = (nucStack[t] == cellIdx).astype(np.uint16)

                # init lists
                for d in relishVals.values():
                    d.setdefault(key, [])

                if cytoMask.any() and nucMask.any():
                    meanCyto = calculateMeanIntensity(cytoMask, frameRel)
                    meanNuc  = calculateMeanIntensity(nucMask, frameRel)
                    ratio    = meanNuc / (meanNuc + meanCyto)
                else:
                    meanCyto = getLastValueOrDefault(relishVals['relish_cyto'], key)
                    meanNuc  = getLastValueOrDefault(relishVals['relish_nuclear'], key)
                    ratio    = (meanNuc / (meanNuc + meanCyto)) if (meanNuc + meanCyto) else 0
                    missingCells.setdefault(fname, []).append(key)

                relishVals['relish_cyto'][key].append([float(intervalForPlot[t]), meanCyto])
                relishVals['relish_nuclear'][key].append([float(intervalForPlot[t]), meanNuc])
                relishVals['relish_ratio'][key].append([float(intervalForPlot[t]),     ratio])

        dictInt[fname]['relish_ratio'] = relishVals['relish_ratio']

    # Save
    with open(os.path.join(exportDir, dictFilename), 'wb') as f:
        pickle.dump(dictInt, f, protocol=pickle.HIGHEST_PROTOCOL)

    gc.collect()
    return dictInt


def plotFname(
    fname,
    dictInt,
    tifsDir,
    cytoMasksDir,
    nucMasksDir
):
    """
    For a given FOV, produces one multi-page TIFF per cell with 5 subplots:
      1) Overlay (Relish channels + masks)
      2) Single-channel Relish
      3) Relish ratio over time
      4) RhoBAST + peak overlay
      5) Peak intensity timeline
    """
    fData       = dictInt[fname]
    dfRelRatio  = pd.DataFrame({
        cell: [v for _, v in vals]
        for cell, vals in fData['relish_ratio'].items()
    })
    peaksSums   = fData['peak_intensities_sum']
    peaksMask4D = fData['peaks_mask']

    # Memory-map stacks
    ts_stack    = tiff.memmap(os.path.join(tifsDir,      fname))
    cyto_stack  = tiff.memmap(os.path.join(cytoMasksDir, fname))
    nuc_stack   = tiff.memmap(os.path.join(nucMasksDir,  fname))
    n_frames    = ts_stack.shape[0]

    # Output folder
    out_root = os.path.join(
        os.path.dirname(tifsDir),
        'Python', maskSettings, 'Seg+Ratio+Peaks TIF', fname[:-4]
    )
    os.makedirs(out_root, exist_ok=True)

    for cell in dfRelRatio.columns:
        idx     = int(cell.split()[1])
        cell_dir = os.path.join(out_root, cell)
        os.makedirs(cell_dir, exist_ok=True)
        save_path = os.path.join(cell_dir, f"{cell}.tif")

        # Compute bounding box once per cell
        ymins, ymas, xmins, xmas = [], [], [], []
        ymins_n, ymas_n, xmins_n, xmas_n = [], [], [], []

        for t in range(n_frames):
            cm = (cyto_stack[t] == idx)
            nm = (nuc_stack[t]  == idx)
            if not cm.any(): 
                continue
            miny_c, minx_c, maxy_c, maxx_c = regionprops(cm.astype(int))[0].bbox
            miny_n, minx_n, maxy_n, maxx_n = regionprops(nm.astype(int))[0].bbox

            ymins.append(min(miny_c, miny_n))
            ymas.append(max(maxy_c, maxy_n))
            xmins.append(min(minx_c, minx_n))
            xmas.append(max(maxx_c, maxx_n))

            ymins_n.append(miny_n)
            ymas_n.append(maxy_n)
            xmins_n.append(minx_n)
            xmas_n.append(maxx_n)

        ymin, ymax = min(ymins), max(ymas)
        xmin, xmax = min(xmins), max(xmas)
        ymin_n, ymax_n = min(ymins_n), max(ymas_n)
        xmin_n, xmax_n = min(xmins_n), max(xmas_n)
        buffer = 0

        fig = plt.figure(figsize=(10, 8))
        widths = [1, 1.5]

        with tiff.TiffWriter(save_path, bigtiff=True) as writer:
            for t in range(n_frames):
                fig.clf()
                gs = fig.add_gridspec(2, 2, width_ratios=widths)

                frame4d = ts_stack[t]
                cyto_frame = (cyto_stack[t] == idx)
                nuc_frame  = (nuc_stack[t] == idx)

                # Crop intensity volumes
                cropped_cell = frame4d[:, ymin:ymax, xmin:xmax].copy()
                cropped_nuc  = frame4d[:, ymin_n - buffer:ymax_n + buffer,
                                          xmin_n - buffer:xmax_n + buffer].copy()

                # Crop masks
                cropped_cyto_mask       = cyto_frame[ymin:ymax, xmin:xmax].copy()
                cropped_nuc_mask        = nuc_frame[ymin:ymax, xmin:xmax].copy()
                cropped_nuc_mask_close  = nuc_frame[ymin_n - buffer:ymax_n + buffer,
                                                    xmin_n - buffer:xmax_n + buffer].copy()
                cropped_peak_mask       = peaksMask4D[t,
                                                      ymin_n - buffer:ymax_n + buffer,
                                                      xmin_n - buffer:xmax_n + buffer].copy()

                # Panel 1: Relish overlay
                ax1 = fig.add_subplot(gs[0,0])
                microshow(
                    images=cropped_cell[[nucChannel, relChannel]],
                    cmaps=['pure_blue','pure_red'],
                    fig_scaling=5,
                    label_text='Relish', label_color='red',
                    unit='um', scalebar_unit_per_pix=unitsPerPix,
                    scalebar_size_in_units=10, scalebar_color='white',
                    scalebar_font_size=10, ax=ax1
                )
                ax1.contour(cropped_cyto_mask, colors='orange', alpha=0.5, linewidths=1.3)
                ax1.contour(cropped_nuc_mask,  colors='blue',   alpha=0.5, linewidths=1.3)
                ax1.axis('off')

                # Panel 2: Single-channel Relish
                ax2 = fig.add_subplot(gs[0,1])
                microshow(
                    images=cropped_cell[relChannel],
                    fig_scaling=5, rescale_type='limits', limits=[0,1200],
                    cmaps=['pure_red'], ax=ax2
                )
                ax2.text(0.5,1,"Halotag-Relish\n+JfX650",fontsize=10,
                         ha='center',va='bottom',transform=ax2.transAxes)
                ax2.contour(cropped_cyto_mask, colors='orange', alpha=0.5, linewidths=1.3)
                ax2.contour(cropped_nuc_mask,  colors='blue',   alpha=0.5, linewidths=1.3)
                ax2.axis('off')

                # Panel 3: Relish ratio timeline
                ax3 = fig.add_subplot(gs[1,0])
                ax3.plot(intervalForPlot, dfRelRatio[cell],
                         marker='o', markersize=3.5, linewidth=1.3)
                ax3.set_xlim(intervalForPlot[0]-10, intervalForPlot[-1]+10)
                ax3.set_xlabel('Time (mins)')
                ax3.set_ylim(
                    np.nanmin(dfRelRatio[cell]) - 0.01,
                    np.nanmax(dfRelRatio[cell]) + 0.01
                )
                ax3.text(0.5,1,"Normalized Nuclear:Total Relish",
                         fontsize=14,ha='center',va='bottom',transform=ax3.transAxes)
                ax3.axvline(intervalForPlot[t], color='green')

                # Panel 4: RhoBAST + peaks
                ax4 = fig.add_subplot(gs[1,0])
                microshow(
                    images=cropped_nuc[aptamerChannel],
                    fig_scaling=5, rescale_type='limits', limits=[0,2500],
                    cmaps=['pure_yellow'], label_text='RhoBAST',
                    label_color='yellow', ax=ax4
                )
                ax4.contour(cropped_nuc_mask_close, colors='blue', alpha=0.5, linewidths=0.5)
                ax4.contour(cropped_peak_mask,      colors='blue', alpha=0.4, linewidths=0.5)
                ax4.axis('off')

                # Panel 5: Peak intensity timeline
                ax5 = fig.add_subplot(gs[1,1])
                for tp in intervalForPlot:
                    ax5.axvline(tp, color='gray', linewidth=0.5, alpha=0.5)
                if cell in peaksSums:
                    for tp, val in peaksSums[cell]:
                        if val > 0:
                            ax5.scatter(tp, val, marker='*', s=20, c='black')
                else:
                    ax5.scatter(intervalForPlot, np.zeros(n_frames), marker='None')
                ax5.set_xlim(intervalForPlot[0]-10, intervalForPlot[-1]+10)
                ax5.set_xlabel('Time (mins)')
                ax5.text(0.5,1,"RhoBAST Peak Sum Intensity (AU)",
                         fontsize=14,ha='center',va='bottom',transform=ax5.transAxes)
                ax5.axvline(intervalForPlot[t], color='green')
                ax5.axvline(intervalForPlot[stimFrame], color='red', linestyle='dashed')

                plt.tight_layout()
                fig.canvas.draw()
                buf, (w,h) = fig.canvas.print_to_buffer()
                img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[..., :3]
                writer.write(img, photometric='rgb', compression='zlib')

        plt.close(fig)
        gc.collect()



#%% # === MAIN LOOP ===
# If previously compiled RhoBAST-Relish dictionary, load. Otherwise, make.
if useSavedDict:
    with open(os.path.join(baseDir, 'Python', maskSettings, dictFilename), 'rb') as f:
        dictInt = pickle.load(f)
else:
    dictInt = makeIntensitiesDict(baseDir)

# Make multiTIFFs of each individual cell to sort through
for fname in dictInt:
    plotFname(fname, dictInt, tifsDir, cytoMasksDir, nucMasksDir)

gc.collect()
