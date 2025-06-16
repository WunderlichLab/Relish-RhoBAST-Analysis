
# ------------------------------------------------------------------------------
# 3D Mask Interpolation Script
#  * Author: Noshin Nawar, Boston University (noshin@bu.edu)
# 
# Description:
#   - Loads binary mask TIFF stacks (x, y, t) for “cyto” and “nuclei” objects.
#   - Finds z-slices where labels disappear and interpolates them by combining
#     the nearest nonmissing slices (morphologically dilating/eroding as needed).
#   - Saves out uint16 TIFF stacks with metadata for resolution & voxel size.
#
# Usage:
#   1. Set`maskSettings` and directory paths (lines 25-27).
#   2. Edit `resolution` and `voxSize` to match your imaging calibration.
#   3. Run—interpolated masks will appear under
#      `baseDir/Python/<maskSettings>/Interpolated Masks_fullinterp/{Cyto,Nuclei}`.
# ------------------------------------------------------------------------------

import os
import numpy as np
import tifffile
from scipy.ndimage import binary_dilation, binary_erosion

# === CONFIGURATION ===
maskSettings    = "15link_nuc8_cyto40"
allData      	= "/path/to/your/data/"          		# Base directory for all datasets
datasetName 	= "2025-01-01_DatasetName/"				# Name of the dataset folder
baseDir       	= os.path.join(allData, datasetName) 	# Base directory for current dataset

# Physical scaling (edit as needed)
resolution      = 0.1      # pixels per micron
voxSize         = 1.0      # micron³ per voxel

# Input/output paths
inputMaxZTifs    = os.path.join(baseDir, "TIF_Split_Series_MaxZ")
cytoMasksDir     = os.path.join(baseDir, "Trackmate Files", maskSettings, "Cyto Masks")
nucleiMasksDir   = os.path.join(baseDir, "Trackmate Files", maskSettings, "Nuclei Masks")
outputInterpCyto = os.path.join(baseDir, "Python", maskSettings,
                                "Interpolated Masks_fullinterp", "Cyto")
outputInterpNuc  = os.path.join(baseDir, "Python", maskSettings,
                                "Interpolated Masks_fullinterp", "Nuclei")

# Create output directories if they don’t exist
for folder in (outputInterpCyto, outputInterpNuc):
    os.makedirs(folder, exist_ok=True)

#%%
def interpolate_x1(mask, export_path, resolution, vox_size):
    """
    Interpolate missing labels across z-slices in a 3D mask stack.

    Parameters
    ----------
    mask         : numpy.ndarray, shape (Z, Y, X), integer labels
    export_path  : str, full TIFF path to save interpolated mask
    resolution   : float, pixels per micron
    vox_size     : float, micron³ per voxel

    Returns
    -------
    img_interp   : numpy.ndarray, uint16, same shape as `mask`
    """
    img = mask.copy()
    labels = np.unique(img)
    labels = labels[labels != 0]  # drop background
    missing = {}
    num_slices = img.shape[0]

    # Record which slices each label is missing from
    for z in range(num_slices):
        present = np.unique(img[z])
        for lbl in labels:
            if lbl not in present:
                missing.setdefault(lbl, []).append(z)

    all_slices = list(range(num_slices))
    # For each missing slice, interpolate from two nearest non-missing
    for lbl, zs in missing.items():
        for z in zs:
            # find two closest z not in zs
            nearest = sorted(all_slices, key=lambda x: abs(x - z))
            nearest = [s for s in nearest if s not in zs][:2]
            if len(nearest) < 2:
                continue

            before = (img[nearest[0]] == lbl).astype(np.uint8)
            after  = (img[nearest[1]] == lbl).astype(np.uint8)
            diff = before.sum() - after.sum()

            # balance areas
            if diff > 0:
                before = binary_erosion(before)
            elif diff < 0:
                after  = binary_dilation(after)

            combo = np.logical_or(before, after)
            img[z][combo] = lbl

    img = img.astype(np.uint16)
    metadata = {"Resolution": resolution, "Voxel size": vox_size}
    tifffile.imsave(export_path, img, metadata=metadata)
    return img


#%% # === BATCH PROCESSING ===
fileNames = [f for f in os.listdir(cytoMasksDir) if f.lower().endswith(".tif")]

for fname in fileNames:
    # Cyto
    cyto_path = os.path.join(cytoMasksDir, fname)
    mask_cyto = tifffile.imread(cyto_path)
    out_cyto = os.path.join(outputInterpCyto, fname)
    interpolate_x1(mask_cyto, out_cyto, resolution, voxSize)

    # Nuclei
    nuc_path = os.path.join(nucleiMasksDir, fname)
    mask_nuc = tifffile.imread(nuc_path)
    out_nuc = os.path.join(outputInterpNuc, fname)
    interpolate_x1(mask_nuc, out_nuc, resolution, voxSize)
