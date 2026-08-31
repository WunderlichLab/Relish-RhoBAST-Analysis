#"""
#Sort Cells by Quality (Jython)
#  * Author: Noshin Nawar, Boston University (noshin@bu.edu)
#
#Description:
#    This script iterates through per‐cell multi‐page TIFF folders generated from step 7
#    in Fiji for manual review, and asks you to classify it as a “Good” or “Bad”
#    cell. Selected files are then moved into “Sorted Cells/<maskSettings>/
#    {Good Cells,Bad Cells}/<FOV>”.
#
#Usage:
#    1. Edit the following variables to match your setup:
#         • allData      – base path to your data
#         • datasetName  – folder for the current dataset
#         • maskSettings – subfolder name under Python/
#    2. Launch Fiji, open the Script Editor, paste in this code as a Jython script.
#    3. Run the script. For each TIFF, choose Yes (good) or No (bad). Cancel to stop.
#"""

from ij import IJ
from ij.gui import NonBlockingGenericDialog
import os
import glob
import shutil

# ------------------------------------------------------------------------------
# Base directories
# ------------------------------------------------------------------------------
allData      = "/path/to/your/data/"              # Base directory for all datasets
datasetName  = "2025-01-01_DatasetName"           # Name of the dataset folder
baseDir      = os.path.join(allData, datasetName)

maskSettings = "15link_nuc8_cyto40"                # CLIJ mask settings subfolder

# Directory containing per-cell TIFF subfolders (one per FOV)
mainDirectory = os.path.join(
    baseDir, "Python", maskSettings, "Seg+Ratio+Peaks TIF"
)

# Gather all FOV subfolders, skipping any “Errors” folder
subdirectories = [
    d for d in os.listdir(mainDirectory)
    if os.path.isdir(os.path.join(mainDirectory, d)) and d != "Errors"
]

for subdir in subdirectories:
    inputDirectory   = os.path.join(mainDirectory, subdir)
    goodCellsDir     = os.path.join(baseDir, "Sorted Cells", maskSettings, "Good Cells", subdir)
    badCellsDir      = os.path.join(baseDir, "Sorted Cells", maskSettings, "Bad Cells",  subdir)

    # Ensure output folders exist
    os.makedirs(goodCellsDir, exist_ok=True)
    os.makedirs(badCellsDir,  exist_ok=True)

    # Iterate over each TIFF in this FOV folder
    for tiffPath in glob.glob(os.path.join(inputDirectory, "*")):
        IJ.open(tiffPath)
        IJ.selectWindow(os.path.basename(tiffPath))
        IJ.doCommand("Start Animation [\\]")  # begin playback
        IJ.getImage().getWindow().setLocation(100, 200)

        # Prompt user to classify cell
        dlg = NonBlockingGenericDialog("Cell Quality")
        dlg.addMessage("Is this a good cell?")
        dlg.enableYesNoCancel("Yes", "No")
        dlg.setLocation(1000, 500)
        dlg.showDialog()
        if dlg.wasCanceled():
            break

        fname   = os.path.basename(tiffPath)
        dstGood = os.path.join(goodCellsDir, fname)
        dstBad  = os.path.join(badCellsDir,  fname)

        if dlg.wasOKed():
            shutil.move(tiffPath, dstGood)
        else:
            shutil.move(tiffPath, dstBad)

        IJ.run("Close All")

    # If the FOV folder is now empty, remove it
    if not os.listdir(inputDirectory):
        os.rmdir(inputDirectory)
