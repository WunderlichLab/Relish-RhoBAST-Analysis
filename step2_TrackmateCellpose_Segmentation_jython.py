# ------------------------------------------------------------------------------
# TrackMate + Cellpose Batch Processing Script (Jython)
#  * Author: Noshin Nawar, Boston University (noshin@bu.edu)
#
# Description:
#   - Processes a folder of TIFF series with TrackMate and Cellpose.
#   - Automatically detects and tracks spots in “cyto” or “nuclei” channels.
#   - Saves TrackMate XML and binary mask TIFFs for downstream analysis.
#
# Usage:
#   1. Set channel to segment (line 44).
#   2. Set directory paths (lines 48-49).
#   3. Ensure channel order matches YOUR data (lines 77-82).
#	Current setup assumes Channel 1: Nuclei, Channel 2: Relish, Channel 3: RhoBAST.
#   4. Set path to Cellpose install. Select model for segmentation  (lines 89-94)
#	Can train your own model or built-in cyto/nuclear.
#   5. Manually verify channel order and filters on one file before full batch (lines 107-110).
#	Note: filters (above=true, below=false)
#
# Helpful links:
#	https://forum.image.sc/t/trackmate-labelimgexporter-scripting-v7-11-1/91071/9
#	https://imagej.net/plugins/trackmate/scripting/scripting
#	https://imagej.net/plugins/trackmate/scripting/trackmate-detectors-trackers-keys
# ------------------------------------------------------------------------------

import sys
import os
from ij import IJ
from java.io import File
from datetime import datetime as dt
from fiji.plugin.trackmate import Settings, TrackMate, SelectionModel, Logger
from fiji.plugin.trackmate.cellpose.CellposeSettings import PretrainedModel
from fiji.plugin.trackmate.cellpose import CellposeDetectorFactory
from fiji.plugin.trackmate.tracking.jaqaman import SparseLAPTrackerFactory
from fiji.plugin.trackmate.util import LogRecorder, TMUtils
from fiji.plugin.trackmate.io import TmXmlWriter
from fiji.plugin.trackmate.features import FeatureFilter
from fiji.plugin.trackmate.gui.displaysettings import DisplaySettingsIO, DisplaySettings
from fiji.plugin.trackmate.visualization.hyperstack import HyperStackDisplayer
from fiji.plugin.trackmate.action import LabelImgExporter

# === CONFIGURATION ===
maskSettings  = "15link_nuc8_cyto40/"
channel       = "cyto" #or "nuclei"
showOutput    = False

# Base directories
allData      	= "/path/to/your/data/";          		# Base directory for all datasets
datasetName 	= "2025-01-01_DatasetName/"			# Name of the dataset folder
baseDir       	= os.path.join(allData, datasetName) 		# Base directory for current dataset

inputTifs       = os.path.join(baseDir, "TIF_Split_Series_MaxZ")
outputLabelMask = os.path.join(baseDir, "Trackmate Files", maskSettings, 
                                channel.title() + " Masks")
outputFijiFile  = os.path.join(baseDir, "Trackmate Files", maskSettings, 
                                channel.title() + " Fiji File")

# Create necessary output folders:
for path in [outputLabelMask, outputFijiFile]:
    directory = File(path)
    if not directory.exists():
        directory.mkdirs()
	    
# === PROCESSING FUNCTION ===
def cellposeTrackmateAuto(inputDir, labelMaskDir, fijiFileDir, filename):
    """Run TrackMate+Cellpose on one TIFF and export results."""
    # Prepare logging
    imagePath   = os.path.join(inputTifs, filename)
    imp         = IJ.openImage(imagePath)
    logger      = LogRecorder(Logger.VOID_LOGGER)
    logger.log("TrackMate-Cellpose batch run\n%s\n\n" 
               % dt.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Build settings
    settings    = Settings(imp)
    settings.detectorFactory = CellposeDetectorFactory()
    
    # !!!! TODO !!!!: verify channel order for your data
    if channel == "cyto":
        settings.detectorSettings["TARGET_CHANNEL"]       = 2
        settings.detectorSettings["OPTIONAL_CHANNEL_2"]   = 3
    else:
        settings.detectorSettings["TARGET_CHANNEL"]       = 1
        settings.detectorSettings["OPTIONAL_CHANNEL_2"]   = 0

    # Cellpose model paths — update if needed
    settings.detectorSettings["CELL_DIAMETER"]            = 10.0
    settings.detectorSettings["SIMPLIFY_CONTOURS"]        = True
    
    # !!!! TODO !!!!: choose model for segmentation. Can by custom trained or included  PretrainedModel.NUCLEI or PretrainedModel.CYTO
    settings.detectorSettings["CELLPOSE_PYTHON_FILEPATH"] = "C:/Users/_/_/cellpose/python.exe"
    settings.detectorSettings["CELLPOSE_MODEL_FILEPATH"]  = (			
        "C:/Users/_/.cellpose/models/" +
        ("custommodel_cyto" if channel=="cyto" else "custommodel_nuc") ) 
    settings.detectorSettings["CELLPOSE_MODEL"]           = PretrainedModel.CUSTOM
    settings.detectorSettings["USE_GPU"]                  = True

    # Tracker configuration
    settings.trackerFactory = SparseLAPTrackerFactory()
    settings.trackerSettings = settings.trackerFactory.getDefaultSettings()
    settings.trackerSettings["LINKING_MAX_DISTANCE"]   = 5.0
    settings.trackerSettings["GAP_CLOSING_MAX_DISTANCE"] = 10.0
    settings.trackerSettings["MAX_FRAME_GAP"]            = 15
    settings.initialSpotFilterValue = -1
    settings.addAllAnalyzers()

	# !!!! TODO !!!!: manually verify filters on a single file before batch
    # Spot filters — adjust area thresholds as needed
    if channel == "cyto":
        settings.addSpotFilter(FeatureFilter("AREA", 40, True))
    else:
        settings.addSpotFilter(FeatureFilter("AREA", 8, True))


    # Run TrackMate
    tm = TrackMate(settings)
    tm.getModel().setLogger(logger)
    if not tm.checkInput() or not tm.process():
        print("Error:", tm.getErrorMessage())
        return

    # Save XML
    baseName = os.path.splitext(filename)[0]
    xmlFile  = os.path.join(outputFijiFile, baseName + ".xml")
    writer   = TmXmlWriter(File(xmlFile), logger)
    writer.appendLog(logger.toString())
    writer.appendModel(tm.getModel())
    writer.appendSettings(settings)
    writer.writeToFile()

    # Optionally display
    if showOutput:
        dm = DisplaySettingsIO.readUserDefault()
        dm.spotDisplayedAsRoi = True
        displayer = HyperStackDisplayer(tm.getModel(), SelectionModel(tm.getModel()), imp, dm)
        displayer.render(); displayer.refresh()

    # Export mask and save
    lblExporter = LabelImgExporter()
    lblImg      = lblExporter.createLabelImagePlus(tm, False, True, False, logger)
    lblImg.show()
    IJ.saveAs("Tiff", os.path.join(outputLabelMask, filename))
    imp.close()

# === BATCH RUN ===
for fname in os.listdir(inputTifs):
    if fname.lower().endswith(".tif"):
        cellposeTrackmateAuto(inputTifs, outputLabelMask, outputFijiFile, fname)
