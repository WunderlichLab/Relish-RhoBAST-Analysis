/* ------------------------------------------------------------------------------
 * ImageJ Macro: Max Intensity Z-Projection with Per-Channel Brightness Adjustment
 * Author: Noshin Nawar, Boston University (noshin@bu.edu)
 * 
 * Description:
 * This script processes all ND2 files in a specified dataset by performing a maximum
 * intensity Z-projection, applying user-defined brightness settings for each channel,
 * and saving the output as TIFF files for downstream analysis.
 *
 * Instructions:
 * 1. Set 'baseDir' to the directory where your dataset folders reside.
 * 2. Set 'datasetName' to the specific folder name for your dataset.
 * 3. Ensure ND2 files are placed in the 'ND2_Split_Series' subfolder inside the dataset.
 * 4. Adjust 'nChannels', 'channelMins', and 'channelMaxs' arrays for your data.
 * 5. Run this macro in ImageJ or Fiji.
 * ------------------------------------------------------------------------------
 */

// === User configuration ===
allData       = "/path/to/your/data/";          // Base directory for all datasets
datasetName = "2025-01-01_DatasetName/"			// Name of the dataset folder
baseDir       = allData + datasetName			// Base directory for current dataset

// Number of channels in your ND2 images
nChannels = 3;

// Brightness settings: one min and max per channel. Keep consistent across whole dataset
// Example: channel 1 -> min 32, max 4095; channel 2 -> min 45, max 2733; channel 3 -> min 33, max 4095
channelMins = newArray(32, 45, 33);
channelMaxs = newArray(4095, 2733, 4095);

inputDir  = baseDir + datasetName + "ND2_Split_Series/";
outputDir = baseDir + datasetName + "TIF_Split_Series_MaxZ/";

// Create output directory if it doesn't exist
File.makeDirectory(outputDir);

// === End of configuration ===


// Apply brightness settings for each channel
function applyBrightnessSettings() {
    for (c = 1; c <= nChannels; c++) {
        Stack.setChannel(c);
        setMinAndMax(channelMins[c-1], channelMaxs[c-1]);
    }
}

// Perform max intensity Z-projection, apply brightness, and save
function processFile(inputPath, outputPath, filename) {
    // Open the ND2 file
    open(inputPath + filename);
    
    // Derive a base name without extension
    nameNoExt = substring(filename, 0, lengthOf(filename) - 4);
    
    // Run max intensity Z-projection on all slices
    run("Z Project...", "projection=[Max Intensity] all");
    
    // Adjust brightness for each channel
    applyBrightnessSettings();
    
    // Save the projection as a TIFF
    saveAs("Tiff", outputPath + nameNoExt + "_maxZ.tif");
    
    // Close images to free memory
    close();
    close();
}

// Process all files in the input directory
fileList = getFileList(inputDir);
for (i = 0; i < fileList.length; i++) {
    processFile(inputDir, outputDir, fileList[i]);
}
