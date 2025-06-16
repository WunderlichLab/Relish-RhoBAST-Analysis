/* ------------------------------------------------------------------------------
 * Match nuclei to cell-body labels (ImageJ Macro using CLIJ2)
 * * Authors:  Bram van den Broek, The Netherlands Cancer Institute. (b.vd.broek@nki.nl), 
 *  adapted by Noshin Nawar, Boston University (noshin@bu.edu)
 *  Please cite the CLIJ article if you use this macro in a publication:
 *  https://www.nature.com/articles/s41592-019-0650-1.
 *
 * Description:
 *   - Assign each nucleus label to the cell with the largest overlap.
 *   - Remove nuclei without a cell body.
 *   - Generate:
 *       • Reassigned nuclei labelmap (labels match cell IDs)
 *       • Cytoplasm labelmap (cell bodies minus nuclei)
 *       • Empty-cell labelmap (cells with no nuclei)
 *       • Masked empty-cell labelmap (excludes overlapping nuclei)
 *
 * Requirements:
 *   - Fiji with CLIJ & CLIJ2 update sites enabled.
 *
 * Usage:
 *   1. Set `maskSettings` and directory paths (lines 26-28).
 *   2. Ensure cyto & nuclei mask filenames match.
 *   3. Run the macro.
 * ------------------------------------------------------------------------------ */

maskSettings    = "15link_nuc8_cyto40";
allData      	= "/path/to/your/data/";          		// Base directory for all datasets
datasetName 	= "2025-01-01_DatasetName/"				// Name of the dataset folder
baseDir       	= os.path.join(allData, datasetName) 	// Base directory for current dataset


inputCytomask      = baseDir + "/Python/" + maskSettings + "/Interpolated Masks_fullinterp/Cyto/";
inputNucMask       = baseDir + "/Python/" + maskSettings + "/Interpolated Masks_fullinterp/Nuclei/";
outputMatchedNuc   = baseDir + "/Trackmate Files/" + maskSettings + "/Nuclei Matched Masks/";
outputMatchedCyto  = baseDir + "/Trackmate Files/" + maskSettings + "/Cyto Matched Masks/";

// Ensure output folders exist
File.makeDirectory(outputMatchedNuc);
File.makeDirectory(outputMatchedCyto);

// Main processing function
function matchCellNucLabelMaps(cytoPath, nucPath, outNucDir, outCytoDir, filename) {
    Ext.CLIJ2_clear();
    open(cytoPath + filename);          labelmapCells   = getTitle();
    open(nucPath + filename);           labelmapNuclei  = getTitle();
    Ext.CLIJ2_push(labelmapCells);
    Ext.CLIJ2_push(labelmapNuclei);

    // Get counts
    Ext.CLIJ2_getMaximumOfAllPixels(labelmapCells,  nrCells);
    Ext.CLIJ2_getMaximumOfAllPixels(labelmapNuclei, nrNuclei);

    // Compute overlap & reassign nuclei → cells
    Ext.CLIJ2_generateJaccardIndexMatrix(labelmapCells, labelmapNuclei, jaccardMatrix);
    Ext.CLIJ2_transposeXZ(jaccardMatrix, jaccardMatrixT);
    Ext.CLIJ2_argMaximumZProjection(jaccardMatrixT, maxOverlap, idxMax);
    Ext.CLIJ2_transposeXY(idxMax, idxMaxT);
    reassignedNuclei = "reassigned_nuclei";
    Ext.CLIJ2_replaceIntensities(labelmapNuclei, idxMaxT, reassignedNuclei);
    Ext.CLIJ2_pull(reassignedNuclei);
    Ext.CLIJ2_release(jaccardMatrix, jaccardMatrixT, idxMax, idxMaxT, maxOverlap);

    // Identify cells with no nuclei
    selectWindow(reassignedNuclei);
    resetMinAndMax();
    getRawStatistics(nPx, mean, min, max, std, hist);
    emptyArray = newArray(nrCells);
    for (i = 0; i < nrCells; i++) {
        emptyArray[i] = (i <= max && hist[i] == 0) ? i : 0;
    }
    Ext.CLIJ2_pushArray("emptyArr", emptyArray, nrCells, 1, 1);
    emptyCells    = "emptyCells";
    Ext.CLIJ2_replaceIntensities(labelmapCells, "emptyArr", emptyCells);
    Ext.CLIJ2_pull(emptyCells);

    // Build cytoplasm map (cell minus nucleus)
    cytoplasm = "cytoplasm";
    Ext.CLIJ2_subtractImages(labelmapCells, reassignedNuclei, cytoplasm);
    Ext.CLIJ2_pull(cytoplasm);

    // Mask empty cells overlapping any nuclei
    Ext.CLIJ2_binaryIntersection(reassignedNuclei, emptyCells, overlapMask);
    Ext.CLIJ2_binaryNot(overlapMask, invertMask);
    maskedEmpty = "emptyMasked";
    Ext.CLIJ2_mask(emptyCells, invertMask, maskedEmpty);
    Ext.CLIJ2_pull(maskedEmpty);

    // Save results
    selectWindow(cytoplasm);
    saveAs("Tiff", outCytoDir + filename);
    selectWindow(reassignedNuclei);
    saveAs("Tiff", outNucDir  + filename);

    Ext.CLIJ2_clear();
    close("*");
}

// Batch over all cyto masks
files = getFileList(inputCytomask);
for (i = 0; i < files.length; i++) {
    matchCellNucLabelMaps(
        inputCytomask, inputNucMask,
        outputMatchedNuc, outputMatchedCyto,
        files[i]
    );
}
