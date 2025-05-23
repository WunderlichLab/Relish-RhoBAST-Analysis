# Relish-RhoBAST Image Analysis

Scripts to analyze the fluorescence microscopy time-series ND2 files resulting in the cell figures and videos in ["Heterogeneous NF-κB activation and enhancer features shape transcription in Drosophila immunity"](https://doi.org/10.1101/2025.05.19.654881).
Included are a collection of ImageJ Macros, Jython/Python scripts, and ilastik classification files to be ran sequentially to process nuclear, Relish, and RhoBAST channel signals.

  - Steps 1-4:  raw data processing scripts for confocal ND2 file z-projection, cell body/nuclei segmentation and tracking, mask interpolation, and cell body/nuclei label-map matching
  - Steps 5-6:  example files for [ilastik's Pixel+Object classificaton](https://www.ilastik.org/documentation/) workflow for RhoBAST foci segmentation. Pixel classificaion file (Step 5) and TIF used for labeling are available upon request (file size exceeds limitation). 
  - Step 7:  Python script for compiling the nuclear Relish fraction and nuclear RhoBAST foci intensity for each cell across each time point.
  - Step 8:  optional ImageJ macro for manual sorting of all cells based on resulting cytoplasm, nuclei, and RhoBAST foci masks (based on the criteria listed in Methods). 

Each step builds on the last, producing data and visualizations suitable for downstream analysis and figure plotting.     

## Prerequisites & Setup

1. **Fiji/ImageJ** enabled with:
   -  CLIJ & CLIJ2 update sites
   -  [TrackMate-Cellpose](https://imagej.net/plugins/trackmate/detectors/trackmate-cellpose)
2. **Python 3.8+** with the following packages:  
   - `numpy`  
   - `scipy`  
   - `scikit-image`  
   - `tifffile`  
   - `pandas`  
   - `microfilm` (for `microplot`)  
   - `matplotlib`
3. **ilastik** [interactive learning and segmentation toolkit](https://www.ilastik.org/)

4. **Directory structure** (customize `allData` and `datasetName` in each script):
```text
/path/to/your/data/                    # <allData>
└── 2025-01-01_DatasetName/            # <datasetName>
    ├── ND2_Split_Series/              # Step 1 Input (raw .ND2 series)
    ├── TIF_Split_Series_MaxZ/         # Step 1 Output (Max-Z projections)
    ├── Trackmate Files/
    │   └── <maskSettings>/
    │       ├── Cyto Fiji File/        # Step 2 Output (Cellpose+TrackMate overlays)
    │       ├── Cyto Masks/            # Step 2 Output (binary masks)
    │       ├── Cyto Matched Masks/    # Step 4 Output (nuclei reassigned → cyto IDs)
    │       ├── Nuclei Fiji File/      # Step 2 Output (Cellpose+TrackMate overlays)
    │       ├── Nuclei Masks/          # Step 2 Output (binary masks)
    │       └── Nuclei Matched Masks/  # Step 4 Output (nuclei reassigned → cyto IDs)
    ├── Python/
    │   └── <maskSettings>/
    │       ├── Interpolated Masks_fullinterp/
    │       │   ├── Cyto/               # Step 3 Output (interpolated cell masks)
    │       │   └── Nuclei/             # Step 3 Output (interpolated cell masks)
    │       └── IntensitiesDF/
    │           ├── dictIntensities_{datetimeStr}.pkl
    │           └── dictIntensitiesNomask_{datetimeStr}.pkl
    │                                   # Step 7 Output (pickle intensity dicts)
    ├── ilastik Outputs/
    │   ├── Probabilities/              # Step 5 Output (ilastik Pixel Classification)
    │   └── Aptamer Masks/              # Step 6 Output (ilastik Object Classification)
    └── Sorted Cells/                   # Step 8 Output (manual QC)
        └── <maskSettings>/
            ├── Good Cells/
            └── Bad Cells/
