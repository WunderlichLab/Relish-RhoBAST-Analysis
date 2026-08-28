# Relish-RhoBAST Analysis
 This repository contains all analysis scripts for data presented in ["Heterogeneous NF-κB activation and enhancer features shape transcription in Drosophila immunity"](https://www.cell.com/biophysj/fulltext/S0006-3495%2826%2900013-5).

## [Confocal Image Analysis Pipeline](https://github.com/WunderlichLab/Relish-RhoBAST-Analysis/tree/Relish-RhoBAST-Image-Analysis)

Scripts to analyze the fluorescence microscopy time-series ND2 files resulting in the cell figures and videos in ["Heterogeneous NF-κB activation and enhancer features shape transcription in Drosophila immunity"](https://www.cell.com/biophysj/fulltext/S0006-3495%2826%2900013-5).
Included are a collection of ImageJ Macros, Jython/Python scripts, and ilastik classification files to be ran sequentially to process nuclear, Relish, and RhoBAST channel signals.

  - Steps 1-4:  raw data processing scripts for confocal ND2 file z-projection, cell body/nuclei segmentation and tracking, mask interpolation, and cell body/nuclei label-map matching. Cellpose custom masks for nuclear and cytoplasmic segmentation are included.
  - Steps 5-6:  example files for [ilastik's Pixel+Object classificaton](https://www.ilastik.org/documentation/) workflow for RhoBAST foci segmentation. Pixel classificaion file (Step 5) and TIF used for labeling are available upon request (file size exceeds limitation). 
  - Step 7:  Python script for compiling the nuclear Relish fraction and nuclear RhoBAST foci intensity for each cell across each time point.
  - Step 8:  optional ImageJ macro for easy visualization and interactive sorting of all cells based on overlaid masks for cytoplasm, nuclei, and RhoBAST foci (based on the criteria listed in Methods). 

Each step builds on the last, producing data and visualizations suitable for downstream analysis and figure plotting.     

**Prerequisites & Setup:**

1. **Fiji/ImageJ** enabled with:
   -  CLIJ & CLIJ2 update sites
   -  [Install Cellpose](https://github.com/MouseLand/cellpose), [Link Cellpose to Fiji](https://imagej.net/plugins/trackmate/detectors/trackmate-cellpose), [Helpful video for install](https://www.youtube.com/watch?v=A_PW_N0np9A)
2. **Python 3.8+**  
   - This repository was developed and tested using Python 3.12. The provided requirements.txt includes all external packages required to run the figure generation and analysis scripts. To ensure reproducibility across systems, core scientific and visualization libraries (e.g., NumPy, SciPy, pandas, scikit-learn, matplotlib, seaborn) are explicitly version-pinned. Imaging and bioinformatics-related dependencies (e.g., tifffile, Pillow, scikit-image, microfilm) are also included. Non-essential development environment packages (e.g., Jupyter, Spyder, conda tooling) are intentionally excluded to keep the dependency list minimal and portable.
3. **ilastik** [interactive learning and segmentation toolkit](https://www.ilastik.org/)

4. **Directory structure** (customize `allData` and `datasetName` in each script):
```text
/path/to/your/data/                    # <allData>
└── 2025-01-01_DatasetName/            # <datasetName>
    ├── ND2_Split_Series/              # Step 1 Input (raw .ND2 series)
    ├── TIF_Split_Series_MaxZ/         # Step 1 Output (Max-Z projections)
=======
### Prerequisites:
1. **Python 3.8+** with the following packages:
    - `collections`
    - `copy`
    - `datetime`
    - `matplotlib`
    - `numpy`
    - `pandas`
    - `pickle`
    - `PIL`
    - `random`
    - `re`
    - `scipy`
    - `seaborn`
    - `sklearn`
    - `statistics`
    - `tiffile`
    - `time`
    - `tkinter`
    
    See [requirements_steps9-12.txt](https://github.com/WunderlichLab/Relish-RhoBAST-Analysis/blob/SVM/requirements_steps9-12.txt) for specific package requirements.
1. **Directory structure** (customize `allData` and `datasetName` in each script):
```
/path/to/your/data/                           # <allData>
└── 2025-01-01_datasetName/                   # <datasetName>
    ├── ND2_Split_Series/                     # Step 1 Input (raw .ND2 series)
    ├── TIF_Split_Series_MaxZ/                # Step 1 Output (Max-Z projections)
>>>>>>> SVM
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
```

## [Support Vector Machine Analysis](https://github.com/WunderlichLab/Relish-RhoBAST-Analysis/tree/SVM)
### Installation:
1. Clone the repo.
```
git clone https://github.com/WunderlichLab/Relish-RhoBAST-Analysis.git
```
2. Install Python 3.8+ with all necessary packages (see [requirements_steps9-12.txt](https://github.com/WunderlichLab/Relish-RhoBAST-Analysis/blob/SVM/requirements_steps9-12.txt)).
3. Change the git remote url to avoid accidental pushes to the base project.
```
git remote set-url origin https://github.com/github_username/Relish-RhoBAST-Analysis.git
git remote -v`
```
4. Run Steps 1-8 of the Relish-RhoBAST image analysis.

### Roadmap
- [ ] Step 9 (`step9_Relish_TracePreProcessing_python.py`): Python script for preprocessing Relish nuclear timecourse data and extracting trace descriptor.
- [ ] Step 10 (`step10_Relish_Trace_ClassifierGUI_python.py`): Python script creating GUI for manual trace classification of single-cell traces for SVM training set.
- [ ] Step 11 (`step11_Relish_Trace_ClassifierSVM_python.py`): Python script for SVM trace behavior classification based on long-timecourse post-stimulus single-cell Relish traces.
- [ ] Step 12 (`step12_Relish_Trace_PredictorSVM_python.py`): Python script for SVM trace behavior prediction based on short-timecourse pre-stimulus single-cell Relish traces.

Each step builds on the last, producing data and visualizations suitable for downstream analysis and figure plotting.

Step 12 is designed to be run on Relish-only data (i.e. no RhoBAST transcriptional data), and Steps 9-11 are tailored accordingly.  Modified code for running Steps 9-11 on Relish-RhoBAST data is available upon request.

  
## [Figure Generation](https://github.com/WunderlichLab/Relish-RhoBAST-Analysis/tree/Figure-Generation)
- Scripts to plot figures from Python dictionaries produced in Image Analysis Pipeline

# Contributions
Contributions and suggestions are greatly appreciated.  If you have a suggestion to make this project better, please fork the repo and create a pull request.

1. Fork the project.
2. Create your feature branch.
```
git checkout -b feature/NewFeature
```
3. Commit your changes.
```
git commit -m 'Add some NewFeature'
```
4. Push to the branch.
```
git push origin feature/NewFeature
```
5. Open a pull request.

# Contact

# Acknowledgements
- `README` template created by [othneildrew](https://github.com/othneildrew/Best-README-Template/blob/main/BLANK_README.md).