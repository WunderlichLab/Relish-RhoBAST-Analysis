# Support Vector Machine Analysis

## About the Project 
Scripts to run the Relish nuclear localization Classifier and Predictor SVMs in "[Heterogeneous NF-κB activation and enhancer features shape transcription in Drosophila immunity]"(https://www.biorxiv.org/content/10.1101/2025.05.19.654881v1).

### Built with:
- [Anaconda](https://www.anaconda.com/)
- [Python](https://www.python.org/)

## Getting Started

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
1. **Directory structure** (customize `allData` and `datasetName` in each script):
```
/path/to/your/data/                           # <allData>
└── 2025-01-01_datasetName/                   # <datasetName>
    ├── ND2_Split_Series/                     # Step 1 Input (raw .ND2 series)
    ├── TIF_Split_Series_MaxZ/                # Step 1 Output (Max-Z projections)
    ├── Trackmate Files/
    │   └── <maskSettings>/
    │       ├── Cyto Fiji File/               # Step 2 Output (Cellpose+TrackMate overlays)
    │       ├── Cyto Masks/                   # Step 2 Output (binary masks)
    │       ├── Cyto Matched Masks/           # Step 4 Output (nuclei reassigned → cyto IDs)
    │       ├── Nuclei Fiji File/             # Step 2 Output (Cellpose+TrackMate overlays)
    │       ├── Nuclei Masks/                 # Step 2 Output (binary masks)
    │       └── Nuclei Matched Masks/         # Step 4 Output (nuclei reassigned → cyto IDs)
    ├── Python/
    │   └── <maskSettings>/
    │       ├── Interpolated Masks_fullinterp/
    │       │   ├── Cyto/                      # Step 3 Output (interpolated cell masks)
    │       │   └── Nuclei/                    # Step 3 Output (interpolated cell masks)
    │       └── IntensitiesDF/
    │           ├── dictIntensities_{datetimeStr}.pkl
    │           └── dictIntensitiesNomask_{datetimeStr}.pkl
    │                                          # Step 7 Output (pickle intensity dicts)
    ├── ilastik Outputs/
    │   ├── Probabilities/                     # Step 5 Output (ilastik Pixel Classification)
    │   └── Aptamer Masks/                     # Step 6 Output (ilastik Object Classification)
    └── Sorted Cells/                          # Step 8 Output (manual QC)
    │   └── <maskSettings>/
    │       ├── Good Cells/
    │       └── Bad Cells/
    └── Processed Traces/
    │   ├── subcluster_traces_smooth.pkl       # Step 9 Output (smoothed traces by subcluster)
    │   ├── dict_trace_descriptors.pkl         # Step 9 Output (extraced trace descriptors)
    │   ├── all_traces_df.pkl                  # Step 9 Output (all smoothed traces)
    │   └── times_smooth.pkl                   # Step 9 Output (smoothed timepoints)
    └── SVM Classifier/
    │   ├── cell_categories.csv                # Step 10 Output (manually sorted training set - .csv)
    │   ├── behavior_categories_df.pkl         # Step 10 Output (manually sorted training set - .pkl)
    │   ├── dict_trace_descriptors_SVM.pkl     # Step 11 Output (dict_trace_descriptors with "Behavior" column populated)
    │   ├── df_descriptor_vals_all.pkl         # Step 11 Output (flattened dict_trace_descriptors containing only key features)
    │   ├── SVM_model.pkl                      # Step 11 Output (classifier SVM model)
    │   ├── SVM_results_dict.pkl               # Step 11 Output (SVM results)
    │   └── percent_behaviors_df.pkl           # Step 11 Output (percent cells per behavior category)
    └── SVM Predictor/
        ├── predictor_SVM_model.pkl            # Step 12 Output (predictor SVM model)
        ├── predictor_SVM_results_df.pkl       # Step 12 Output (predictor SVM results)
        ├── threshold_features_RFE.pkl         # Step 12 Output (RFE-optimized predictor SVM features)
        ├── predictor_SVM_model_RFE.pkl        # Step 12 Output (post-RFE predictor SVM model)
        └── predictor_SVM_results_df_RFE.pkl   # Step 12 Output (post-RFE predictor SVM results)

```

### Installation:
1. Clone the repo.
```
git clone https://github.com/WunderlichLab/Relish-RhoBAST-Analysis.git
```
2. Install Python 3.8+ with all necessary packages.
3. Change the git remote url to avoid accidental pushes to the base project.
```
git remote set-url origin https://github.com/github_username/Relish-RhoBAST-Analysis.git
git remote -v`
```
4. Run Steps 1-8 of the Relish-RhoBAST image analysis.

## Usage

### Roadmap
- [ ] Step 9 (`step9_Relish_TracePreProcessing_python.py`): Python script for preprocessing Relish nuclear timecourse data and extracting trace descriptor.
- [ ] Step 10 (`step10_Relish_Trace_ClassifierGUI_python.py`): Python script creating GUI for manual trace classification of single-cell traces for SVM training set.
- [ ] Step 11 (`step11_Relish_Trace_ClassifierSVM_python.py`): Python script for SVM trace behavior classification based on long-timecourse post-stimulus single-cell Relish traces.
- [ ] Step 12 (`step12_Relish_Trace_PredictorSVM_python.py`): Python script for SVM trace behavior prediction based on short-timecourse pre-stimulus single-cell Relish traces.

Each step builds on the last, producing data and visualizations suitable for downstream analysis and figure plotting.

Step 12 is designed to be run on Relish-only data (i.e. no RhoBAST transcriptional data), and Steps 9-11 are tailored accordingly.  Modified code for running Steps 9-11 on Relish-RhoBAST data is available upon request.

### Contributions
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

## Contact

## Acknowledgements
- `README` template created by [othneildrew](https://github.com/othneildrew/Best-README-Template/blob/main/BLANK_README.md).
=======
# Relish-RhoBAST Analysis
 This repository contains all analysis scripts for data presented in ["Heterogeneous NF-κB activation and enhancer features shape transcription in Drosophila immunity"](https://doi.org/10.1101/2025.05.19.654881).

## [Confocal Image Analysis Pipeline](https://github.com/WunderlichLab/Relish-RhoBAST-Analysis/tree/Relish-RhoBAST-Image-Analysis)
- Scripts to analyze the single-cell fluorescence microscopy time-lapse signals of labeled nuclei, Relish, and RhoBAST

## [Support Vector Machine Analysis](https://github.com/WunderlichLab/Relish-RhoBAST-Analysis/tree/SVM)
- Scripts to run Support Vector Machine analysis of:
  - Relish spatiotemporal classification
  - Pre-stimulus predictive classification
## [Figure Generation](https://github.com/WunderlichLab/Relish-RhoBAST-Analysis/tree/Figure-Generation)
- Scripts to plot figures from Python dictionaries produced in Image Analysis Pipeline

