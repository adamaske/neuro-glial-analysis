# Neuro-Glia Analysis

## Overview
Neuro-Glia Analysis is a pipeline project for the preprocessing and analysis of EEG and fNIRS data. The goal is to provide a structured, automated, and reproducible framework for handling neuroimaging data efficiently. The pipeline includes preprocessing, artifact removal, and various analysis techniques tailored for EEG and fNIRS studies.

### Supported Modalities:
- **EEG (Electroencephalography)**: Signal filtering, artifact removal, event-related potential (ERP) analysis, time-frequency decomposition.
- **fNIRS (Functional Near-Infrared Spectroscopy)**: Optical density and hemoglobin conversion, high order bandpass filtering, and frequency analysis.

## Preprocessing Pipeline
This module is designed to preprocess all data from experiments in the same manner.
- **fNIRS**: Includes digital bandpass filtering designed to maximize the fitler or
The preprocessing steps ensure clean and reliable data before analysis. The main steps include:

## Analysis Pipeline

### Epochs
This segments the data around the onset of stimuli. 

### **Common Preprocessing Steps for EEG & fNIRS:**
1. **Loading Data**: Read EEG (e.g., `.edf`, `.set`) and fNIRS (e.g., `.snirf`) files.
2. **Artifact Removal**: Identify and correct motion artifacts and noise.
3. **Filtering**: Apply bandpass filters to remove unwanted frequency components.
4. **Normalization**: Standardize data across channels.
5. **Segmentation**: Extract meaningful epochs based on experimental conditions.
6. **Quality Control**: Automated and manual inspection of data integrity.

## TODO
- [x] Define the preprocessing steps.
- [ ] Implement data loading functions for EEG (`.edf`, `.set`) and fNIRS (`.snirf`).
- [ ] Implement optical density conversion for fNIRS.
- [ ] Implement bandpass filtering for both EEG and fNIRS.
- [ ] Implement artifact removal methods (e.g., ICA for EEG, wavelet filtering for fNIRS).
- [ ] Implement z-normalization for standardization.
- [ ] Implement event-related analysis for EEG.
- [ ] Implement functional connectivity analysis for fNIRS.
- [ ] Save the preprocessed data in an appropriate format.
- [ ] Test the pipeline with sample EEG and fNIRS datasets.
- [ ] Optimize for batch processing of multiple experiments.
- [ ] Add logging and error handling.
- [ ] Document each function for better maintainability.

## Future Enhancements
- Implement a **GUI** or **CLI** tool for ease of use.
- Develop an interactive visualization module for signal inspection.
- Integrate advanced artifact correction techniques (e.g., PCA, regression-based corrections).
- Enable real-time data streaming and preprocessing.
- Add support for multimodal integration (e.g., EEG-fNIRS fusion).

This document serves as a roadmap for building a robust **Neuro-Glia Analysis** pipeline. Contributions and improvements are welcome!

