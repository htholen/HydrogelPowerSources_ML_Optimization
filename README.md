# HydrogelPowerSources_ML_Optimization

MATLAB code for the study: "Design optimization of electric-fish-inspired power sources using statistical and machine learning approaches"

Haley M. Tholen, Ahmed S. Mohamed, Valentino I. Wilson, Derek M. Hall, and Joseph S. Najem

Journal of Power Sources, 2025

# Overview
This repository provides a demonstration of the machine learning and inverse design framework developed for "Design optimization of electric-fish-inspired power sources using statistical and machine learning approaches"
The main MATLAB script produces
- Gaussian Process Regression (GPR) surrogate modeling using ARD kernels
- Hyperparameter optimization and feature relevance analysis
- Inverse design using multi-start fmincon and simulated annealing
- User-friendly multi-target query
- Visualization of example model accuracy and results

All computations are self-contained and run on the included demo dataset in the subfolder "Data Files"

# Folder contents
"ML_with_Optimization_demo.m" | Main script that executes the full workflow

"Data Files/" | Contains sample input data

  "SampleDataset_MLOptimizationDemo.csv" | Sample dataset
    
"bestGPRSettingsARD.mat" | Stored GPR model parameters and accuracy metrics

"InverseResults.mat" | Saved inverse optimization results

"lengthScaleData.mat | Feature relevance information from ARD analysis

# Requirements
- Tested on MATLAB R2024b, compatible with R2018a or newer
- Statistics and Machine Learning Toolbox
- Optimization Toolbox

# Instructions to run
- Download or clone this repository
- Open MATLAB and set the working directory
- Run the main script section by section
