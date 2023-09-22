# Hyperspectral and Thermal Imaging for the Detection of <i>Xylella fastidiosa</i>

This repository contains the codes and extracted data needed to reproduce the project.

## Instructions

The repository includes the following files with the extension `.m`. To replicate the results, follow these steps:

1. Clone the repository.
2. Open MATLAB with the working directory set to the project's root directory.
3. Run the [**main.m**](https://github.com/lopsandrea/xylella-ip/blob/main/main.m) and [**classification.m**](https://github.com/lopsandrea/xylella-ip/blob/main/classification.m) files.

## *.m Files

There are 3 MATLAB files in the project:

   - [**main.m**](https://github.com/lopsandrea/xylella-ip/blob/main/main.m): This code section performs various operations on georeferenced images and KML data to extract meaningful information about trees. It includes data loading, image processing, and feature extraction.

   - [**classification.m**](https://github.com/lopsandrea/xylella-ip/blob/main/classification.m): This code serves as a comprehensive framework for evaluating different machine learning models and feature reduction techniques on the extracted features. It includes data preprocessing, feature reduction, neural network training, and metrics evaluation.

   - [**testsimpleSVM.m**](https://github.com/lopsandrea/xylella-ip/blob/main/testsimpleSVM.m): This is a test code that was used to test the over-sampling technique on a simple SVM with and without PCA.

## *.csv Files

The outputs from the [**main.m**](https://github.com/lopsandrea/xylella-ip/blob/main/main.m) file have been saved in two CSV files:

- [**originalFeatures.csv**](https://github.com/lopsandrea/xylella-ip/blob/main/originalFeatures.csv): Contains the feature matrix where rows represent trees. The first 47 columns represent the average values of the bands obtained from the hyperspectral file for each tree. Columns 48, 49, and 50 represent the average values of NDVI, NPQI, and thermal values, respectively.

- [**originalTargets.csv**](https://github.com/lopsandrea/xylella-ip/blob/main/originalTargets.csv): Contains the target vector, where each element represents a tree. It takes on a value of 0 when negative and 1 when positive.

## *.kml Files

- [**doc.kml**](https://github.com/Quantalab/Xf-NPlants-2018/blob/master/codes/Analysis1.R): This file contains data extracted from the raw KMZ file, providing information about trees, including their position, name, and GT, in XML format.

## Raw Data

This repository follows the principles of reproducible research [(Peng, 2011)](http://science.sciencemag.org/content/334/6060/1226). For access to raw data, please contact [Prof. Andrea Guerriero](mailto:andrea.guerriero@poliba.it).

## Acknowledgments

I would like to express my sincere gratitude to Prof. Andrea Guerriero and Dr. Raffaella Matarrese for their guidance and support throughout this project.

