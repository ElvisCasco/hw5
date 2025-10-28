# Group composition:
- Elvis Casco
- Maríajosé Argote
- María Victoria Suriel 

## Library Description
The project consists of a Python library called process_data, developed to apply object-oriented programming principles to the processing and modeling of the sample_diabetes_mellitus_data.csv dataset.

The library is organized into four main folders, each responsible for a specific part of the workflow:

- Data : Contains the classes responsible for importing the dataset, splitting it into training and test sets, and cleaning the data (removing missing values and filling them with the mean).
- Features: Includes classes that transform or modify variables, such as those performing encoding (e.g., gender or ethnicity).
- Interfaces:  Defines interfaces that structure how the various functions and classes interact
- Model:  Contains the predictive model class, implemented using Logistic Regression from scikit-learn.

## Test Package
The test_package file installs the library directly from GitHub and then uses its functions to execute the full data processing workflow.This script validates that all classes work correctly, from data loading to final predictions on the sample_diabetes_mellitus_data.csv dataset.

## Pipeline Example
The pipeline_example file demonstrates the complete processing workflow. Once the repository is downloaded, this script imports the necessary classes from the process_data library and executes the following steps:
- Load and clean the dataset.
- Transform the features.
- Train the predictive model.
- Generate predictions and prediction probabilities.
