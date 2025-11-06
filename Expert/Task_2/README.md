# Task 2: Implement Hyperparameter Tuning

## Overview
This task demonstrates how to optimize model performance through hyperparameter tuning using `RandomizedSearchCV`. The RandomForest model is tuned across multiple parameter combinations and evaluated for accuracy and F1-score.

## Steps
1. Load and preprocess the Wine dataset.
2. Define a range of hyperparameters for RandomForestClassifier.
3. Use RandomizedSearchCV to find the best parameters.
4. Train the best model and evaluate it on the test set.
5. Display the best parameter combination and classification report.

## Requirements
- Python 3.x
- scikit-learn
- numpy
