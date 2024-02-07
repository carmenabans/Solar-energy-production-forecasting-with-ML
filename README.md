# Solar Energy Production Forecasting using Machine Learning

## Introduction:

The goal of this project is to practice different machine learning methods and hyperparameter tuning/optimization (HPO) for time series forecasting of solar power generation. The project involves:
* Selecting the best model for a given dataset (including hyperparameter tuning)
* Estimating the future performance of the best model (model evaluation)
* Building the final model and using it to make new predictions on unseen data (model usage)

## Context
Modern power grids rely on renewable energy sources, such as solar and wind power. However, these sources are intermittent and require accurate forecasting to ensure grid stability. Numerical weather prediction (NWP) models can be used to predict weather variables, which can then be used as input to machine learning models to predict solar power generation.

## Dataset
The dataset for this project consists of 12 years of historical data for a solar plant in Oklahoma. The data includes 75 weather variables predicted by the GFS NWP model, as well as the actual solar power generation. 

The data is split into two sets:
* Training set: The first 10 years of data
* Test set: The last 2 years of data

Data from Kaggle:  AMS 2013-2014 Solar Energy Prediction Contest: https://www.kaggle.com/c/ams-2014-solar-energy-prediction-contest

## Includes
1. **Exploratory Data Analysis (EDA)**: Analyzed the dataset to understand the distribution of the variables and identify any potential outliers.
2. **Data Preprocessing**: Cleaned and preprocess the data for machine learning. This may involve scaling the features, handling missing values, and dealing with categorical features.
3. **Baseline Models**: Implemented and evaluated basic machine learning models, such as linear regression, K-nearest neighbors, and decision trees.
4. **Dimensionality Reduction**: Investigated techniques for reducing the dimensionality of the dataset, such as principal component analysis (PCA) or feature selection.
5. **Advanced Models**: Implemented and evaluated more advanced machine learning models, such as support vector machines (SVMs) and random forests.
6. **Model Selection**: Selected the best model based on the results of the previous steps.
7. **Final Model**: Trained the final model on the training set and use it to make predictions on the test set.

## Team
* Carmen Abans Maciel: https://github.com/carmenabans
* Noelia Hernández Rodríguez: https://github.com/Noeliahr10

## References:
* Pandas: https://pandas.pydata.org
* NumPy: https://numpy.org
* Scikit-learn: https://scikit-learn.org/stable/
* Skopt: https://scikit-optimize.github.io/stable/
* XGBoost: https://xgboost.readthedocs.io/en/stable/
* LightGBM: https://lightgbm.readthedocs.io/en/stable/
