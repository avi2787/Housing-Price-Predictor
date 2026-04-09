# Housing Price Predictor

This repository contains a supervised machine learning project for predicting housing prices using the California housing dataset.

## Project Overview

The project trains regression models to predict `median_house_value` from housing features such as:
- longitude, latitude
- housing age
- total rooms, total bedrooms
- population, households
- median income
- ocean proximity

The models include:
- `LinearRegression`
- `DecisionTreeRegressor`
- `RandomForestRegressor`
- `XGBRegressor`
- `LGBMRegressor`
- stacked ensembles with spatial cross-validation

## Key Files

- `housingmarket1.py` - end-to-end housing price regression pipeline with preprocessing, model training, cross-validation, and evaluation.
- `housingmarketnew.py` - advanced housing price modeling with feature engineering, spatial validation, quantile prediction, conformal calibration, and SHAP explainability.
- `first.py` - a small regression example predicting life satisfaction from GDP per capita.
- `datasets/housing/housing.csv` - the dataset used by the housing price models.

## Dataset

The housing data is stored in:

- `datasets/housing/housing.csv`

This file should be included in the repository so the models can run without downloading.

## Requirements

Install required Python packages before running the scripts:

```bash
pip install numpy pandas matplotlib scikit-learn xgboost lightgbm shap scipy
```

## Running the project

To run the main housing price model scripts, use:

```bash
python housingmarket1.py
python housingmarketnew.py
```

## Notes

- This project uses **supervised learning** for regression.
- The current Git remote is configured for Bitbucket. To publish this repository to a GitHub repository named `Housing Price Predictor`, update the remote origin to your GitHub repository URL and push the branch.
- 
<img width="1013" height="833" alt="image" src="https://github.com/user-attachments/assets/cbd6c1ab-c1d2-472d-a5f0-08cac9a34007" />
<img width="1248" height="952" alt="image" src="https://github.com/user-attachments/assets/2a3fefae-64f7-4db1-830e-18bed35b7ea5" />


