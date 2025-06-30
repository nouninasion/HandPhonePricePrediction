# Handphone Price Prediction

This repository contains a Jupyter Notebook (`HandPhonePrediction.ipynb`) that demonstrates a machine learning model to predict the prices of mobile phones. The project encompasses data loading, comprehensive preprocessing, feature engineering, model training, and evaluation.

## Table of Contents

  - [Project Overview](https://github.com/nouninasion/HandPhonePricePrediction/blob/main/README.md#project-overview)
  - [Dataset](https://github.com/nouninasion/HandPhonePricePrediction/blob/main/README.md#project-overview)
  - [Features](https://github.com/nouninasion/HandPhonePricePrediction/blob/main/README.md#project-overview)
  - [Preprocessing and Feature Engineering](https://github.com/nouninasion/HandPhonePricePrediction/blob/main/README.md#project-overview)
  - [Exploratory Data Analysis (EDA)](https://github.com/nouninasion/HandPhonePricePrediction/blob/main/README.md#project-overview)
  - [Model Training](https://github.com/nouninasion/HandPhonePricePrediction/blob/main/README.md#project-overview)
  - [Model Performance](https://github.com/nouninasion/HandPhonePricePrediction/blob/main/README.md#project-overview)
  - [Prediction Example](https://github.com/nouninasion/HandPhonePricePrediction/blob/main/README.md#project-overview)
  - [Getting Started](https://github.com/nouninasion/HandPhonePricePrediction/blob/main/README.md#project-overview)
  - [Libraries Used](https://github.com/nouninasion/HandPhonePricePrediction/blob/main/README.md#project-overview)

## Project Overview

The main objective of this project is to build a predictive model for handphone prices. The notebook explores various phone specifications as features and applies data cleaning, transformation, and machine learning algorithms (Linear Regression and RandomForestRegressor) to estimate prices.

## Dataset

The dataset used for this project is `https://www.kaggle.com/datasets/rkiattisak/mobile-phone-price`. It includes various specifications of mobile phones and their corresponding prices.

## Features

The dataset initially contains the following columns:

  - `Brand`: The brand of the mobile phone.
  - `Model`: The specific model name of the phone.
  - `Storage`: Internal storage capacity (e.g., '128 GB', '256 GB').
  - `RAM`: Random Access Memory (e.g., '6 GB', '8 GB').
  - `Screen Size (inches)`: Display size in inches.
  - `Camera (MP)`: Camera megapixel information (can be multiple cameras, e.g., '108 + 12 + 12 + 12').
  - `Battery Capacity (mAh)`: Battery capacity in milliampere-hours.
  - `Price ($)`: The price of the handphone in US dollars (target variable).

## Preprocessing and Feature Engineering

Several steps were taken to prepare the data for modeling:

  - The `Model` column was dropped as it was deemed to have too many unique values and little predictive power directly.
  - The `Brand` column, which is categorical, was encoded into numerical format using `OrdinalEncoder` with a predefined order.
  - `Storage` and `RAM` columns were cleaned by removing "GB" suffixes and spaces, then converted to integer type.
  - `Screen Size (inches)` was converted to a float type, handling potential string-based expressions using a custom `eval_screen_size` function.
  - The `Camera (MP)` column, often containing multiple camera values (e.g., "108 + 12 + 12"), was parsed to sum up the megapixel values into a single float, removing "MP" and spaces.
  - The `Price ($)` column was cleaned by removing '$' and commas, then converted to an integer.
  - A new feature, `Build_Quality`, was engineered by assigning a score based on `Storage`, `RAM`, `Screen Size (inches)`, and `Camera (MP)`. This aims to capture a combined quality metric.
  - Missing values in `Screen Size (inches)` and `Camera (MP)` (which might result from parsing errors) were imputed using the mean strategy.
  - All numerical features were scaled using `StandardScaler` to ensure they contribute equally to the model.

## Exploratory Data Analysis (EDA)

The notebook includes initial data inspection using `data.info()` to understand data types and non-null counts. Histograms were generated for all numerical features to visualize their distributions. A heatmap of the correlation matrix was also plotted to understand the relationships between the features.

## Model Training

The data was split into training (70%) and testing (30%) sets. Two regression models were trained:

1.  **Linear Regression**: A baseline model.
2.  **RandomForestRegressor**: An ensemble method known for its robustness and accuracy in regression tasks.

## Model Performance

The performance of the models was evaluated using the R-squared score on the test set:

  - **Linear Regression**: R-squared score: `0.671`.
  - **RandomForestRegressor**: R-squared score: `0.846`.

The RandomForestRegressor showed significantly better performance, indicating its suitability for this prediction task. The notebook also visualizes some of the individual decision trees from the Random Forest model.

## Prediction Example

The notebook includes an example of how to make a prediction for a new, hypothetical phone specification, demonstrating the model's practical application.

## Getting Started

To run this notebook, you will need Google Colab or a local Jupyter environment with the following libraries installed:

  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`

You can install them using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Once the dependencies are installed, you can open and run the `HandPhonePrediction.ipynb` file in your Google Colab or Jupyter environment.

## Libraries Used

  - `pandas`
  - `numpy`
  - `matplotlib.pyplot`
  - `seaborn`
  - `sklearn.model_selection`
  - `sklearn.preprocessing`
  - `sklearn.impute`
  - `sklearn.linear_model`
  - `sklearn.ensemble`
  - `sklearn.tree`
