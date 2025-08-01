# Health Data Analysis and Classification

This repository contains a Jupyter Notebook (`task.ipynb`) that performs comprehensive data analysis and classification on a health dataset (`health_data_10000_chunk.csv`). The goal is to predict the "Current status of microbiota" (Optimal, Suboptimal, At Risk) using various machine learning and deep learning models.

## Project Overview

The notebook processes a health dataset, performs feature engineering, visualizes data distributions, handles imbalanced classes, and evaluates multiple classification models. It also implements a hierarchical TabTransformer model using PyTorch for advanced classification. The code includes hyperparameter tuning via GridSearchCV and SHAP analysis for feature importance.

## Requirements

To run the notebook, install the required Python packages. The key dependencies are listed below:

```bash
pip install gspread google-colab oauth2client
pip install optuna
pip install scikeras
pip install tab-transformer-pytorch
pip install pytorch-tabular[all]
pip install numpy pandas seaborn matplotlib scikit-learn imblearn xgboost lightgbm tensorflow shap torch
```

Ensure you have Python 3.8+ and Jupyter Notebook installed. The dataset (`health_data_10000_chunk.csv`) must be available in the working directory or Google Colab environment.

## Dataset

The dataset (`health_data_10000_chunk.csv`) contains health-related features such as:

- **Demographic data**: Age, Gender, BMI, etc.
- **Lifestyle factors**: Smoking status, physical activity, sleep hours, stress level.
- **Dietary habits**: Weekly consumption of vegetables, fruits, fermented foods, etc.
- **Medical history**: Medical conditions, family history, diagnosed conditions.
- **Microbiota-related features**: Current status of microbiota (target variable), intestinal health indicators.

The target variable, "Current status of microbiota," has three classes: Optimal, Suboptimal, and At Risk.

## Notebook Structure

The notebook is organized into the following sections:

1. **Setup and Imports**:
   - Installs necessary libraries.
   - Imports Python packages for data processing, visualization, and modeling.

2. **Data Loading and Preprocessing**:
   - Loads the dataset using `pandas`.
   - Drops irrelevant columns (e.g., Residential Address).
   - Encodes multi-label categorical columns using `MultiLabelBinarizer`.
   - Processes smoking status and boolean columns.
   - Applies `LabelEncoder` to remaining categorical features.
   - Removes outliers using the Interquartile Range (IQR) method for BMI and intestinal health indicators.

3. **Exploratory Data Analysis (EDA)**:
   - Visualizes distributions of categorical and numerical features using `seaborn` and `matplotlib`.
   - Displays value counts for categorical columns.
   - Generates a correlation matrix heatmap for selected features (BMI, Current diet, Current status of microbiota).

4. **Feature Engineering**:
   - Creates composite features:
     - Gut Health Score
     - Diet Quality Score
     - Metabolic Risk Score
     - Lifestyle Balance Index
     - Supplement Compliance Score
     - Genetic Risk Score
   - Uses SHAP values to aggregate multi-label feature importance for medical conditions, family history, etc.

5. **Data Preparation for Modeling**:
   - Splits features (`X`) and target (`y`).
   - Applies `StandardScaler` for feature scaling.
   - Balances classes using SMOTE (`imblearn.over_sampling.SMOTE`).
   - Splits data into training and test sets (80/20 split).

6. **Model Training and Evaluation**:
   - Evaluates multiple models using a `Pipeline`:
     - RandomForestClassifier
     - KNeighborsClassifier
     - SVM (SVC)
     - LogisticRegression
     - AdaBoostClassifier
     - GradientBoostingClassifier
     - GaussianNB
     - XGBoostClassifier
     - Multi-Layer Perceptron (MLP) using Keras
   - Plots confusion matrices and ROC curves for each model.
   - Reports accuracy, precision, recall, F1-score, and ROC-AUC.

7. **Hyperparameter Tuning**:
   - Uses `GridSearchCV` to tune hyperparameters for all models, including the MLP.
   - Defines parameter grids for each model (e.g., `n_estimators`, `max_depth` for RandomForest).
   - Reports best parameters and cross-validation scores.

8. **TabTransformer Implementation**:
   - Implements a hierarchical TabTransformer model using `tab_transformer_pytorch`.
   - Trains two models:
     - Top model: Binary classification (At Risk vs. Not At Risk).
     - Bottom model: Binary classification (Optimal vs. Suboptimal for non-At Risk samples).
   - Uses early stopping to prevent overfitting.
   - Combines predictions hierarchically and evaluates using classification metrics.

## How to Run

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**:
   Run the pip commands listed in the Requirements section.

3. **Prepare the Dataset**:
   Place `health_data_10000_chunk.csv` in the working directory or upload it to Google Colab.

4. **Open the Notebook**:
   Launch Jupyter Notebook or open the `.ipynb` file in Google Colab:
   ```bash
   jupyter notebook task.ipynb
   ```

5. **Run the Notebook**:
   Execute cells sequentially. Note that some sections (e.g., model training, GridSearchCV) may take significant time depending on your hardware.

## Outputs

- **Visualizations**:
  - Distribution plots for categorical and numerical features.
  - Correlation matrix heatmap.
  - Confusion matrices and ROC curves for each model.

- **Metrics**:
  - Accuracy, precision, recall, F1-score, and ROC-AUC for all models.
  - Best hyperparameters and cross-validation scores from GridSearchCV.
  - Classification report for the hierarchical TabTransformer model.

- **Feature Importance**:
  - SHAP summary for XGBoost model, highlighting top features influencing microbiota status.
  - Aggregated SHAP values for multi-label feature groups.

## Notes

- The notebook assumes access to a GPU for faster training of the TabTransformer model. If no GPU is available, the code defaults to CPU.
- The SVM model is commented out in the GridSearchCV section due to computational cost. Uncomment and adjust parameters if needed.
- The dataset is not included in the repository due to potential size or privacy constraints. Ensure you have it before running the code.
- The TabTransformer model requires PyTorch and `tab_transformer_pytorch`. Ensure proper installation to avoid import errors.

## Future Improvements

- Add cross-validation for the TabTransformer model.
- Explore additional feature engineering techniques (e.g., interaction terms).
- Optimize computational efficiency for GridSearchCV and deep learning models.
- Incorporate additional advanced models (e.g., LightGBM, CatBoost).
