# Customer Churn Prediction

This project is an end-to-end machine learning classification task aimed at predicting customer churn for a telecommunications company. By analyzing customer data, the model identifies individuals likely to cancel their subscriptions, allowing the business to take proactive steps for customer retention.

![Correlation Heatmap](https://i.imgur.com/T0tB44U.png)

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Key Findings from EDA](#key-findings-from-eda)
- [Model Performance](#model-performance)
- [Technologies Used](#technologies-used)
- [How to Run This Project](#how-to-run-this-project)
- [Predictive System](#predictive-system)
- [Future Work](#future-work)

---

## Project Overview

Customer churn is a critical metric for subscription-based businesses. This project implements a complete machine learning pipeline to tackle this problem:

- **Data Cleaning and Preprocessing:** Handles missing values and incorrect data types.
- **Exploratory Data Analysis (EDA):** Uncovers insights and patterns in the data through visualization.
- **Feature Engineering:** Encodes categorical variables for model compatibility.
- **Imbalance Handling:** Uses the Synthetic Minority Oversampling Technique (SMOTE) to address the imbalanced class distribution in the target variable.
- **Model Training and Selection:** Compares the performance of Decision Tree, Random Forest, and XGBoost classifiers using 5-fold cross-validation.
- **Model Evaluation:** Assesses the final model's performance on unseen test data using various metrics like Accuracy, Confusion Matrix, and a Classification Report.
- **Model Persistence:** Saves the trained model and data encoders for future use in a predictive system.

---

## Dataset

The dataset used is the **Telco Customer Churn** dataset, which contains customer-level information, including:
- **Demographics:** Gender, Senior Citizen status, Partner, Dependents.
- **Account Information:** Tenure, Contract type, Payment Method, Paperless Billing, Monthly Charges, Total Charges.
- **Services:** Phone Service, Multiple Lines, Internet Service, Online Security, Online Backup, etc.
- **Target Variable:** `Churn` (Yes/No).

---

## Project Workflow

1.  **Data Loading & Understanding:** The `data.csv` file is loaded into a Pandas DataFrame. Initial analysis (`.info()`, `.describe()`, `.shape`) is performed.
2.  **Data Cleaning:**
    - The `customerID` column, being a unique identifier, is dropped.
    - The `TotalCharges` column is cleaned by converting empty strings to '0.0' and changing its data type to float.
3.  **Exploratory Data Analysis (EDA):**
    - The distribution of numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) is visualized using histograms and box plots.
    - The count of each category in categorical features is visualized using count plots.
    - A correlation heatmap is generated for numerical columns.
    - **Crucially, a significant class imbalance is identified in the `Churn` column.**
4.  **Data Preprocessing:**
    - The target variable `Churn` is label encoded (Yes=1, No=0).
    - All other categorical features are label encoded. The encoders are saved using `pickle` for consistent transformation in the predictive system.
5.  **Train-Test Split:** The dataset is split into training (80%) and testing (20%) sets.
6.  **Handling Class Imbalance:** **SMOTE** is applied **only to the training data** to create a balanced dataset for the model to learn from, avoiding data leakage into the test set.
7.  **Model Selection:**
    - Three models are trained and evaluated using 5-fold cross-validation on the SMOTE-resampled training data:
        - Decision Tree Classifier
        - Random Forest Classifier
        - XGBoost Classifier
    - The **Random Forest Classifier** showed the best cross-validation accuracy (84%).
8.  **Model Training & Evaluation:** The final Random Forest model is trained on the full SMOTE-resampled training data and evaluated on the original, imbalanced test set.

---

## Key Findings from EDA
- There is a strong positive correlation between `tenure` and `TotalCharges`, and between `MonthlyCharges` and `TotalCharges`.
- The dataset is imbalanced, with significantly more customers not churning than churning (`No`: 5174, `Yes`: 1869). This makes accuracy a less reliable metric and highlights the need for techniques like SMOTE.

---

## Model Performance

The final **Random Forest Classifier** achieved the following results on the test data:

- **Accuracy Score:** **78.2%**
