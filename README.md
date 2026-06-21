# Hospital Readmission Prediction

## Project Overview
This project predicts the likelihood of a patient being readmitted to the hospital within 30 days. The goal is to build a machine learning model that identifies high-risk patients, enabling healthcare providers to intervene early and improve patient outcomes.

The workflow covers the entire data science lifecycle, from data preprocessing and feature engineering to building, evaluating, and deploying a predictive model.

## Key Features & Insights
*   **Data Preprocessing**: Cleaned and prepared the `diabetic_data.csv` dataset with over 100,000 patient records, handling missing values and removing irrelevant columns.
*   **Feature Engineering**: Created 9 new features, including `total_visits`, `medication_complexity`, and `chronic_condition`, to capture more complex clinical patterns.
*   **Model Building**: Developed and compared two machine learning models:
    *   **Random Forest**: A strong baseline model.
    *   **XGBoost with SMOTE**: An advanced ensemble model that uses SMOTE to handle class imbalance.
*   **Model Performance**: The XGBoost model achieved an AUC of 0.62 and is able to identify **85% of actual readmissions**, making it a valuable tool for clinical intervention.
*   **Interactive Dashboard**: Built a dynamic Power BI dashboard to visualize risk scores, model performance, and a list of high-risk patients.

## Tools & Technologies
*   **Python**: Primary language for data processing and machine learning.
*   **Pandas**: For data manipulation and cleaning.
*   **Scikit-learn**: Used for data splitting, model evaluation, and implementing the Random Forest classifier.
*   **XGBoost**: The main machine learning library for the final predictive model.
*   **Imbalanced-learn (SMOTE)**: Used to handle the imbalanced nature of the readmission target variable.
*   **Matplotlib**: Created static visualizations for feature importance and model performance.
*   **Power BI**: Designed and published the final interactive dashboard.

## Repository Contents
*   `readmission_analysis.py`: The initial data analysis and modeling script.
*   `readmission_improved.py`: The final, improved script with feature engineering and the XGBoost model.
*   `diabetic_data.csv`: The raw, unprocessed dataset.
*   `cleaned_readmission_data.csv`: The data after preprocessing.
*   `improved_readmission_predictions.csv`: The final predictions and risk scores for all patients.
*   `power Bi dashboard.pbix`: The Power BI dashboard file.
*   `*.png`: Image exports of key visualizations and model performance charts.

## How to Use This Project
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/JAYBITE98/Hospital-Readmission-Prediction.git
