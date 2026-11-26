# 📊 Telco Customer Churn Prediction

## 📌 Project Overview

Customer churn is one of the major challenges faced by telecommunication companies. Identifying customers who are likely to leave helps businesses take proactive steps to improve customer retention.

This project uses Machine Learning algorithms and ensemble learning techniques to predict whether a telecom customer is likely to churn. It includes data preprocessing, Exploratory Data Analysis (EDA), hyperparameter tuning, model comparison, and an interactive Streamlit application for customer churn prediction.

## 🎯 Objectives

- Analyze customer data to identify patterns associated with churn.
- Develop machine learning models to predict customer churn.
- Compare individual and ensemble learning algorithms.
- Improve model performance through hyperparameter tuning.
- Identify important features influencing churn predictions.
- Build an interactive web application for real-time predictions.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Boosting Algorithms:** AdaBoost, XGBoost
- **Ensemble Learning:** Soft Voting, Stacking
- **Hyperparameter Tuning:** RandomizedSearchCV
- **Model Deployment:** Streamlit
- **Model Serialization:** Joblib

## 📂 Project Structure

```text
Telco-Customer-Churn-Prediction/
│
├── data/
│   └── dataset.csv
│
├── models/
│   └── best_churn_model.pkl
│
├── eda.py
├── train.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 📊 Dataset

The project uses the Telco Customer Churn dataset.

- Total Records: 7,043
- Total Features: 20 input features
- Target Variable: Churn
- Problem Type: Binary Classification

The target variable indicates whether a customer has left the telecom company.

- **0 – No Churn:** Customer stayed.
- **1 – Churn:** Customer left.

## 🔄 Project Workflow

1. **Data Cleaning and Preprocessing**
   - Handled missing values.
   - Converted TotalCharges into numerical format.
   - Removed unnecessary customer ID column.
   - Encoded categorical variables.
   - Scaled numerical features.

2. **Exploratory Data Analysis**
   - Analyzed customer churn distribution.
   - Examined contract types and churn patterns.
   - Studied customer tenure and charges.
   - Visualized relationships between features.

3. **Model Development**

   Implemented the following machine learning models:

   - Logistic Regression
   - AdaBoost Classifier
   - XGBoost Classifier
   - Soft Voting Classifier
   - Stacking Classifier

4. **Hyperparameter Tuning**
   - Applied RandomizedSearchCV.
   - Optimized AdaBoost and XGBoost hyperparameters.
   - Used 3-fold cross-validation for parameter search.

5. **Model Evaluation**
   - Accuracy
   - Precision
   - Recall
   - F1 Score
   - ROC-AUC
   - Confusion Matrix

6. **Feature Importance Analysis**
   - Identified important features influencing customer churn predictions using XGBoost.

7. **Model Deployment**
   - Saved the best-performing model using Joblib.
   - Developed an interactive Streamlit application for customer churn prediction.

## 📈 Model Performance

The following table shows the evaluation results on the test dataset.

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 73.81% | 50.43% | 78.34% | 61.36% | 0.8416 |
| AdaBoost (Original) | 69.34% | 45.82% | 85.03% | 59.55% | 0.8208 |
| XGBoost (Original) | 74.73% | 51.58% | 78.61% | 62.29% | 0.8424 |
| Soft Voting (Original) | 74.02% | 50.67% | 81.02% | 62.35% | 0.8436 |
| Stacking (Original) | 74.38% | 51.12% | 79.14% | 62.12% | 0.8455 |
| AdaBoost (Tuned) | 74.10% | 50.76% | 79.95% | 62.10% | 0.8389 |
| XGBoost (Tuned) | 75.44% | 52.51% | 78.34% | 62.88% | 0.8448 |
| Soft Voting (Tuned) | 74.88% | 51.73% | 79.95% | 62.82% | 0.8459 |
| **Stacking (Tuned)** | **75.23%** | **52.18%** | **79.95%** | **63.15%** | **0.8464** |

### 🏆 Best Performing Model

**Tuned Stacking Classifier**

- Accuracy: 75.23%
- Precision: 52.18%
- Recall: 79.95%
- F1 Score: 63.15%
- ROC-AUC: 0.8464

The Tuned Stacking Classifier achieved the highest ROC-AUC score among the evaluated models.

## 🔍 Important Features

XGBoost feature importance analysis identified the following influential features:

1. Month-to-month Contract
2. Fiber Optic Internet Service
3. No Online Security
4. Two-year Contract
5. One-year Contract
6. DSL Internet Service
7. Streaming Movies
8. No Tech Support
9. Customer Tenure
10. Electronic Check Payment Method

These features help identify customer characteristics associated with churn.

## 🚀 Installation and Execution

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/Telco-Customer-Churn-Prediction.git
```

### 2. Navigate to the Project Folder

```bash
cd Telco-Customer-Churn-Prediction
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Exploratory Data Analysis

```bash
python eda.py
```

### 5. Train the Models

```bash
python train.py
```

The training process compares all models and saves the best-performing model inside the `models` directory.

### 6. Launch the Streamlit Application

```bash
python -m streamlit run app.py
```

The application allows users to enter customer details and receive churn predictions with probability scores and suggested retention actions.

## 💡 Future Improvements

- Improve churn prediction performance through additional feature engineering.
- Explore advanced ensemble and optimization techniques.
- Enhance the Streamlit dashboard with interactive analytics.
- Incorporate customer segmentation for targeted retention strategies.

## 👩‍💻 Author

**Devika Dileep**

AI/ML Intern | Machine Learning Enthusiast

---

⭐ If you find this project interesting, consider giving the repository a star!
