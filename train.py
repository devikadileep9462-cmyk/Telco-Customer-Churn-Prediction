

# ==========================================
# TELCO CUSTOMER CHURN PREDICTION
# LOGISTIC REGRESSION, ADABOOST AND XGBOOST
# HYPERPARAMETER TUNING, VOTING AND STACKING
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    VotingClassifier,
    StackingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv(
    "WA_Fn-UseC_-Telco-Customer-Churn (1).csv"
)

print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())


# ==========================================
# 2. DATA CLEANING
# ==========================================

# Convert TotalCharges into numeric

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Fill missing values

df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Remove customerID because it is not useful for prediction

df.drop("customerID", axis=1, inplace=True)

print("\nMissing Values:")
print(df.isnull().sum())


# ==========================================
# 3. SEPARATE FEATURES AND TARGET
# ==========================================

# Convert target into numerical values

df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

X = df.drop("Churn", axis=1)

y = df["Churn"]

print("\nTarget Distribution:")
print(y.value_counts())


# ==========================================
# 4. PREPROCESSING
# ==========================================

# Identify numerical and categorical columns

numerical_features = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=np.number
).columns.tolist()


# Numerical preprocessing

numerical_transformer = StandardScaler()


# Categorical preprocessing

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore"
)


# Combine preprocessing steps

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# ==========================================
# 5. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# Calculate class imbalance ratio for XGBoost

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_weight = negative_count / positive_count


# ==========================================
# 6. DEFINE ORIGINAL MODELS
# ==========================================

# Logistic Regression

logistic_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ))
])


# AdaBoost

ada_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", AdaBoostClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=1,
            class_weight="balanced"
        ),
        n_estimators=150,
        learning_rate=0.5,
        random_state=42
    ))
])


# XGBoost

xgb_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    ))
])


# ==========================================
# 7. TRAIN ORIGINAL MODELS
# ==========================================

print("\nTraining Original Models...")


logistic_model.fit(X_train, y_train)

print("Logistic Regression Training Completed!")


ada_model.fit(X_train, y_train)

print("AdaBoost Training Completed!")


xgb_model.fit(X_train, y_train)

print("XGBoost Training Completed!")


# ==========================================
# 8. ORIGINAL SOFT VOTING
# ==========================================

voting_model = VotingClassifier(
    estimators=[
        ("lr", logistic_model),
        ("ada", ada_model),
        ("xgb", xgb_model)
    ],
    voting="soft"
)

voting_model.fit(X_train, y_train)

print("Original Soft Voting Training Completed!")


# ==========================================
# 9. ORIGINAL STACKING
# ==========================================

stacking_model = StackingClassifier(
    estimators=[
        ("lr", logistic_model),
        ("ada", ada_model),
        ("xgb", xgb_model)
    ],
    final_estimator=LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ),
    cv=5,
    stack_method="predict_proba"
)

stacking_model.fit(X_train, y_train)

print("Original Stacking Training Completed!")


# ==========================================
# 10. ADABOOST HYPERPARAMETER TUNING
# ==========================================

print("\nStarting AdaBoost Hyperparameter Tuning...")


ada_params = {
    "classifier__n_estimators": [50, 100, 150, 200, 300],

    "classifier__learning_rate": [
        0.01, 0.05, 0.1, 0.3, 0.5, 1.0
    ],

    "classifier__estimator__max_depth": [1, 2, 3]
}


ada_search = RandomizedSearchCV(
    estimator=ada_model,
    param_distributions=ada_params,
    n_iter=12,
    scoring="roc_auc",
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

ada_search.fit(X_train, y_train)


tuned_ada_model = ada_search.best_estimator_


print("\nBest AdaBoost Parameters:")
print(ada_search.best_params_)

print("Best AdaBoost CV ROC-AUC:")
print(ada_search.best_score_)


# ==========================================
# 11. XGBOOST HYPERPARAMETER TUNING
# ==========================================

print("\nStarting XGBoost Hyperparameter Tuning...")


# Create a separate pipeline for tuning

xgb_tuning_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        scale_pos_weight=scale_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=1
    ))
])


xgb_params = {
    "classifier__n_estimators": [100, 200, 300, 400],

    "classifier__max_depth": [2, 3, 4, 5, 6],

    "classifier__learning_rate": [
        0.01, 0.03, 0.05, 0.1, 0.2
    ],

    "classifier__subsample": [
        0.7, 0.8, 0.9, 1.0
    ],

    "classifier__colsample_bytree": [
        0.7, 0.8, 0.9, 1.0
    ],

    "classifier__min_child_weight": [
        1, 3, 5, 7
    ],

    "classifier__gamma": [
        0, 0.1, 0.2, 0.3
    ]
}


xgb_search = RandomizedSearchCV(
    estimator=xgb_tuning_model,
    param_distributions=xgb_params,
    n_iter=12,
    scoring="roc_auc",
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

xgb_search.fit(X_train, y_train)


tuned_xgb_model = xgb_search.best_estimator_


print("\nBest XGBoost Parameters:")
print(xgb_search.best_params_)

print("Best XGBoost CV ROC-AUC:")
print(xgb_search.best_score_)


# ==========================================
# 12. TUNED SOFT VOTING
# ==========================================

tuned_voting_model = VotingClassifier(
    estimators=[
        ("lr", logistic_model),
        ("ada", tuned_ada_model),
        ("xgb", tuned_xgb_model)
    ],
    voting="soft"
)

tuned_voting_model.fit(X_train, y_train)

print("\nTuned Soft Voting Training Completed!")


# ==========================================
# 13. TUNED STACKING
# ==========================================

tuned_stacking_model = StackingClassifier(
    estimators=[
        ("lr", logistic_model),
        ("ada", tuned_ada_model),
        ("xgb", tuned_xgb_model)
    ],
    final_estimator=LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ),
    cv=3,
    stack_method="predict_proba"
)

tuned_stacking_model.fit(X_train, y_train)

print("Tuned Stacking Training Completed!")


# ==========================================
# 14. MODEL COMPARISON
# ==========================================

models = {
    "Logistic Regression": logistic_model,

    "AdaBoost (Original)": ada_model,

    "XGBoost (Original)": xgb_model,

    "Soft Voting (Original)": voting_model,

    "Stacking (Original)": stacking_model,

    "AdaBoost (Tuned)": tuned_ada_model,

    "XGBoost (Tuned)": tuned_xgb_model,

    "Soft Voting (Tuned)": tuned_voting_model,

    "Stacking (Tuned)": tuned_stacking_model
}


results = []


for name, model in models.items():

    print("\nEvaluating:", name)

    y_pred = model.predict(X_test)

    y_probability = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test, y_probability
    )

    results.append({
        "Model": name,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4),
        "ROC AUC": round(roc_auc, 4)
    })


# Convert results into DataFrame

results_df = pd.DataFrame(results)

# Sort according to ROC-AUC

results_df = results_df.sort_values(
    by="ROC AUC",
    ascending=False
)

print("\n========== MODEL COMPARISON ==========")

print(results_df)

# Save comparison results

results_df.to_csv(
    "tuned_model_comparison.csv",
    index=False
)


# ==========================================
# 15. SELECT BEST MODEL
# ==========================================

best_index = results_df["ROC AUC"].idxmax()

best_name = results_df.loc[
    best_index, "Model"
]

best_model = models[best_name]


print("\nBest Model:", best_name)

print(
    "Best ROC-AUC:",
    results_df.loc[best_index, "ROC AUC"]
)


# Predictions using the best model

y_best_pred = best_model.predict(X_test)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_best_pred,
        target_names=["Stayed", "Churned"]
    )
)


# ==========================================
# 16. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test, y_best_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Stayed", "Churned"],
    yticklabels=["Stayed", "Churned"]
)

plt.title("Confusion Matrix - " + best_name)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()


# ==========================================
# 17. XGBOOST FEATURE IMPORTANCE
# ==========================================

# Use the tuned XGBoost model

trained_xgb = tuned_xgb_model


# Get feature importance

importance = trained_xgb.named_steps[
    "classifier"
].feature_importances_


feature_names = trained_xgb.named_steps[
    "preprocessor"
].get_feature_names_out()


importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})


importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)


print("\nTop 15 Important Features:")

print(importance_df.head(15))


# Plot feature importance

plt.figure(figsize=(10, 6))

sns.barplot(
    data=importance_df.head(15),
    x="Importance",
    y="Feature"
)

plt.title("Top 15 Churn Prediction Features")

plt.show()


# ==========================================
# 18. SAVE BEST MODEL
# ==========================================

os.makedirs("models", exist_ok=True)


joblib.dump(
    best_model,
    "models/best_churn_model.pkl"
)


print("\nBest Model Saved Successfully!")

print("\nTraining Completed!")
