
# ==========================================
# CUSTOMER CHURN PREDICTION - EDA
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Dataset

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn (1).csv")

print("First 5 Rows:")
print(df.head())

# 2. Dataset Information

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

# 3. Check Missing Values

print("\nMissing Values:")
print(df.isnull().sum())

# 4. Convert TotalCharges into Numeric

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Fill missing values with 0
df["TotalCharges"] = df["TotalCharges"].fillna(0)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# 5. Check Duplicate Records

print("\nDuplicate Records:")
print(df.duplicated().sum())

# 6. Remove Customer ID

df.drop("customerID", axis=1, inplace=True)

# ==========================================
# DATA VISUALIZATION
# ==========================================

# 7. Churn Distribution

plt.figure(figsize=(6, 4))

sns.countplot(data=df, x="Churn")

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.show()

# 8. Churn Percentage

print("\nChurn Percentage:")

print(
    df["Churn"].value_counts(normalize=True) * 100
)

# 9. Churn Based on Contract Type

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Contract",
    hue="Churn"
)

plt.title("Churn Based on Contract Type")
plt.xticks(rotation=15)

plt.show()

# 10. Tenure Distribution

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="tenure",
    hue="Churn",
    bins=30,
    kde=True
)

plt.title("Customer Tenure Distribution")

plt.show()

# 11. Monthly Charges Analysis

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges"
)

plt.title("Monthly Charges vs Churn")

plt.show()

# 12. Total Charges Analysis

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="TotalCharges"
)

plt.title("Total Charges vs Churn")

plt.show()

# 13. Correlation Heatmap

numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(10, 6))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.show()

print("\nEDA Completed Successfully!")
