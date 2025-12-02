# ===============================================================
# 1. Load libraries and dataset
# This section loads all necessary packages and reads the dataset.
# Only matplotlib.pyplot is used for visualizations.
# ===============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("abc_bank_churn.csv")  # use your real file name


# ===============================================================
# 2. Basic structure of the dataset
# Check shape, data types, head, and general information.
# ===============================================================

print("Dataset shape:", df.shape)
print("\nData types:\n", df.dtypes)

print("\nHead:\n", df.head())

print("\nInfo:")
df.info()

print("\nSummary statistics:\n", df.describe(include="all"))


# ===============================================================
# 3. Missing values analysis
# Inspect missing values count and percentage.
# ===============================================================

missing_count = df.isna().sum()
missing_percent = df.isna().mean() * 100

print("\nMissing values (count):\n", missing_count)
print("\nMissing values (percent):\n", missing_percent)


# ===============================================================
# 4. Duplicate records
# Check for complete duplicates and duplicates based on customer_id.
# ===============================================================

print("\nTotal full-row duplicates:", df.duplicated().sum())

if "customer_id" in df.columns:
    print("Duplicate customer_id:", df["customer_id"].duplicated().sum())


# ===============================================================
# 5. Numerical distributions
# Plot histograms for all numerical columns using matplotlib.
# ===============================================================

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

for col in numeric_cols:
    plt.figure(figsize=(6,4))
    plt.hist(df[col].dropna(), bins=30)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


# ===============================================================
# 6. Boxplots for outlier detection
# Visual inspection of potential outliers.
# ===============================================================

for col in numeric_cols:
    plt.figure(figsize=(5,4))
    plt.boxplot(df[col].dropna(), vert=True)
    plt.title(f"Boxplot of {col}")
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()


# ===============================================================
# 7. Categorical feature analysis
# Display category frequencies and simple bar charts.
# ===============================================================

categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

for col in categorical_cols:
    print(f"\nValue counts for {col}:\n", df[col].value_counts())

    plt.figure(figsize=(6,4))
    df[col].value_counts().plot(kind="bar")
    plt.title(f"Category distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


# ===============================================================
# 8. Correlation matrix (numerical only)
# Compute correlations and plot them using imshow.
# ===============================================================

corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(8,6))
plt.imshow(corr_matrix, cmap="viridis", interpolation="nearest")
plt.colorbar()
plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
plt.yticks(range(len(numeric_cols)), numeric_cols)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

print("\nCorrelation matrix:\n", corr_matrix)


# ===============================================================
# 9. Relationship between numerical features and churn
# For the binary target 'churn', compare distributions across classes.
# ===============================================================

if "churn" in df.columns:
    for col in numeric_cols:
        if col != "churn":
            plt.figure(figsize=(6,4))
            plt.hist(df[df["churn"] == 0][col].dropna(), bins=30, alpha=0.5, label="churn=0")
            plt.hist(df[df["churn"] == 1][col].dropna(), bins=30, alpha=0.5, label="churn=1")
            plt.title(f"{col} by churn class")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.legend()
            plt.tight_layout()
            plt.show()


# ===============================================================
# 10. Relationship between categorical features and churn
# Simple bar plots comparing churn rates across categories.
# ===============================================================

if "churn" in df.columns:
    for col in categorical_cols:
        churn_rate = df.groupby(col)["churn"].mean()

        print(f"\nChurn rate by {col}:\n", churn_rate)

        plt.figure(figsize=(6,4))
        churn_rate.plot(kind="bar")
        plt.title(f"Churn rate by {col}")
        plt.xlabel(col)
        plt.ylabel("Churn rate")
        plt.tight_layout()
        plt.show()


# ===============================================================
# 11. Initial observations and data-quality checks
# This section simply prints additional checks.
# ===============================================================

print("\nUnique values per column:\n", df.nunique())

if "age" in df.columns:
    print("\nAge range:", df["age"].min(), "-", df["age"].max())

if "credit_score" in df.columns:
    print("Credit score range:", df["credit_score"].min(), "-", df["credit_score"].max())

if "balance" in df.columns:
    print("Balance negative values:", (df["balance"] < 0).sum())

****