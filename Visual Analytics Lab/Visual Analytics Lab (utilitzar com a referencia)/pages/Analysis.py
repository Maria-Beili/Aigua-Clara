import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('loan_dataset.csv')

st.set_page_config(
    page_title="Loan Approval Data Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Loan Approval Data Analysis")

# 1. Data Overview
st.subheader("Data Overview")

st.write("First 5 rows of the dataset:")
st.write(df.head())

st.write(f"Number of rows: {df.shape[0]}")
st.write(f"Number of columns: {df.shape[1]}")

st.write("Columns and their data types:")
col_types = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes,
})
st.write(col_types)

# 2. Missing Values
st.subheader("Missing Values")

missing = df.isnull().sum()
st.write(missing[missing > 0])

# 3. Data Distribution
st.subheader("Feature Distributions of Numerical Features")

numeric_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']

n_cols = 2
n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    sns.histplot(df[col], kde=True, ax=axes[i], bins=20, color='skyblue')
    axes[i].set_title(f"Distribution of {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Frequency")

for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

fig.tight_layout(pad=2.0)

st.pyplot(fig)

st.subheader("Feature Distributions of Categorical Features")

categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']

n_cols = 2
n_rows = (len(categorical_cols) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    value_counts = df[col].value_counts(normalize=True).dropna()
    sns.barplot(x=value_counts.index, y=value_counts.values, ax=axes[i])
    axes[i].set_title(f"Proportion of {col}")
    axes[i].set_ylabel("Proportion")
    axes[i].set_xlabel("")
    axes[i].tick_params(axis='x', rotation=45)

for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

fig.tight_layout(pad=2.0)

st.pyplot(fig)

# 4. Correlation Heatmap
st.subheader("Correlation Heatmap")

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
st.pyplot(fig)

# 5. Categorical Features vs Loan Status
st.subheader("Categorical Features vs Loan Status")

categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
n_cols = 2
n_rows = (len(categorical_cols) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    sns.countplot(data=df, x=col, hue='Loan_Status', ax=axes[i])
    axes[i].set_title(f"{col} vs Loan Status")
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Count")

for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

fig.tight_layout(pad=2.0)

st.pyplot(fig)