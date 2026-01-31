# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ks_2samp

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# 2. STREAMLIT PAGE SETUP
# ==========================================

st.set_page_config(page_title="ML Classification Project", layout="wide")
st.title("📊 Machine Learning Classification Project")
st.write("Upload any CSV or Excel dataset to perform EDA and build ML models.")

# ==========================================
# 3. FILE UPLOADER
# ==========================================

uploaded_file = st.file_uploader(
    "Upload your dataset (CSV or Excel)",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is not None:

    # ==========================================
    # 4. LOAD DATA
    # ==========================================

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("Dataset loaded successfully!")
    st.write("### Dataset Preview")
    st.dataframe(df.head())
    st.write("Shape:", df.shape)

    # ==========================================
    # 5. DATA CLEANING
    # ==========================================

    df.drop_duplicates(inplace=True)
    df.columns = df.columns.str.lower().str.strip()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

    st.write("### Missing Values After Cleaning")
    st.write(df.isnull().sum())

    # ==========================================
    # 6. ENCODE CATEGORICAL FEATURES
    # ==========================================

    label_encoder = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = label_encoder.fit_transform(df[col])

    # ==========================================
    # 7. REMOVE ID COLUMNS
    # ==========================================

    for col in df.columns:
        if "id" in col:
            df.drop(columns=col, inplace=True)

    # ==========================================
    # 8. TARGET COLUMN SELECTION
    # ==========================================

    target_column = st.selectbox(
        "Select Target Column",
        df.columns,
        index=len(df.columns)-1
    )

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # ==========================================
    # 9. DATASET STABILITY & SCALING
    # ==========================================

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ==========================================
    # 10. EDA
    # ==========================================

    st.write("## 📈 Exploratory Data Analysis")
    st.write(df.describe())

    numeric_cols = df.select_dtypes(include=np.number).columns

    # KS Test
    if len(numeric_cols) >= 2:
        ks_stat, p_value = ks_2samp(df[numeric_cols[0]], df[numeric_cols[1]])
        st.write(f"**KS Statistic:** {ks_stat}")
        st.write(f"**P-Value:** {p_value}")

    # Scatter Plot
    if len(numeric_cols) >= 2:
        fig, ax = plt.subplots()
        ax.scatter(df[numeric_cols[0]], df[numeric_cols[1]])
        ax.set_xlabel(numeric_cols[0])
        ax.set_ylabel(numeric_cols[1])
        ax.set_title("Scatter Plot")
        st.pyplot(fig)

    # Box Plot
    fig, ax = plt.subplots(figsize=(10,5))
    df[numeric_cols].boxplot(ax=ax)
    ax.set_title("Box Plot")
    st.pyplot(fig)

    # Heatmap
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

    # Bar Graph
    fig, ax = plt.subplots()
    df[target_column].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Target Distribution")
    st.pyplot(fig)

    # ==========================================
    # 11. TRAIN-TEST SPLIT
    # ==========================================

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # ==========================================
    # 12. MODEL TRAINING
    # ==========================================

    st.write("## 🤖 Model Training & Evaluation")

    model_choice = st.selectbox(
        "Select Model",
        ["Logistic Regression", "KNN"]
    )

    if model_choice == "Logistic Regression":

        params = {"C": [0.01, 0.1, 1, 10]}
        grid = GridSearchCV(LogisticRegression(max_iter=1000), params, cv=5)
        grid.fit(X_train, y_train)

        model = grid.best_estimator_
        y_pred = model.predict(X_test)

    else:
        k = st.slider("Select K value", 1, 15, 5)
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    # ==========================================
    # 13. MODEL EVALUATION
    # ==========================================

    st.subheader("📊 Evaluation Metrics")

    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.write("Precision:", precision_score(y_test, y_pred, average="weighted"))
    st.write("F1 Score:", f1_score(y_test, y_pred, average="weighted"))

    st.write("### Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    st.success("🎉 Project Executed Successfully!")
