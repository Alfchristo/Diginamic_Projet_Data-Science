import numpy as np
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import stats
from sklearn.preprocessing import LabelEncoder

def main():
    st.title("Data Cleaning Advice Page")
    st.write("This page provides data cleaning suggestions based on your dataset.")

    # Check if dataframe is available in session state
    if 'dataframe' in st.session_state and st.session_state['dataframe'] is not None:
        df = st.session_state['dataframe']

        st.header("Data Cleaning Suggestions")

        # 1. Correlation Analysis
        st.subheader("Correlation Analysis")
        corr = df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title("Correlation Heatmap")
        st.pyplot()

        high_corr_pairs = [(corr.columns[i], corr.columns[j]) for i, j in zip(*np.where(corr > 0.8)) if i != j]
        if high_corr_pairs:
            st.write("Highly correlated columns (> 0.8):")
            for pair in high_corr_pairs:
                st.write(f"- {pair}")

        # 2. Missing Data Detection
        st.subheader("Missing Data Detection")
        missing_data = df.isnull().mean() * 100
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        if not missing_data.empty:
            st.write("Columns with missing data (%):")
            st.write(missing_data)

        # 3. Outlier Detection (example)
        st.subheader("Outlier Detection")
        numerical_cols = df.select_dtypes(include=['number']).columns
        for col in numerical_cols:
            # Example: Z-score method for outlier detection
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outliers = df[(z_scores > 3)]
            if not outliers.empty:
                st.write(f"Outliers detected in {col}:")
                st.write(outliers.head())

        # 4. Data Types
        st.subheader("Data Types")
        data_types = df.dtypes
        st.write("Current data types:")
        st.write(data_types)

        # Example: Detecting categorical columns and suggesting label encoding
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if not categorical_cols.empty:
            st.write("Categorical columns detected:")
            st.write(categorical_cols)

            # Example: Label encoding suggestion
            le = LabelEncoder()
            encoded = df[categorical_cols].apply(le.fit_transform)
            st.write("Example of label encoding:")
            st.write(encoded.head())

        # 5. Duplicate Rows
        st.subheader("Duplicate Rows")
        duplicate_rows = df[df.duplicated()]
        if not duplicate_rows.empty:
            st.write("Duplicate rows detected:")
            st.write(duplicate_rows.head())

        # 6. Potential ID Columns
        st.subheader("Potential ID Columns")
        id_columns = []
        for col in df.columns:
            if df[col].nunique() == df.shape[0]:  # All values are unique
                id_columns.append(col)
        if id_columns:
            st.write("Potential ID columns detected:")
            st.write(id_columns)

    else:
        st.write("No data available. Please upload or connect to a data source on the Data Connection page.")


if __name__ == "__main__":
    main()
