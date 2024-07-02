import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import joblib


def main():
    st.title("Data Transformation Page")
    st.write("This is the Data Analysis page.")

    if 'dataframe' in st.session_state:
        df = st.session_state['dataframe']
        st.dataframe(df)

        # Handling Missing Data
        st.header("Handling Missing Data")
        if st.checkbox("Show Missing Data"):
            st.write(df.isnull().sum())

        missing_data_method = st.selectbox("Handle Missing Data",
                                           ["None", "Drop Rows", "Drop Columns", "Fill with Mean", "Fill with Median",
                                            "Fill with Mode", "Fill with Specific Value"])

        if missing_data_method == "Drop Rows":
            df = df.dropna()
        elif missing_data_method == "Drop Columns":
            df = df.dropna(axis=1)
        elif missing_data_method == "Fill with Mean":
            df = df.fillna(df.mean())
        elif missing_data_method == "Fill with Median":
            df = df.fillna(df.median())
        elif missing_data_method == "Fill with Mode":
            df = df.fillna(df.mode().iloc[0])
        elif missing_data_method == "Fill with Specific Value":
            fill_value = st.text_input("Enter value to fill missing data with")
            if fill_value:
                df = df.fillna(fill_value)

        st.dataframe(df)

        # Data Transformation
        st.header("Data Transformation")
        transformation_method = st.selectbox("Select Transformation Method",
                                             ["None", "Normalization (Min-Max)", "Standardization (Z-score)"])

        if transformation_method == "Normalization (Min-Max)":
            scaler = MinMaxScaler()
            df[df.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(
                df.select_dtypes(include=[np.number]))
        elif transformation_method == "Standardization (Z-score)":
            scaler = StandardScaler()
            df[df.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(
                df.select_dtypes(include=[np.number]))

        st.dataframe(df)

        # Encoding Categorical Variables
        st.header("Encoding Categorical Variables")
        encoding_method = st.selectbox("Select Encoding Method", ["None", "One-Hot Encoding", "Label Encoding"])

        if encoding_method == "One-Hot Encoding":
            df = pd.get_dummies(df)
        elif encoding_method == "Label Encoding":
            le = LabelEncoder()
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = le.fit_transform(df[col])

        st.dataframe(df)

        # Data Filtering and Selection
        st.header("Data Filtering and Selection")
        if st.checkbox("Remove Duplicates"):
            df = df.drop_duplicates()

        st.dataframe(df)

        columns_to_drop = st.multiselect("Select Columns to Drop", df.columns)
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)

        st.dataframe(df)

        # Date and Time Data Handling
        st.header("Date and Time Data Handling")
        date_columns = st.multiselect("Select Date Columns", df.columns[df.dtypes == 'object'])

        for date_col in date_columns:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df[f'{date_col}_day'] = df[date_col].dt.day
            df[f'{date_col}_month'] = df[date_col].dt.month
            df[f'{date_col}_year'] = df[date_col].dt.year

        st.dataframe(df)

        # Handling Outliers
        st.header("Handling Outliers")
        outlier_method = st.selectbox("Select Outlier Detection Method", ["None", "Z-score", "IQR"])

        if outlier_method == "Z-score":
            z_scores = np.abs((df - df.mean()) / df.std())
            df = df[(z_scores < 3).all(axis=1)]
        elif outlier_method == "IQR":
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

        st.dataframe(df)

        # Handling Data Types
        st.header("Handling Data Types")
        for col in df.columns:
            col_type = st.selectbox(f"Select type for column {col}", ["None", "int", "float", "str"], index=0)
            if col_type != "None":
                df[col] = df[col].astype(col_type)

        st.dataframe(df)

        # Save Processed Data / Reset DataFrame
        st.header("Save/Reset Processed Data")
        if st.button("Save Processed Data"):
            st.session_state['dataframe'] = df
            st.success("Processed data saved and updated!")

        if st.button("Reset DataFrame"):
            st.session_state['dataframe'] = st.session_state['original_dataframe']
            st.success("DataFrame reset to original state.")

    else:
        st.write("No data available. Please upload or connect to a data source on the Data Connection page.")


if __name__ == "__main__":
    main()
