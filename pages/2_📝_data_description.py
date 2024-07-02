import streamlit as st
import pandas as pd


def main():
    st.title("Data Description Page")
    st.write("This is the Data Description page.")

    if 'dataframe' in st.session_state:
        df = st.session_state['dataframe']

        # Display basic descriptive statistics
        st.header("Descriptive Statistics")
        st.write(df.describe(include='all'))

        # Display data types
        st.header("Data Types")
        st.write(df.dtypes)

        # Display missing values
        st.header("Missing Values")
        st.write(df.isnull().sum())

        # Display unique values for categorical columns
        st.header("Unique Values in Categorical Columns")
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            st.subheader(f"Column: {col}")
            st.write(df[col].value_counts())

    else:
        st.write("No data available. Please upload or connect to a data source on the Data Connection page.")


if __name__ == "__main__":
    main()
