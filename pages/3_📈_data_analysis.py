import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


def main():
    st.title("Data Analysis Page")
    st.write("This page provides Exploratory Data Analysis (EDA) insights and statistical analysis.")

    # Check if dataframe is available in session state
    if 'dataframe' in st.session_state and st.session_state['dataframe'] is not None:
        df = st.session_state['dataframe']

        st.header("Summary Statistics")
        st.write("Basic statistics of the dataset:")
        st.write(df.describe())
        st.write("""
        **Explanation**: Summary statistics provide a quick overview of numerical features in the dataset, including measures like mean, standard deviation, and quartiles. 
        They help in understanding the central tendency, dispersion, and range of values for each numerical column.
        """)

        st.header("Value Counts")
        st.write("Value counts for categorical variables:")
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            st.subheader(col)
            st.write(df[col].value_counts())
            st.write(f"""
            **Explanation**: Value counts show the frequency of each category in categorical variables. They help in understanding the distribution and imbalance of categories, which is important for feature engineering and modeling decisions.
            """)

        st.header("Statistical Analysis")

        # Example: Hypothesis Testing (t-test example)
        st.subheader("Hypothesis Testing (Example: t-test)")
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()  # Convert Index to list

        if len(numeric_cols) >= 2:
            col1 = st.selectbox("Select first numeric column for t-test", numeric_cols, index=0)
            col2 = st.selectbox("Select second numeric column for t-test", numeric_cols, index=1)
            if st.button("Run t-test"):
                t_stat, p_value = stats.ttest_ind(df[col1].dropna(), df[col2].dropna())
                st.write(f"T-test result between {col1} and {col2}:")
                st.write(f"T-statistic: {t_stat}")
                st.write(f"P-value: {p_value}")
                st.write("""
                **Explanation**: The t-test compares the means of two groups of numeric data to determine if they are significantly different from each other. 
                The T-statistic measures the difference relative to the variation in data, while the P-value indicates the significance of the difference.
                """)

                if p_value < 0.05:
                    st.write("There is a significant difference between the groups.")
                else:
                    st.write("There is no significant difference between the groups.")
                st.write("""
                **Interpretation**: If the p-value is less than 0.05, it suggests that the difference observed between the groups is unlikely due to random chance (i.e., statistically significant). 
                This could imply a meaningful difference between the variables under comparison.
                """)
        else:
            st.write("Insufficient numeric columns available for t-test. Requires at least two numeric columns.")
            st.write("""
            **Explanation**: The t-test requires at least two numeric columns with sufficient data points for comparison. Please ensure you have enough numeric columns to perform this test.
            """)

        # Example: Correlation Analysis
        st.subheader("Correlation Analysis")
        st.write("Correlation matrix:")
        corr_matrix = df.corr()
        st.write(corr_matrix)
        st.write("""
        **Explanation**: The correlation matrix shows the pairwise correlations between all numeric columns in the dataset. 
        Correlation coefficients range from -1 to +1, where +1 indicates a strong positive correlation, -1 indicates a strong negative correlation, and 0 indicates no correlation.
        This helps in identifying relationships between variables, which is crucial for feature selection and understanding dependencies in the data.
        """)

        # Example: Regression Analysis
        st.subheader("Regression Analysis (Example: Linear Regression)")
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 1:
            target_col = st.selectbox("Select target variable for regression", numeric_cols, index=0)
            feature_cols = st.multiselect("Select feature variables for regression", numeric_cols)
            if st.button("Run Regression"):
                X = df[feature_cols]
                y = df[target_col]

                # Ordinary Least Squares (OLS) regression using statsmodels
                X = sm.add_constant(X)  # Adding a constant for the intercept
                model = sm.OLS(y, X)
                results = model.fit()
                st.write(results.summary())

                # Scatter plot of actual vs predicted values
                y_pred = results.predict(X)
                fig, ax = plt.subplots()
                ax.scatter(y, y_pred)
                ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
                ax.set_xlabel('Actual')
                ax.set_ylabel('Predicted')
                ax.set_title('Actual vs Predicted')
                st.pyplot(fig)
                st.write("""
                **Explanation**: Linear regression models the relationship between a dependent variable (target) and one or more independent variables (features).
                The regression summary provides insights into the significance and coefficients of each feature, helping in interpreting their impact on the target variable.
                The scatter plot visualizes how well the model predictions match the actual values, assessing the model's performance.
                """)

                if results.rsquared > 0.5:
                    st.write("The model explains more than 50% of the variance in the target variable.")
                else:
                    st.write("The model explains less than 50% of the variance in the target variable.")
                st.write("""
                **Interpretation**: The R-squared value indicates how well the regression model fits the data. 
                An R-squared value closer to 1 suggests that the model explains a larger proportion of the variance in the target variable, indicating a better fit.
                """)

        else:
            st.write(
                "Insufficient numeric columns available for regression analysis. Requires at least one target variable and one feature variable.")
            st.write("""
            **Explanation**: Regression analysis requires numeric columns for both the target variable (dependent variable) and at least one feature variable (independent variable).
            Please ensure you have enough numeric columns selected for regression analysis.
            """)

    else:
        st.write("No data available. Please upload or connect to a data source on the Data Connection page.")


if __name__ == "__main__":
    main()
