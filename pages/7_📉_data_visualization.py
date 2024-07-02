import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    st.title("Data Visualization Page")
    st.write("This is the Data Visualization page.")

    # Check if dataframe is available in session state
    if 'dataframe' in st.session_state and st.session_state['dataframe'] is not None:
        df = st.session_state['dataframe']

        st.header("Select Visualization Options")

        # Select visualization type
        plot_types = st.multiselect("Select Visualization Type",
                                    ["Histogram", "Bar Plot", "Count Plot", "Scatter Plot", "Box Plot",
                                     "Correlation Heatmap"])

        # Select columns for visualization
        columns = df.columns.tolist()
        columns_to_visualize = st.multiselect("Select Columns for Visualization", columns)

        # Select target column for specific plots (e.g., scatter plot)
        target_column = st.selectbox("Select Target Column (for Scatter Plot)", columns)

        # Generate selected plots
        if "Histogram" in plot_types:
            st.header("Histogram")
            for column in columns_to_visualize:
                fig, ax = plt.subplots()
                sns.histplot(df[column], kde=True, ax=ax)
                ax.set_title(f"Histogram of {column}")
                st.pyplot(fig)

        if "Bar Plot" in plot_types:
            st.header("Bar Plot")
            for column in columns_to_visualize:
                fig, ax = plt.subplots()
                sns.countplot(x=column, data=df, ax=ax)
                ax.set_title(f"Bar Plot of {column}")
                ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig)

        if "Count Plot" in plot_types:
            st.header("Count Plot")
            for column in columns_to_visualize:
                fig, ax = plt.subplots()
                sns.countplot(y=column, data=df, ax=ax)
                ax.set_title(f"Count Plot of {column}")
                st.pyplot(fig)

        if "Scatter Plot" in plot_types and target_column:
            st.header("Scatter Plot")
            for column in columns_to_visualize:
                fig, ax = plt.subplots()
                sns.scatterplot(x=df[column], y=df[target_column], ax=ax)
                ax.set_title(f"Scatter Plot of {column} vs {target_column}")
                st.pyplot(fig)

        if "Box Plot" in plot_types:
            st.header("Box Plot")
            for column in columns_to_visualize:
                fig, ax = plt.subplots()
                sns.boxplot(x=column, data=df, ax=ax)
                ax.set_title(f"Box Plot of {column}")
                ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig)

        if "Correlation Heatmap" in plot_types:
            st.header("Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 8))
            corr = df.corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)

    else:
        st.write("No data available. Please upload or connect to a data source on the Data Connection page.")


if __name__ == "__main__":
    main()
