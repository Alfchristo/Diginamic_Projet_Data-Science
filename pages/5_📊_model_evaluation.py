import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score

# Function to fetch data from session state
def fetch_data():
    if 'dataframe' in st.session_state:
        return st.session_state['dataframe']
    else:
        st.warning("No dataframe found in session state.")
        return None

def main():
    st.title("Model Evaluation")

    # Fetch dataframe from session state
    df = fetch_data()

    if df is not None:
        # Feature Selection (Multiselect)
        st.write("Select Features:")
        selected_features = st.multiselect("Select features to include in the model", df.columns.tolist())

        # Target Selection (Dropdown)
        st.write("Select Target Column:")
        target_column = st.selectbox("Select the target column", df.columns.tolist())

        st.header("Model Evaluation Results")

        # Initialize models
        models = {
            "Linear Regression": LinearRegression(),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "SVM": SVC(),
            "Naive Bayes": GaussianNB(),
            "Random Forest": RandomForestClassifier()
        }

        # Evaluate each model
        best_model = None
        best_score = -1
        best_model_name = ""

        for model_name, model in models.items():
            st.subheader(model_name)

            # Split data into features and target
            X = df[selected_features]
            y = df[target_column]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

            # Train model
            model.fit(X_train, y_train)

            # Evaluate model
            if isinstance(model, LinearRegression):
                y_pred = model.predict(X_test)
                score = r2_score(y_test, y_pred)
                st.write(f"R-squared score: {score:.2f}")
                st.write("""
                **Interpretation**: The R-squared score measures how well the linear regression model fits the data. 
                A higher R-squared value (close to 1.0) indicates that the model explains a larger proportion of the variance in the target variable.
                """)

            elif isinstance(model, (LogisticRegression, DecisionTreeClassifier, RandomForestClassifier)):
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"Accuracy: {accuracy:.2f}")
                st.write("""
                **Interpretation**: Accuracy measures the percentage of correctly predicted instances out of total instances. 
                It indicates how well the model can classify or predict outcomes based on the input features.
                """)

            elif isinstance(model, SVC):
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"Accuracy: {accuracy:.2f}")
                st.write("""
                **Interpretation**: Accuracy measures the percentage of correctly classified instances out of total instances. 
                It assesses the model's ability to distinguish between different classes using the chosen kernel function and regularization parameter.
                """)

            elif isinstance(model, GaussianNB):
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"Accuracy: {accuracy:.2f}")
                st.write("""
                **Interpretation**: Accuracy measures the percentage of correctly classified instances out of total instances. 
                Naive Bayes assumes that features are conditionally independent, making it suitable for text classification or when features are relatively independent.
                """)

            # Track the best model based on score
            if isinstance(model, (LinearRegression, LogisticRegression, DecisionTreeClassifier, RandomForestClassifier)):
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_model_name = model_name
            elif isinstance(model, SVC) or isinstance(model, GaussianNB):
                if accuracy > best_score:
                    best_score = accuracy
                    best_model = model
                    best_model_name = model_name

        # Display best model and its performance
        if best_model:
            st.header("Best Model")
            st.write(f"The best model is: {best_model_name}")
            st.write(f"With a score of: {best_score:.2f}")
        else:
            st.warning("No models evaluated. Please check your selections.")

if __name__ == "__main__":
    main()
