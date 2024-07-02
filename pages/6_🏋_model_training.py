import streamlit as st
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score
import uuid

# Function to fetch data from session state
def fetch_data():
    if 'dataframe' in st.session_state:
        return st.session_state['dataframe']
    else:
        st.warning("No dataframe found in session state.")
        return None

def main():
    # Fetch existing trained_models from session state, or initialize if not present
    trained_models = st.session_state.get('trained_models', {})

    st.title("Machine Learning Model Training")

    st.header("Feature and Target Selection")

    # Fetch dataframe from session state
    df = fetch_data()

    if df is not None:
        # Feature Selection (Multiselect)
        st.write("Select Features:")
        selected_features = st.multiselect("Select features to include in the model", df.columns.tolist())

        # Target Selection (Dropdown)
        st.write("Select Target Column:")
        target_column = st.selectbox("Select the target column", df.columns.tolist())

        st.header("Model Selection")

        # Suggest models based on dataset characteristics
        numeric_cols = df.select_dtypes(include=['number']).shape[1]
        text_cols = df.select_dtypes(include=['object']).shape[1]
        categorical_data = False  # Placeholder for categorical data check (to be implemented based on your data)
        data_size = df.shape[0]

        if numeric_cols > 0:
            st.write("Numeric Columns Detected.")
            st.write("Suggested Models:")
            st.write("- Linear Regression")
            st.write("- Decision Tree")
            st.write("- SVM")

        if text_cols > 0:
            st.write("Text Columns Detected.")
            st.write("Suggested Models:")
            st.write("- Naive Bayes")

        if categorical_data:
            st.write("Categorical Data Detected.")
            st.write("Suggested Models:")
            st.write("- Logistic Regression")
            st.write("- Decision Tree")
            st.write("- Random Forest")

        st.header("Model Training")

        model_name = st.selectbox("Select Model", ["Linear Regression", "Logistic Regression",
                                                   "Decision Tree", "SVM", "Naive Bayes",
                                                   "Random Forest"])

        if model_name == "Linear Regression":
            st.write("Adjust Parameters:")
            fit_intercept = st.checkbox("Fit Intercept", value=True)
            model = LinearRegression(fit_intercept=fit_intercept)

        elif model_name == "Logistic Regression":
            st.write("Adjust Parameters:")
            C = st.slider("Regularization strength (C)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
            model = LogisticRegression(C=C, max_iter=1000)

        elif model_name == "Decision Tree":
            st.write("Adjust Parameters:")
            criterion = st.selectbox("Criterion", ["gini", "entropy"])
            max_depth = st.slider("Max Depth", min_value=1, max_value=20, value=3, step=1)
            model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)

        elif model_name == "SVM":
            st.write("Adjust Parameters:")
            C = st.slider("Regularization parameter (C)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
            kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
            model = SVC(C=C, kernel=kernel)

        elif model_name == "Naive Bayes":
            st.write("Adjust Parameters:")
            model = GaussianNB()

        elif model_name == "Random Forest":
            st.write("Adjust Parameters:")
            n_estimators = st.slider("Number of estimators", min_value=10, max_value=200, value=100, step=10)
            max_depth = st.slider("Max Depth", min_value=1, max_value=20, value=None, step=1)
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

        if st.button("Train Model"):
            if len(selected_features) == 0:
                st.warning("Please select at least one feature.")
                return

            if not target_column:
                st.warning("Please select a target column.")
                return

            # Split data into features and target
            X = df[selected_features]
            y = df[target_column]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if isinstance(model, LinearRegression):
                score = r2_score(y_test, y_pred)
                st.write(f"R-squared score: {score:.2f}")
                st.write("""
                **Interpretation**: The R-squared score measures how well the linear regression model fits the data. 
                A higher R-squared value (close to 1.0) indicates that the model explains a larger proportion of the variance in the target variable.
                """)

            elif isinstance(model, LogisticRegression) or isinstance(model, DecisionTreeClassifier) or isinstance(model,
                                                                                                                  RandomForestClassifier):
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"Accuracy: {accuracy:.2f}")
                st.write("""
                **Interpretation**: Accuracy measures the percentage of correctly predicted instances out of total instances. 
                It indicates how well the model can classify or predict outcomes based on the input features.
                """)

            elif isinstance(model, SVC):
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"Accuracy: {accuracy:.2f}")
                st.write("""
                **Interpretation**: Accuracy measures the percentage of correctly classified instances out of total instances. 
                It assesses the model's ability to distinguish between different classes using the chosen kernel function and regularization parameter.
                """)

            elif isinstance(model, GaussianNB):
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"Accuracy: {accuracy:.2f}")
                st.write("""
                **Interpretation**: Accuracy measures the percentage of correctly classified instances out of total instances. 
                Naive Bayes assumes that features are conditionally independent, making it suitable for text classification or when features are relatively independent.
                """)

            # Generate a unique identifier for the model
            model_id = str(uuid.uuid4())

            # Store the trained model in the session state
            trained_models[model_id] = model

            # Export trained model
            st.write(f"Export Trained Model (ID: {model_id}):")
            filename = f"trained_model_{model_name}_{model_id}.pkl"
            st.markdown(f"Download model [here](./{filename})")
            with open(filename, 'wb') as file:
                pickle.dump(model, file)

            # Update session state with trained_models
            # Store X_test and y_test in session state
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state['trained_models'] = trained_models

        st.header("Download Trained Models")

        # Display download buttons for each trained model
        for model_id, model in trained_models.items():
            filename = f"trained_model_{model_name}_{model_id}.pkl"
            st.download_button(
                label=f"Download Model {model_id}",
                data=pickle.dumps(model),
                file_name=filename,
                mime="application/octet-stream"
            )

    else:
        st.warning("No dataframe found in session state. Please upload or connect to data first.")

if __name__ == "__main__":
    main()
