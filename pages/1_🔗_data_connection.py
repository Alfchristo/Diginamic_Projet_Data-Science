import streamlit as st
import pandas as pd
import mysql.connector
from pymongo import MongoClient


def main():
    st.title("Data Connection Page")
    st.write("This is the Data Connection page.")

    # Option to choose between file upload or database connection
    option = st.selectbox("Select Data Source", ["Upload CSV/Excel", "Connect to Database"])

    if option == "Upload CSV/Excel":
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.session_state['dataframe'] = df
                st.session_state['original_dataframe'] = df.copy()  # Initialize original_dataframe
                st.dataframe(df)
            except Exception as e:
                st.error(f"Error: {e}")

    elif option == "Connect to Database":
        db_type = st.selectbox("Select Database Type", ["MySQL", "MongoDB"])

        if db_type == "MySQL":
            mysql_host = st.text_input("MySQL Host", "localhost")
            mysql_user = st.text_input("MySQL User", "root")
            mysql_password = st.text_input("MySQL Password", type="password")
            mysql_database = st.text_input("MySQL Database")

            if st.button("Connect to MySQL"):
                try:
                    conn = mysql.connector.connect(
                        host=mysql_host,
                        user=mysql_user,
                        password=mysql_password,
                        database=mysql_database
                    )
                    cursor = conn.cursor()
                    cursor.execute("SHOW TABLES")
                    tables = [table[0] for table in cursor.fetchall()]
                    selected_table = st.selectbox("Select Table", tables)
                    if selected_table:
                        query = f"SELECT * FROM {selected_table}"
                        df = pd.read_sql(query, conn)
                        st.session_state['dataframe'] = df
                        st.session_state['original_dataframe'] = df.copy()  # Initialize original_dataframe
                        st.dataframe(df)
                except Exception as e:
                    st.error(f"Error: {e}")

        elif db_type == "MongoDB":
            mongo_host = st.text_input("MongoDB Host", "localhost")
            mongo_port = st.text_input("MongoDB Port", "27017")
            mongo_dbname = st.text_input("MongoDB Database")

            if st.button("Connect to MongoDB"):
                try:
                    client = MongoClient(f"mongodb://{mongo_host}:{mongo_port}/")
                    st.session_state['mongo_client'] = client
                    st.session_state['mongo_db'] = client[mongo_dbname]
                    st.session_state['connected'] = True
                except Exception as e:
                    st.error(f"Error: {e}")

            if st.session_state.get('connected'):
                db = st.session_state['mongo_db']
                collections = db.list_collection_names()
                selected_collection = st.selectbox("Select Collection", collections)
                if selected_collection:
                    collection = db[selected_collection]
                    df = pd.DataFrame(list(collection.find()))
                    if "_id" in df.columns:
                        df = df.drop(columns=["_id"])  # Drop the MongoDB default ID column
                    st.session_state['dataframe'] = df
                    st.session_state['original_dataframe'] = df.copy()  # Initialize original_dataframe
                    st.dataframe(df)


if __name__ == "__main__":
    if 'connected' not in st.session_state:
        st.session_state['connected'] = False
    main()
