import streamlit as st

# Custom CSS for icons and banner
st.markdown("""
    <style>
        .sidebar .sidebar-content {
            padding-top: 0;
        }
        .icon {
            font-size: 1.2em;
            margin-right: 0.5em;
        }
        .banner {
            width: 100%;
            height: auto;
            border-radius: 10px;  /* Example border-radius */
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);  /* Example box-shadow */
        }
        .title {
            font-size: 2.5em;
            font-weight: bold;
            color: #333;  /* Example text color */
            margin-bottom: 20px;
        }
        .subtitle {
            font-size: 1.5em;
            color: #666;  /* Example text color */
            margin-bottom: 30px;
        }
        .instruction {
            font-size: 1.2em;
            color: #444;  /* Example text color */
        }
    </style>
""", unsafe_allow_html=True)

#Display Logo
st.logo("assets/images/cegefos.png", icon_image="assets/images/cegefos.png")

# Display The Banner
st.image("assets/images/banner.jpg", use_column_width=True)

# Title of the main page
st.markdown('<h1 class="title">Data Science App</h1>', unsafe_allow_html=True)

# Subtitle
st.markdown('<div class="subtitle">Welcome</div>', unsafe_allow_html=True)

# How to use this app section with icons
st.markdown("""
    <div class="instruction">
        <p>This app provides functionalities for:</p>
        <ul>
            <li><span class="icon">🔗</span>Data Connection: Upload your CSV file or connect to a database.</li>
            <li><span class="icon">📝</span>Data Description: Get a summary and overview of your dataset.</li>
            <li><span class="icon">📈</span>Data Analysis: Perform exploratory data analysis.</li>
            <li><span class="icon">🔄</span>Data Transformation: Apply transformations to your data.</li>
            <li><span class="icon">🏋️</span>Model Training: Train your machine learning models.</li>
            <li><span class="icon">📊</span>Model Evaluation: Evaluate the performance of your models.</li>
            <li><span class="icon">📉</span>Data Visualization: Visualize your data results.</li>
            <li><span class="icon">💡</span>Data Cleaning Suggestions: Get suggestions for cleaning your data.</li>
        </ul>
    </div>
""", unsafe_allow_html=True)
