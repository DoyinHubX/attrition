import streamlit as st
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.dataframe_explorer import dataframe_explorer
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score


#from streamlit_kpi import kpi


st.set_page_config(layout="wide")


import streamlit as st

# Apply custom CSS to adjust margins and padding
st.markdown(
    """
    <style>
    /* Remove top margin */
    .css-1d391kg {
        margin-top: -80px;  /* Adjust the value to your needs */
    }
    
    </style>
    """,
    unsafe_allow_html=True
)


# Custom CSS to beautify text
st.markdown("""
    <style>
        .processing-text {
            font-size: 20px;
            font-weight: bold;
            color: #4CAF50;
            background-color: #f4f4f4;
            padding: 10px;
            border-radius: 5px;
           
        }
    </style>
""", unsafe_allow_html=True)

 #text-align: center;

 # Add Bootstrap CDN to the app
st.markdown("""
    <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
""", unsafe_allow_html=True)

# Inject custom CSS for styling
st.markdown("""
    <style>
        .custom-header {
            font-size: 24px;
            font-weight: bold;
            color: #007BFF;
            margin-top: 20px;
        }
        .custom-subheader {
            font-size: 20px;
            font-weight: 600;
            color: #28A745;
            margin-top: 10px;
        }
        .custom-line {
            border-top: 2px solid #007BFF;
            margin: 20px 0;
        }
        .custom-text {
            font-size: 16px;
            font-weight: 500;
            color: #6c757d;
        }
    </style>
""", unsafe_allow_html=True)


# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")  # Replace with your dataset path
    return data

# Preload the dataset
data = load_data()

# Sidebar
st.sidebar.title("HR Analytics Dashboard")
menu = st.sidebar.radio("Navigation", ["Overview",  "Performance Insights", "Attrition Analysis", "Predictive Modeling"])

# Header
#st.title("HR Analytics Employee Attrition & Performance Dashboard")

# Overview Section
#------------------------------------------------------------------------------
if menu == "Overview":
    
    # Create three columns to align buttons closely
    col1, col2, col3 = st.columns([1, 1, 1])

    # Create buttons in columns for tabs
    with col1:
        if st.button("Descriptive Stats"):
            selected_tab = "Descriptive Stats"
    with col2:
        if st.button("Exploratory Data Analysis (EDA)"):
            selected_tab = "EDA"
    with col3:
        if st.button("Tab 3"):
            selected_tab = "Tab 3"

    # Default to "Tab 1" if no tab is selected
    #-----------------------------------------------
    if 'selected_tab' not in locals():
        selected_tab = "Descriptive Stats"

    # Display content based on the selected tab
    if selected_tab == "Descriptive Stats":
        st.header("Overview")
        st.write("This dashboard provides insights into employee attrition and performance with the help of exploratory data analysis (EDA), machine learning models, and interactive visualizations.")

        st.subheader("Dataset Preview")
        st.dataframe(data.head())

        st.subheader("Summary Statistics")
        st.write(data.describe())


    elif selected_tab == "EDA":
        st.header("Exploratory Data Analysis (EDA)")
    
        # Select visualization type
        viz_type = st.selectbox("Select a chart type", ["Bar Chart", "Heatmap", "Distribution Plot"])
        
        if viz_type == "Bar Chart":
            col = st.selectbox("Select a column", ["Department", "Gender", "JobRole", "MaritalStatus"])
            fig, ax = plt.subplots()
            sns.countplot(data=data, x=col, ax=ax)
            st.pyplot(fig)

        elif viz_type == "Heatmap":
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        elif viz_type == "Distribution Plot":
            col = st.selectbox("Select a numeric column", ["MonthlyIncome", "Age", "YearsAtCompany"])
            fig, ax = plt.subplots()
            sns.histplot(data[col], kde=True, ax=ax)
            st.pyplot(fig)

    else:
        st.subheader("This is Tab 3")
        st.write("Content for Tab 3")



# Performance Insights Section
#------------------------------------------------------------------------------
from streamlit_extras.metric_cards import style_metric_cards

# Tabbed Layout for Analysis Sections
if menu == "Performance Insights":
    st.title("Performance Insights")

    # Metric Cards
    avg_perf_rating = data["PerformanceRating"].mean().round(2)
    avg_job_sat = data["JobSatisfaction"].mean().round(2)
    avg_worklife = data["WorkLifeBalance"].mean().round(2)
    style_metric_cards()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Performance Rating", avg_perf_rating)
    with col2:
        st.metric("Avg Job Satisfaction", avg_job_sat)
    with col3:
        st.metric("Avg Work-Life Balance", avg_worklife)

    # Heatmap: Correlation of Performance Factors
    st.subheader("Correlation Heatmap")
    corr_matrix = data[["PerformanceRating", "JobSatisfaction", "WorkLifeBalance"]].corr()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Line Plot: Trends in Monthly Income by Performance Rating
    st.subheader("Monthly Income vs Performance Rating")
    income_by_rating = data.groupby("PerformanceRating")["MonthlyIncome"].mean()
    st.line_chart(income_by_rating)

if menu == "Attrition Analysis":
    st.title("Attrition Analysis")

    # Metric Cards
    attrition_counts = data["Attrition"].value_counts()
    total_employees = len(data)
    attrition_rate = (attrition_counts.get("Yes", 0) / total_employees * 100).round(2)
    avg_income = data["MonthlyIncome"].mean().round(2)

    style_metric_cards()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Attrition Count (Yes)", attrition_counts.get("Yes", 0))
    with col2:
        st.metric("Attrition Count (No)", attrition_counts.get("No", 0))
    with col3:
        st.metric("Attrition Rate (%)", attrition_rate)

    # Pie Chart: Attrition Breakdown
    # Create columns for side-by-side display
    col1, col2 = st.columns(2)

    # Pie Chart: Attrition Breakdown
    with col1:
        st.subheader("Attrition Breakdown")
        fig1, ax1 = plt.subplots()
        ax1.pie(
            attrition_counts, 
            labels=attrition_counts.index, 
            autopct="%1.1f%%", 
            colors=["#FF9999", "#66B3FF"]
        )
        ax1.set_title("Attrition Distribution")
        st.pyplot(fig1)

    # Bar Chart: Attrition by Department
    with col2:
        st.subheader("Attrition by Department")
        attrition_dept = data.groupby("Department")["Attrition"].value_counts().unstack(fill_value=0)
        st.bar_chart(attrition_dept)

# Predictive Modeling Section
#------------------------------------------------------------------------------
if menu == "Predictive Modeling":
    st.header("Predictive Modeling")
    st.markdown('<div class="custom-line"></div>', unsafe_allow_html=True)  # Horizontal line

    st.write('<p class="custom-header">This section includes ML models for attrition prediction.</p>', unsafe_allow_html=True)

    # Step 1: Upload Dataset in the Sidebar
    uploaded_file = st.sidebar.file_uploader("Upload a dataset for training", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)

        # Display the data
        st.write("Uploaded Dataset")
        st.dataframe(data.head())


        # Step 2: Preprocessing
        # Beautified Preprocessing Text using Bootstrap class
        st.markdown('<div class="alert alert-info" role="alert"><strong>Preprocessing the dataset...done</strong></div>', unsafe_allow_html=True)
        categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
        data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
        data['Attrition'] = data['Attrition'].map({'Yes': 1, 'No': 0})
        X = data.drop(['Attrition', 'EmployeeNumber', 'Over18', 'StandardHours', 'EmployeeCount'], axis=1, errors='ignore')
        y = data['Attrition']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Feature scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Step 3: Train the Model
        #st.write("Training the Random Forest model...")
        st.markdown('<div class="alert alert-warning" role="alert"><strong>Training the Random Forest model...done</strong></div>', unsafe_allow_html=True)
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Step 4: Evaluate the Model
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # st.subheader("Model Performance Metrics")
        # st.write("Classification Report:")
        # st.text(classification_report(y_test, y_pred))

        # roc_auc = roc_auc_score(y_test, y_prob)
        # st.write(f"ROC-AUC Score: {roc_auc:.2f}")

            # Model Performance Metrics Section
        st.subheader("Model Performance Metrics")

        # Convert the classification report to a DataFrame
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        # Display the classification report as a table
        st.write('<p class="custom-text">Classification Report:</p>', unsafe_allow_html=True)
        st.table(report_df)

        # Calculate and display the ROC-AUC Score using st.metric for emphasis
        roc_auc = roc_auc_score(y_test, y_prob)
        st.metric(label="ROC-AUC Score", value=f"{roc_auc:.2f}", delta="")

        # Summary of model performance
        st.subheader("General Summary")
        summary_text = f"""
        **Model Overview:**
        - The model is designed to predict employee attrition, aiming to identify which employees are more likely to leave the company.
        - The performance is evaluated based on various metrics, including precision, recall, and F1-score.

        **Key Findings:**
        - The ROC-AUC score is **{roc_auc:.2f}**, indicating how well the model can distinguish between employees who will stay vs. those who will leave.
        - The classification report shows the precision, recall, and F1-score for both classes:
          - **Attrition (Yes)**: The model is optimized to accurately predict attrition cases, ensuring timely interventions.
          - **Non-Attrition (No)**: The model aims to accurately classify employees who are likely to stay, helping in retaining key employees.

        **Next Steps:**
        - Based on the results, we can fine-tune the model by experimenting with different algorithms, hyperparameters, or additional features.
        - It's also important to monitor the model performance regularly with updated datasets to ensure its effectiveness.

        This report provides actionable insights into the model's performance, which can be used to inform business decisions regarding employee retention strategies.
        """
        
        # Render the summary
        st.markdown(summary_text)

        # Optional: Beautify the Classification Report table with Bootstrap (injecting HTML)
        st.markdown("""
        <style>
            .table {
                width: 100%;
                border-collapse: collapse;
            }
            .table, th, td {
                border: 1px solid #ddd;
            }
            .table th {
                background-color: #f8f9fa;
                color: #343a40;
            }
            .table td {
                text-align: center;
                color: #6c757d;
            }
            .table th, .table td {
                padding: 8px;
            }
        </style>
        """, unsafe_allow_html=True)

        st.markdown(f'<table class="table">{report_df.to_html(index=False)}</table>', unsafe_allow_html=True)

        # Step 5: Feature Importance Visualization
        st.subheader("Feature Importance")
        feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        st.bar_chart(feature_importances)

        # Step 6: Interactive Predictions with Two Columns Layout
        st.subheader("Make Predictions")

        # Default values for the input form
        input_data = {
            'Age': 30, 
            'DistanceFromHome': 5, 
            'MonthlyIncome': 5000, 
            'YearsAtCompany': 5, 
            'JobSatisfaction': 3, 
            'OverTime_Yes': 0  # Assuming 'No' for OverTime
        }

        # Create two columns
        col1, col2 = st.columns(2)

        # Categorical features in the first column
        with col1:
            for col in X.columns:
                if col.startswith(tuple(categorical_cols)):  # Categorical features
                    input_data[col] = st.selectbox(f"{col}", [0, 1], index=0 if input_data.get(col) == 0 else 1)

        # Numerical features in the second column
        with col2:
            for col in X.columns:
                if not col.startswith(tuple(categorical_cols)):  # Numerical features
                    input_data[col] = st.number_input(f"{col}", value=input_data.get(col, 0))

        # Predict using user input
        if st.button("Predict Attrition"):
            input_df = pd.DataFrame([input_data])

            # Align the columns with the training data (ensure the input data has the same columns)
            input_df = input_df.reindex(columns=X.columns, fill_value=0)  # Fill missing columns with 0 if needed

            # Now, transform the input data using the same scaler as the training data
            input_scaled = scaler.transform(input_df)
            
            # Make prediction and probability calculation
            prediction = model.predict(input_scaled)[0]
            prediction_prob = model.predict_proba(input_scaled)[0][1]

            # Display prediction results
            prediction_text = "Attrition Prediction: **" + ('Yes' if prediction == 1 else 'No') + "**"
            prob_text = f"Probability of Attrition: **{prediction_prob:.2f}**"

            # Display results
            st.markdown(f"<h3 style='text-align: center; color: #4CAF50;'>{prediction_text}</h3>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='text-align: center; color: #FF6347;'>{prob_text}</h4>", unsafe_allow_html=True)
            st.metric(label="Predicted Attrition", value="Yes" if prediction == 1 else "No")
            st.metric(label="Attrition Probability", value=f"{prediction_prob:.2f}")


        # # Predict using user input
        # if st.button("Predict Attrition"):
        #     input_df = pd.DataFrame([input_data])
        #     input_scaled = scaler.transform(input_df)
        #     prediction = model.predict(input_scaled)[0]
        #     prediction_prob = model.predict_proba(input_scaled)[0][1]

        #     # Beautify the prediction display
        #     prediction_text = "Attrition Prediction: **" + ('Yes' if prediction == 1 else 'No') + "**"
        #     prob_text = f"Probability of Attrition: **{prediction_prob:.2f}**"

        #     # Display prediction and probability with markdown for emphasis
        #     st.markdown(f"<h3 style='text-align: center; color: #4CAF50;'>{prediction_text}</h3>", unsafe_allow_html=True)
        #     st.markdown(f"<h4 style='text-align: center; color: #FF6347;'>{prob_text}</h4>", unsafe_allow_html=True)

        #     # Alternatively, use st.metric for a clean display
        #     st.metric(label="Predicted Attrition", value="Yes" if prediction == 1 else "No")
        #     st.metric(label="Attrition Probability", value=f"{prediction_prob:.2f}")




