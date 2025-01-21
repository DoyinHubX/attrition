# Data Analysis and Visualizations Libraries
import streamlit as st
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.dataframe_explorer import dataframe_explorer
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

#ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score


import plotly.express as px
              

#STEP 1: Page Config
#------------------------------------------------------------------------------
# Layout setup
st.set_page_config(
    page_title="End-to-End HR Analytics: Predicting Employee Attrition with Streamlit",
    page_icon=":art:",
    layout="wide",
    #initial_sidebar_state="collapsed",
)

# Bootstrap CDN 
st.markdown("""
    <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
""", unsafe_allow_html=True)

# Inject custom CSS for styling
st.markdown("""
    <style>
        /* Target the main page container */
        div[data-testid="stAppViewContainer"] {
            margin-top: -70px; /* Adjust only the main page */
        }

        /* Style the sidebar container - background-color: #AABBCC; */
        [data-testid="stSidebar"] {
            background-color: #494858;
            padding: 0.2rem;
            margin-top: 60px; /* Adjust only the sidebar */
            border-bottom: 1px solid #ccc; /* Subtle border */
        }

        /* Style for the title */
        [data-testid="stSidebar"] h1 {
            color: #fff;
            font-size: 2.5rem;
            margin-bottom: 1rem;
        }

        /*  Custom CSS to beautify text */
        .processing-text {
            font-size: 20px;
            font-weight: bold;
            color: #4CAF50;
            background-color: #f4f4f4;
            padding: 10px;
            border-radius: 5px; 
        }
}
    </style>
""", unsafe_allow_html=True)



# STEP 2: Data and Sidebar Prep
#------------------------------------------------------------------------------
# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv") 
    return data

# Preload the dataset
data = load_data()

# Sidebar menu
with st.sidebar:
    selected_menu = option_menu(
        menu_title="HR Attrition Analytics",  
        options=["Overview", "Performance Insights", "Attrition Analysis", "Predictive Modeling"], 
        icons=["house", "bar-chart", "search", "graph-up"],  # Icon names from https://icons.getbootstrap.com/
        menu_icon="cast",  
        default_index=0,  
    )


# STEP 3: Overview Section
#------------------------------------------------------------------------------
if selected_menu == "Overview":
    
    # Create a selectbox to simulate tabs
    selected_tab = st.selectbox("Choose a Tab", ["Descriptive Stats", "Exploratory Data Analysis (EDA)"])

    # Display content based on the selected tab
    if selected_tab == "Descriptive Stats":
        st.header("Overview")
        st.write("""
        In the **Descriptive Statistics** section, we summarize the key statistics of the dataset to gain an understanding of the central tendency, dispersion, and shape of the data’s distribution.
        Key metrics like mean, median, standard deviation, and percentiles are calculated for each feature. These statistical summaries help in identifying patterns, outliers, and potential issues 
        in the data before performing more advanced analysis.
        """
        )
        st.subheader("Dataset Preview")
        st.dataframe(data.head())

        st.subheader("Summary Statistics")
        st.write(data.describe())

    elif selected_tab == "Exploratory Data Analysis (EDA)":
        st.header("Exploratory Data Analysis (EDA)")
        st.write("""
        In this section, we perform Exploratory Data Analysis (EDA) to understand the underlying patterns and distributions of various features in the dataset.
        We visualize different columns using suitable plots, such as bar charts, pie charts, and more. 
        """
        )

        # Dropdown for selecting the column
        col = st.selectbox("Select a column to visualize", ["JobRole", "Department", "Gender", "MaritalStatus"])

        # Check if data is loaded
        if data is not None:
            try:
                if col == "JobRole":  # Apply horizontal bar plot only for JobRole
                    # Count the occurrences of each category in the selected column
                    counts = data[col].value_counts()

                    # Create a Plotly bar chart
                    fig = px.bar(
                        x=counts.values,
                        y=counts.index,
                        orientation='h',
                        color=counts.values,
                        color_continuous_scale='viridis',
                        title=f"Distribution of {col}"
                    )

                    # Customize the plot
                    fig.update_layout(
                        xaxis_title="Count",
                        yaxis_title=col,
                        title_font_size=16,
                        xaxis=dict(title_font=dict(size=14)),
                        yaxis=dict(title_font=dict(size=14)),
                    )
                    fig.update_traces(marker=dict(line=dict(width=0.5, color="black")))

                    # Display the interactive plot in Streamlit
                    st.plotly_chart(fig)

                elif col == "Gender":  # Create pie chart for Gender
                    # Create a pie chart for the "Gender" column
                    fig, ax = plt.subplots(figsize=(6, 6))  
                    
                    # Count the occurrences of each category in the "Gender" column
                    gender_counts = data[col].value_counts()

                    # Plot pie chart
                    ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
                    ax.set_title(f"Gender Distribution", fontsize=16)

                    # Display the pie chart in Streamlit
                    st.pyplot(fig)

                else:
                    # Create a regular count plot for other columns
                    fig, ax = plt.subplots(figsize=(8, 5))  # Set figure size for better visuals
                    counts = data[col].value_counts()
                    ax.bar(counts.index, counts.values, color='skyblue')

                    # Customize the plot
                    ax.set_title(f"Distribution of {col}", fontsize=16)
                    ax.set_xlabel(col, fontsize=14)
                    ax.set_ylabel("Count", fontsize=14)
                    ax.tick_params(axis="x", rotation=45)  # Rotate x-axis labels if needed
                    ax.grid(axis="y", linestyle="--", alpha=0.7)

                    # Display the plot in Streamlit
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"An error occurred while plotting: {e}")
        else:
            st.warning("Please upload a dataset to perform EDA.")



# STEP 4: Performance Insights Section
#------------------------------------------------------------------------------
if selected_menu == "Performance Insights":
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

    # Selection Box for chart type
    chart_type = st.selectbox("Select a chart to view", 
                              ["Correlation Heatmap", 
                               "Performance Rating vs Monthly Income Distribution (Boxplot)",
                               "Monthly Income Distribution by Performance Rating"])

    # Display based on selected chart
    if chart_type == "Correlation Heatmap":
        # Correlation Heatmap
        corr_matrix = data[["PerformanceRating", "JobSatisfaction", "WorkLifeBalance"]].corr()
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    elif chart_type == "Performance Rating vs Monthly Income Distribution (Boxplot)":
        # Boxplot: Performance Rating vs Monthly Income Distribution
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x="PerformanceRating", y="MonthlyIncome", data=data, ax=ax, palette="viridis")
        ax.set_title("Performance Rating vs Monthly Income Distribution", fontsize=16, loc='center')
        ax.set_xlabel("Performance Rating", fontsize=14)
        ax.set_ylabel("Monthly Income", fontsize=14)
        st.pyplot(fig)

    elif chart_type == "Monthly Income Distribution by Performance Rating":
        # Histogram: Monthly Income Distribution for different Performance Ratings
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(data=data, x="MonthlyIncome", hue="PerformanceRating", kde=True, ax=ax, palette="Set1")
        ax.set_title("Monthly Income Distribution by Performance Rating", fontsize=16, loc='center')
        st.pyplot(fig)



#STEP 5: Attrition Analysis Section
#------------------------------------------------------------------------------
if selected_menu == "Attrition Analysis":
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

    # Selection Box for different chart types
    chart_type = st.selectbox("Select a chart to view", 
                              ["Attrition Breakdown", 
                               "Attrition by Department (Horizontal Bar Chart)",
                               "Attrition by JobRole",
                               "Attrition by Gender",
                               "Attrition by MaritalStatus",
                               "Age Distribution of Employees Leaving",
                               "Attrition vs Monthly Income",
                               "Attrition vs Work-Life Balance"])

    if chart_type == "Attrition Breakdown":
        # Pie Chart: Attrition Breakdown
        fig1, ax1 = plt.subplots()
        ax1.pie(
            attrition_counts, 
            labels=attrition_counts.index, 
            autopct="%1.1f%%", 
            colors=["#FF9999", "#66B3FF"]
        )
        ax1.set_title("Attrition Distribution")
        ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)  

        # Adjust layout for reduced padding/margins
        plt.tight_layout()
        st.pyplot(fig1)
        
    elif chart_type == "Attrition by Department (Horizontal Bar Chart)":
        # Horizontal Bar Chart: Attrition by Department
        attrition_dept = data.groupby("Department")["Attrition"].value_counts().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(10, 6))
        attrition_dept.plot(kind="barh", stacked=False, ax=ax, color=["#FF9999", "#66B3FF"])
        ax.set_title("Attrition by Department", fontsize=16)
        ax.set_xlabel("Count of Employees", fontsize=14)
        ax.set_ylabel("Department", fontsize=14)
        ax.legend(title="Attrition", loc="upper right", labels=["Stayed", "Left"])
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)  # Adding faint gridlines
        st.pyplot(fig)

    elif chart_type == "Attrition by JobRole":
        # Horizontal Bar Chart: Attrition by JobRole
        attrition_jobrole = data.groupby("JobRole")["Attrition"].value_counts().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(10, 6))
        attrition_jobrole.plot(kind="barh", stacked=False, ax=ax, color=["#FF9999", "#66B3FF"])
        ax.set_title("Attrition by JobRole", fontsize=16)
        ax.set_xlabel("Count of Employees", fontsize=14)
        ax.set_ylabel("JobRole", fontsize=14)
        ax.legend(title="Attrition", loc="upper right", labels=["Stayed", "Left"])
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)  # Adding faint gridlines
        st.pyplot(fig)

    elif chart_type == "Attrition by Gender":
        # Bar Chart: Attrition by Gender
        attrition_gender = data.groupby("Gender")["Attrition"].value_counts().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(8, 5))
        attrition_gender.plot(kind="bar", stacked=False, ax=ax, color=["#FF9999", "#66B3FF"])
        ax.set_title("Attrition by Gender", fontsize=16)
        ax.set_xlabel("Gender", fontsize=14)
        ax.set_ylabel("Count of Employees", fontsize=14)
        ax.legend(title="Attrition", loc="upper right", labels=["Stayed", "Left"])
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)  # Adding faint gridlines
        st.pyplot(fig)

    elif chart_type == "Attrition by MaritalStatus":
        # Bar Chart: Attrition by MaritalStatus
        attrition_marital = data.groupby("MaritalStatus")["Attrition"].value_counts().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(8, 5))
        attrition_marital.plot(kind="bar", stacked=False, ax=ax, color=["#FF9999", "#66B3FF"])
        ax.set_title("Attrition by MaritalStatus", fontsize=16)
        ax.set_xlabel("MaritalStatus", fontsize=14)
        ax.set_ylabel("Count of Employees", fontsize=14)
        ax.legend(title="Attrition", loc="upper right", labels=["Stayed", "Left"])
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)  # Adding faint gridlines
        st.pyplot(fig)

    elif chart_type == "Age Distribution of Employees Leaving":
        # Histogram: Age Distribution of Employees Who Left
        attrition_left = data[data["Attrition"] == "Yes"]
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(attrition_left["Age"], kde=True, ax=ax, color="#66B3FF")
        ax.set_title("Age Distribution of Employees Who Left", fontsize=16)
        ax.set_xlabel("Age", fontsize=14)
        ax.set_ylabel("Count", fontsize=14)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)  # Adding faint gridlines
        st.pyplot(fig)

    elif chart_type == "Attrition vs Monthly Income":
        # Scatter Plot: Attrition vs Monthly Income
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=data, x="MonthlyIncome", y="Age", hue="Attrition", palette="coolwarm", ax=ax)
        ax.set_title("Attrition vs Monthly Income", fontsize=16)
        ax.set_xlabel("Monthly Income", fontsize=14)
        ax.set_ylabel("Age", fontsize=14)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)  # Adding faint gridlines
        st.pyplot(fig)

    elif chart_type == "Attrition vs Work-Life Balance":
        # Violin Plot: Attrition vs Work-Life Balance
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.violinplot(x="Attrition", y="WorkLifeBalance", data=data, ax=ax, palette="coolwarm")
        ax.set_title("Attrition vs Work-Life Balance", fontsize=16)
        ax.set_xlabel("Attrition", fontsize=14)
        ax.set_ylabel("Work-Life Balance", fontsize=14)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)  # Adding faint gridlines
        st.pyplot(fig)
        st.write(
            """
            **Attrition vs Work-Life Balance**: This violin plot visualizes the distribution of work-life balance scores for employees who stayed vs. those who left the company. 
            The x-axis represents the attrition status (whether the employee stayed or left), while the y-axis shows the work-life balance scores. 
            The width of each violin plot represents the density of employees at each work-life balance score level, and the central line indicates the median value.
            """
        )


# STEP 6: Predictive Modeling Section
#------------------------------------------------------------------------------
if selected_menu == "Predictive Modeling":
    st.header("Predictive Modeling")
    st.write('<p class="custom-header">This section includes ML models for attrition prediction. The models aim to identify key factors contributing to employee attrition. By understanding these factors, businesses can take proactive steps to improve retention. </p>', unsafe_allow_html=True)

    # Step 6.1: Upload Dataset in the Sidebar
    #------------------------------------
    uploaded_file = st.sidebar.file_uploader("Upload a dataset for training", type=["csv"])

    if uploaded_file:
        data = pd.read_csv(uploaded_file)

        # Display the data
        st.write("Uploaded Dataset")
        
        # Use custom CSS to make the table text bolder
        st.markdown(
            """
            <style>
                .stDataFrame {
                    font-weight: bold;
                }
            </style>
            """, 
            unsafe_allow_html=True
        )
        
        # Display the table
        st.dataframe(data.head())

        # Step 6.2: Preprocessing
        #------------------------------------
        # Beautified Preprocessing Text using Bootstrap class
        st.markdown('<div class="alert alert-warning" role="alert"><strong>Preprocessing the dataset...done</strong></div>', unsafe_allow_html=True)
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

        # Step 6.3: Train the Model
        #------------------------------------
        #st.write("Training the Random Forest model...")
        st.markdown('<div class="alert alert-success" role="alert"><strong>Training the Random Forest model...done</strong></div>', unsafe_allow_html=True)
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Step 6.4: Evaluate the Model
        #------------------------------------
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]


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

        # SEP 6.5 Summary of model performance
        #------------------------------------
        # Create a collapsible section
        with st.expander("Summary of Model Performance"):
            st.subheader("General Summary")
            summary_text = f"""
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
            st.markdown(summary_text)
        
       
        # Step 6.6: Interactive Predictions with Two Columns Layout
        #------------------------------------
        with st.expander("Make Predictions"):
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
                st.metric(label="Predicted Attrition", value="Yes" if prediction == 1 else "No")
                st.metric(label="Attrition Probability", value=f"{prediction_prob:.2f}")


        # Step 6.6: Feature Importance Visualization
        #------------------------------------
        # Feature importance extraction
        with st.expander("Feature Importance"):
            feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

            # Apply the threshold of 4 to filter features
            threshold = 0.03  # Define the importance threshold (adjust as per your scale)
            filtered_importances = feature_importances[feature_importances > threshold]

            # Step 6.6: Plot the feature importance chart for the selected features
            
            # Generate a color map based on feature importance values
            colors = plt.cm.viridis(np.linspace(0, 1, len(filtered_importances)))

            # Plotting the feature importance horizontally with varying colors
            plt.figure(figsize=(10, 6))
            filtered_importances.sort_values(ascending=True).plot(kind='barh', color=colors)
            plt.title('Feature Importance')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()

            # Display the chart in Streamlit
            st.pyplot(plt)

        
            # Step 6.5: Display the filtered features and their importance in a DataFrame
            filtered_importances_df = filtered_importances.reset_index()
            filtered_importances_df.columns = ['Feature', 'Importance']

            # Display the filtered features and importance in a DataFrame
            st.write("Features with Importance Above Threshold (3%):")
            st.dataframe(filtered_importances_df)

       


      


