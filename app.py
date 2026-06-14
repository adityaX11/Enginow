import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)
import joblib
import os
st.markdown("""
<style>

/* ===============================
   GLOBAL APP
================================ */
.stApp{
    background:
    radial-gradient(circle at top left, #6d28d9 0%, transparent 30%),
    radial-gradient(circle at bottom right, #2563eb 0%, transparent 30%),
    #0f172a;
    color:white;
}

/* Main container */
.block-container{
    padding-top:2rem;
    max-width:1300px;
}

/* ===============================
   TITLE
================================ */
h1{
    text-align:center;
    font-size:3rem !important;
    font-weight:800 !important;
    background:linear-gradient(90deg,#8b5cf6,#06b6d4,#ec4899);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    margin-bottom:15px;
}

h2,h3{
    color:#f8fafc !important;
    border-left:5px solid #8b5cf6;
    padding-left:10px;
}

/* ===============================
   TEXT
================================ */
p, label, div{
    color:#e2e8f0;
}

/* ===============================
   GLASS CARDS
================================ */
[data-testid="stDataFrame"],
[data-testid="stMetric"],
.element-container{
    background:rgba(255,255,255,0.05);
    backdrop-filter:blur(18px);
    border:1px solid rgba(255,255,255,0.1);
    border-radius:20px;
}

/* ===============================
   BUTTONS
================================ */
.stButton button{
    background:linear-gradient(
    135deg,
    #7c3aed,
    #2563eb
    );

    color:white !important;
    border:none;
    border-radius:12px;
    font-weight:700;
    transition:0.3s;
}

.stButton button:hover{
    transform:translateY(-3px);
    box-shadow:0 0 25px #8b5cf6;
}

/* ===============================
   SELECTBOX
================================ */
.stSelectbox > div > div{
    background:rgba(255,255,255,0.08);
    border-radius:12px;
    border:1px solid rgba(255,255,255,0.15);
}

/* ===============================
   FILE UPLOADER
================================ */
[data-testid="stFileUploader"]{
    border:2px dashed #8b5cf6;
    border-radius:16px;
    padding:15px;
    background:rgba(255,255,255,0.05);
}

/* ===============================
   DATAFRAME
================================ */
thead tr th{
    background:#1e293b !important;
    color:#60a5fa !important;
}

tbody{
    color:white !important;
}

/* ===============================
   SIDEBAR
================================ */
[data-testid="stSidebar"]{
    background:
    linear-gradient(
    180deg,
    #111827,
    #1e1b4b
    );
}

[data-testid="stSidebar"] *{
    color:white !important;
}

/* ===============================
   METRICS
================================ */
[data-testid="metric-container"]{
    background:rgba(255,255,255,0.05);
    border:1px solid rgba(255,255,255,0.1);
    border-radius:18px;
    padding:15px;
    box-shadow:0px 0px 20px rgba(139,92,246,.15);
}

/* ===============================
   SCROLLBAR
================================ */
::-webkit-scrollbar{
    width:10px;
}

::-webkit-scrollbar-thumb{
    background:#8b5cf6;
    border-radius:10px;
}

/* ===============================
   PLOTS
================================ */
canvas{
    border-radius:15px;
}

</style>
""", unsafe_allow_html=True)
# -- Executive summary -----------------------------------------------------------
st.title("Customer Purchase Prediction")
st.write("This Streamlit app predicts whether customers will make a purchase based on their demographic and behavioral features. It includes data preprocessing, exploratory visualizations, model training, evaluation metrics, and interactive prediction.")

# -- Data Loading and Preprocessing ----------------------------------------------
DATA_PATH = "customer_purchase_prediction_dataset_1.csv"
try:
    data = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    st.error(f"Dataset file '{DATA_PATH}' not found. Please ensure the CSV is in the project directory.")
    st.stop()

st.write("### Raw Data (first few rows)")
st.write(data.head())

# Check for missing values
st.write("Checking for missing values...")
if data.isnull().values.any():
    st.warning("Dataset contains missing values. Please handle them before training.")
else:
    st.write("No missing values found.")

# Encode categorical Gender using One-Hot Encoding
# Encode categorical Gender using One-Hot Encoding
if 'Gender' in data.columns:
    encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown='ignore'
    )

    gender_encoded = encoder.fit_transform(data[['Gender']])

    gender_cols = encoder.get_feature_names_out(['Gender'])

    gender_df = pd.DataFrame(
        gender_encoded,
        columns=gender_cols,
        index=data.index
    )

    data = pd.concat(
        [data.drop('Gender', axis=1), gender_df],
        axis=1
    )

    st.write("Gender column one-hot encoded into:", gender_cols.tolist())

# Scale numeric features
numeric_cols = ['Age', 'Annual_Income', 'Spending_Score', 'Previous_Purchases', 'Engagement_Score']
if all(col in data.columns for col in numeric_cols):
    scaler = StandardScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    st.write("Numeric features scaled using StandardScaler.")

# Define features (X) and target (y)
X = data.drop(['Customer_ID', 'Purchase'], axis=1, errors='ignore')
y = data['Purchase']

# -- Train-Test Split ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
st.write(f"Split data: {len(X_train)} training samples, {len(X_test)} test samples.")

# -- Model Training or Loading ---------------------------------------------------
model_files = {
    "Logistic Regression": "logistic_model.joblib",
    "Decision Tree": "tree_model.joblib"
}

# Train models if not already saved
if os.path.exists(model_files["Logistic Regression"]) and os.path.exists(model_files["Decision Tree"]):
    log_model = joblib.load(model_files["Logistic Regression"])
    tree_model = joblib.load(model_files["Decision Tree"])
    st.write("Loaded pre-trained models from disk.")
else:
    # Logistic Regression model
    log_model = LogisticRegression(random_state=42, max_iter=1000)
    log_model.fit(X_train, y_train)
    joblib.dump(log_model, model_files["Logistic Regression"])
    st.write("Trained and saved Logistic Regression model.")

    # Decision Tree model
    tree_model = DecisionTreeClassifier(random_state=42, max_depth=5)
    tree_model.fit(X_train, y_train)
    joblib.dump(tree_model, model_files["Decision Tree"])
    st.write("Trained and saved Decision Tree model.")

# -- Model Selection & Metrics ---------------------------------------------------
model_option = st.selectbox("Select Model:", ["Logistic Regression", "Decision Tree"])
model = log_model if model_option == "Logistic Regression" else tree_model

# Predictions on test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

st.write(f"### {model_option} Evaluation Metrics:")
st.write(f"- Accuracy: {accuracy:.3f}")
st.write(f"- Precision: {precision:.3f}")
st.write(f"- Recall: {recall:.3f}")
st.write(f"- F1 Score: {f1:.3f}")

# Confusion matrix plot
st.write("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Purchase", "Purchase"])
fig, ax = plt.subplots(figsize=(4,4))
disp.plot(ax=ax, cmap='Blues', values_format='d')
st.pyplot(fig)

# Classification report
st.write("Classification Report:")
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df.round(3))

# -- Exploratory Data Analysis -----------------------------------------------
st.write("## Exploratory Data Analysis")

# Histograms for each feature
st.write("### Feature Distributions")
for col in numeric_cols:
    fig = plt.figure(figsize=(4,3))
    plt.hist(data[col], bins=10, color='lightblue', edgecolor='black')
    plt.title(col)
    st.pyplot(fig)

# Correlation heatmap
st.write("### Correlation Heatmap")
corr = data.corr()
fig_heat = plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
st.pyplot(fig_heat)

# Scatter plot: Income vs Spending Score
if 'Annual_Income' in data.columns and 'Spending_Score' in data.columns:
    st.write("### Income vs Spending Score")
    fig_scatter = plt.figure(figsize=(4,3))
    sns.scatterplot(
        x='Annual_Income', y='Spending_Score',
        hue='Purchase', palette='coolwarm', data=data
    )
    st.pyplot(fig_scatter)

# -- Predictions on New Data ---------------------------------------------------
st.write("## Predict on New Data")
uploaded_file = st.file_uploader("Upload CSV with new customer data", type=["csv"])
if uploaded_file is not None:
    try:
        new_data = pd.read_csv(uploaded_file)
        st.write("New data (first rows):")
        st.write(new_data.head())

        # Preprocess new data: Gender, scaling
        if 'Gender' in new_data.columns:
            gender_new = encoder.transform(new_data[['Gender']])
            gender_new_df = pd.DataFrame(gender_new, columns=gender_cols)
            new_data = pd.concat([new_data.drop('Gender', axis=1).reset_index(drop=True), gender_new_df], axis=1)
        new_data[numeric_cols] = scaler.transform(new_data[numeric_cols])

        # Predict
        features_new = new_data.drop(['Customer_ID'], axis=1, errors='ignore')
        preds = model.predict(features_new)
        new_data['Predicted_Purchase'] = preds
        st.write("Predictions on uploaded data:")
        st.write(new_data)
    except Exception as e:
        st.error(f"Error processing file: {e}")
