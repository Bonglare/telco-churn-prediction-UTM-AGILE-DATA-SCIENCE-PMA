import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.title("Telco Customer Churn Dashboard")

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Data preprocessing
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

st.header("Interactive Filters")
col1, col2 = st.columns(2)

with col1:
    contract_filter = st.selectbox("Select Contract Type", df["Contract"].unique())

with col2:
    tenure_slider = st.slider(
        "Select Tenure Range",
        int(df["tenure"].min()),
        int(df["tenure"].max()),
        (int(df["tenure"].min()), int(df["tenure"].max()))
    )

# Filter data
filtered_df = df[
    (df["Contract"] == contract_filter) &
    (df["tenure"] >= tenure_slider[0]) &
    (df["tenure"] <= tenure_slider[1])
]

st.header("Data Visualizations")

# Visualization 1: Churn Distribution by Contract Type
st.subheader("1. Churn Distribution by Contract Type")
fig1, ax1 = plt.subplots()
sns.countplot(x="Contract", hue="Churn", data=filtered_df, ax=ax1)
ax1.set_title("Churn by Contract Type")
ax1.set_xlabel("Contract Type")
ax1.set_ylabel("Count")
st.pyplot(fig1)

# Visualization 2: Tenure Distribution
st.subheader("2. Tenure Distribution")
fig2, ax2 = plt.subplots()
sns.histplot(filtered_df["tenure"], bins=30, kde=True, ax=ax2)
ax2.set_title("Distribution of Customer Tenure")
ax2.set_xlabel("Tenure (months)")
ax2.set_ylabel("Count")
st.pyplot(fig2)

# Visualization 3: Monthly Charges vs Total Charges
st.subheader("3. Monthly Charges vs Total Charges")
fig3, ax3 = plt.subplots()
sns.scatterplot(x="MonthlyCharges", y="TotalCharges", hue="Churn", data=filtered_df, ax=ax3)
ax3.set_title("Monthly Charges vs Total Charges")
ax3.set_xlabel("Monthly Charges")
ax3.set_ylabel("Total Charges")
st.pyplot(fig3)

st.header("Predictive Output")
st.write("Enter customer details to predict churn probability.")

# Input fields for prediction
col_input1, col_input2 = st.columns(2)

with col_input1:
    input_tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
    input_contract = st.selectbox("Contract Type", df["Contract"].unique())

with col_input2:
    input_monthly = st.number_input("Monthly Charges", min_value=0.0, max_value=120.0, value=50.0)
    input_payment = st.selectbox("Payment Method", df["PaymentMethod"].unique())

# Calculate total charges
input_total = input_tenure * input_monthly

if st.button("Predict Churn"):
    # Prepare training data
    temp_df = df.copy()
    
    # Encode categorical variables
    cat_cols = temp_df.select_dtypes(include=["object"]).columns
    temp_encoded = pd.get_dummies(temp_df, columns=cat_cols, drop_first=True)
    
    # Convert boolean to int
    bool_cols = temp_encoded.select_dtypes(include=["bool"]).columns
    temp_encoded[bool_cols] = temp_encoded[bool_cols].astype(int)
    
    # Rename target variable if needed
    if "Churn_Yes" in temp_encoded.columns:
        temp_encoded.rename(columns={"Churn_Yes": "Churn"}, inplace=True)
    
    # Select features for model
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    
    # Scale numerical features
    scaler = StandardScaler()
    temp_encoded[num_cols] = scaler.fit_transform(temp_encoded[num_cols])
    
    # Prepare features and target
    X = temp_encoded.drop("Churn", axis=1)
    y = temp_encoded["Churn"]
    
    # Train model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    
    # Prepare input data for prediction
    input_data = pd.DataFrame({
        "tenure": [input_tenure],
        "MonthlyCharges": [input_monthly],
        "TotalCharges": [input_total],
        "Contract": [input_contract],
        "PaymentMethod": [input_payment]
    })
    
    # Encode input data
    input_encoded = pd.get_dummies(input_data)
    
    # Align columns with training data
    for col in X.columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    # Reorder columns to match training data
    input_encoded = input_encoded[X.columns]
    
    # Scale numerical features
    input_encoded[num_cols] = scaler.transform(input_encoded[num_cols])
    
    # Make prediction
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]
    
    # Display result
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"This customer is likely to churn.")
        st.metric("Churn Probability", f"{probability:.2%}")
    else:
        st.success(f"This customer is likely to stay.")
        st.metric("Churn Probability", f"{probability:.2%}")
    
    # Display feature importance (coefficients)
    st.subheader("Model Coefficients (Top 10)")
    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_[0]
    }).sort_values("Coefficient", key=abs, ascending=False).head(10)
    
    st.dataframe(coef_df)