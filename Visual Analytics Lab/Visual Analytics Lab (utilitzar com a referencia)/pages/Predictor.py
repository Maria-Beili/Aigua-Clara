import streamlit as st

import pandas as pd
import numpy as np

import pickle

st.set_page_config(
    page_title="Loan Approval Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model
with open('loan_app_artifacts.pkl', 'rb') as file:
    data = pickle.load(file)

model_loaded = data["model"]
encoder_loaded = data["encoder"]
scaler_loaded = data["scaler"]
numeric_cols_loaded = data["numeric_cols"]
categorical_cols_loaded = data["categorical_cols"]
model_columns_ordered_loaded = data["model_columns_ordered"]

st.title("Loan Approval Predictor")

# Initialize session_state to preserve input values
for key, default_value in {
    "LoanAmount": 150.0,
    "Loan_Amount_Term": 360.0,
    "TotalIncome": 8000.0,
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "0",
    "Education": "Graduate",
    "Self_Employed": "No",
    "Credit_History": 1.0,
    "Property_Area": "Semiurban",
    "Cutoff": 0.5
}.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

st.header("Enter Applicant Details")

st.session_state["LoanAmount"] = st.number_input("Loan Amount", value=st.session_state["LoanAmount"])
st.session_state["Loan_Amount_Term"] = st.number_input("Loan Term (months)", value=st.session_state["Loan_Amount_Term"])
st.session_state["TotalIncome"] = st.number_input("Total Income", value=st.session_state["TotalIncome"])

st.session_state["Gender"] = st.selectbox("Gender", ["Male", "Female"], index=["Male","Female"].index(st.session_state["Gender"]))
st.session_state["Married"] = st.selectbox("Married", ["Yes", "No"], index=["Yes","No"].index(st.session_state["Married"]))
st.session_state["Dependents"] = st.selectbox("Dependents", ["0", "1", "2", "3+"], index=["0","1","2","3+"].index(st.session_state["Dependents"]))
st.session_state["Education"] = st.selectbox("Education", ["Graduate", "Not Graduate"], index=["Graduate","Not Graduate"].index(st.session_state["Education"]))
st.session_state["Self_Employed"] = st.selectbox("Self Employed", ["Yes", "No"], index=["Yes","No"].index(st.session_state["Self_Employed"]))
st.session_state["Credit_History"] = st.selectbox("Credit History", [1.0, 0.0], index=[1.0,0.0].index(st.session_state["Credit_History"]))
st.session_state["Property_Area"] = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"], index=["Urban","Semiurban","Rural"].index(st.session_state["Property_Area"]))

st.header("Set Cutoff Threshold")
st.session_state["Cutoff"] = st.slider("Cutoff (0 = aggressive, 1 = conservative)", min_value=0.0, max_value=1.0, value=st.session_state["Cutoff"], step=0.01)

input_df = pd.DataFrame([[
    st.session_state["LoanAmount"],
    st.session_state["Loan_Amount_Term"],
    st.session_state["TotalIncome"],
    st.session_state["Gender"],
    st.session_state["Married"],
    st.session_state["Dependents"],
    st.session_state["Education"],
    st.session_state["Self_Employed"],
    st.session_state["Credit_History"],
    st.session_state["Property_Area"]
]], columns=[
    'LoanAmount', 'Loan_Amount_Term', 'TotalIncome', 
    'Gender', 'Married', 'Dependents', 
    'Education', 'Self_Employed', 'Credit_History', 'Property_Area'
])

numeric_data = scaler_loaded.transform(input_df[numeric_cols_loaded])
df_numeric_scaled = pd.DataFrame(numeric_data, columns=numeric_cols_loaded, index=input_df.index)

categorical_data = encoder_loaded.transform(input_df[categorical_cols_loaded])
encoded_col_names = encoder_loaded.get_feature_names_out(categorical_cols_loaded)
df_categorical_encoded = pd.DataFrame(categorical_data, columns=encoded_col_names, index=input_df.index)

df_processed = pd.concat([df_numeric_scaled, df_categorical_encoded], axis=1)
df_processed = df_processed[model_columns_ordered_loaded]

prediction_prob = model_loaded.predict_proba(df_processed)[:, 1][0]
prediction = int(prediction_prob >= st.session_state["Cutoff"])

if st.button("Predict Loan Approval"):
    if prediction == 1:
        st.success(f"Loan Approved (Probability: {prediction_prob:.2f}, Cutoff: {st.session_state['Cutoff']})")
    else:
        st.error(f"Loan Denied (Probability: {prediction_prob:.2f}, Cutoff: {st.session_state['Cutoff']})")
