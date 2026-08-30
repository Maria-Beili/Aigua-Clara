import streamlit as st

st.set_page_config(
    page_title="Loan Eligibility Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Loan Eligibility Visual Analytics App")

st.sidebar.header("Select a page from the sidebar to begin")

st.write("Welcome to the Loan Eligibility Analyzer!\n- Explore historical loan data in the Analysis page\n- Predict loan approval on the Predictor page\n")