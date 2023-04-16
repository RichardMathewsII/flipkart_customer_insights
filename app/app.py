import streamlit as st
import pandas as pd
from cleaning import clean_dataset
from main import generate_customer_insights_report
import toml

# load toml config
config = toml.load("config.toml")

icon = open(config["paths"]["icon"], "rb").read()
st.set_page_config(
    page_title="CustomerInsights",
    page_icon=icon
)
# render Flipkart image
image_file = config["paths"]["image"]
image = open(image_file, "rb").read()
st.image(image)

# portal title
st.header("Flipkart Customer Insights Portal")
st.subheader("An AI-powered system that provides insights to sellers on what customers like and don't like about the product and ways to improve it.")

# streamlit formatting
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

# Read the CSV file into a pandas dataframe
df = pd.read_csv(config["paths"]["data"])

# clean the dataset
df_cleaned = clean_dataset(df)

# Create a selectbox to choose unique values from the ProductName field
with st.form("my_form"):
    selected_product = st.selectbox('Select a product', df_cleaned['ProductName'].unique())
    submitted = st.form_submit_button("Submit")

# when submitted, render markdown report for chosen product
if submitted:
    status = st.empty()
    with st.spinner("AI is working on your request..."):
        report = generate_customer_insights_report(df_cleaned, selected_product, status)
    status.write("")
    st.balloons()
    st.markdown(report, unsafe_allow_html=True)