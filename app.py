import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load pre-trained models and features
scaler = joblib.load(r"D:\Projects\2B_Data_Science_Portfolio\Task_4_(Train_3_Unsupervised_ML_Models)\notebooks\scaler.pkl")
pca = joblib.load(r"D:\Projects\2B_Data_Science_Portfolio\Task_4_(Train_3_Unsupervised_ML_Models)\notebooks\pca.pkl")
kmeans = joblib.load(r"D:\Projects\2B_Data_Science_Portfolio\Task_4_(Train_3_Unsupervised_ML_Models)\notebooks\kmeans.pkl")
features = joblib.load(r"D:\Projects\2B_Data_Science_Portfolio\Task_4_(Train_3_Unsupervised_ML_Models)\notebooks\features.pkl")

st.title("**Country Segmentation app**")
st.write("Enter your Country inputs to cluster. ;)")

# Group inputs into a form to prevent app refresh on every keystroke
with st.form("country_data_form"):
    st.subheader("Macroeconomic & Environmental Indicators")
    
    col1, col2 = st.columns(2)
    
    with col1:
        agri_val = st.number_input("Agriculture, Forestry, and Fishing Value Added (% of GDP)", format="%.6f")
        carbon_int = st.number_input("Carbon Intensity of GDP (kg CO2e)", format="%.6f")
        health_exp = st.number_input("Current Health Expenditure (% of GDP)", format="%.6f")
        fertility = st.number_input("Fertility Rate (Total births per woman)", format="%.6f")
        forest_area = st.number_input("Forest Area (% of land area)", format="%.6f")
        gdp_growth = st.number_input("GDP Growth (Annual %)", format="%.6f")
        industry_val = st.number_input("Industry Value Added (% of GDP)", format="%.6f")
        pop_growth = st.number_input("Population Growth (Annual %)", format="%.6f")
        renew_elec = st.number_input("Renewable Electricity Output (%)", format="%.6f")
        
    with col2:
        renew_energy = st.number_input("Renewable Energy Consumption (%)", format="%.6f")
        school_enroll = st.number_input("School Enrollment, Primary (% Gross)", format="%.6f") # Added new feature here
        services_val = st.number_input("Services Value Added (% of GDP)", format="%.6f")
        unemployment = st.number_input("Unemployment Total (%)", format="%.6f")
        socio_idx = st.number_input("Socioeconomic Development Index", format="%.6f")
        energy_idx = st.number_input("Energy & Environmental Index", format="%.6f")
        
        fdi_raw = st.number_input("Foreign Direct Investment", format="%.6f")
        inflation_raw = st.number_input("Inflation", format="%.6f")
        gdp_capita_raw = st.number_input("GDP per Capita", format="%.6f")

    submit_button = st.form_submit_button(label="Predict Cluster")

if submit_button:
    # Apply the simplified signed log-modulus transformations to the raw inputs
    fdi_log = np.sign(fdi_raw) * np.log1p(np.abs(fdi_raw))
    inflation_log = np.sign(inflation_raw) * np.log1p(np.abs(inflation_raw))
    gdp_capita_log = np.sign(gdp_capita_raw) * np.log1p(np.abs(gdp_capita_raw))

    # Map user inputs to the exact 18 feature names expected by the model
    input_dict = {
        "agriculture_forestry_and_fishing_value_added_of_gdp": [agri_val],
        "carbon_intensity_of_gdp_kg_co2e_per_constant_2015_us_of_gdp": [carbon_int],
        "current_health_expenditure_of_gdp": [health_exp],
        "fertility_rate_total_births_per_woman": [fertility],
        "forest_area_of_land_area": [forest_area],
        "gdp_growth_annual_": [gdp_growth],
        "industry_including_construction_value_added_of_gdp": [industry_val],
        "population_growth_annual_": [pop_growth],
        "renewable_electricity_output_of_total_electricity_output": [renew_elec],
        "renewable_energy_consumption_of_total_final_energy_consumption": [renew_energy],
        "school_enrollment_primary_gross": [school_enroll],
        "services_value_added_of_gdp": [services_val],
        "unemployment_total_of_total_labor_force_modeled_ilo_estimate": [unemployment],
        "Socioeconomic_developmen_Index": [socio_idx],
        "Energy_Environmental_Index": [energy_idx],
        "fdi_log": [fdi_log],
        "inflation_log": [inflation_log],
        "gdp_per_capita_constant_2015_us_log": [gdp_capita_log]
    }
    
    X_new = pd.DataFrame(input_dict)
    
    # Ensure column order matches the original training features exactly
    X_new = X_new[features]
    
    # Process pipeline
    X_new_scaled = scaler.transform(X_new)
    X_new_pca = pca.transform(X_new_scaled)
    cluster = kmeans.predict(X_new_pca)
    
    st.success(f"**Prediction Results:** This country maps to **Cluster {cluster[0]}**")

#--------------------------------------------------------------------------------------------
#                               Connect the ML to an LLM
#--------------------------------------------------------------------------------------------
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

# Store client and chat in session_state to maintain conversation history across rerun triggers
if "client" not in st.session_state:
    st.session_state.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

if "chat" not in st.session_state:
    st.session_state.chat = st.session_state.client.chats.create(
        model="gemini-3.6-flash"  # Updated to supported model
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

# Send the input metrics & predicted cluster to Gemini when form is submitted
if submit_button:
    context_payload = (
        f"[SYSTEM CONTEXT UPDATE]\n"
        f"The user ran the country segmentation model with the following inputs:\n"
        f"- Assigned Cluster: {cluster[0]}\n"
        f"- GDP per Capita: ${gdp_capita_raw:,.2f}\n"
        f"- Inflation Rate: {inflation_raw}%\n"
        f"- Unemployment: {unemployment}%\n"
        f"- Socioeconomic Development Index: {socio_idx}\n"
        f"- Energy & Environmental Index: {energy_idx}\n"
        f"- Renewable Energy Consumption: {renew_energy}%\n"
        f"- Agriculture Value Added: {agri_val}%\n"
        f"- Health Expenditure: {health_exp}%\n"
        f"- Fertility Rate: {fertility}\n"
        f"- School Enrollment: {school_enroll}%\n"
        f"- Foreign Direct Investment: ${fdi_raw:,.2f}\n"
        f"Use these details to answer any questions the user asks."
    )
    st.session_state.chat.send_message(context_payload)

# UI Elements for Chat
st.divider()
st.subheader("💬 Ask Gemini About This Cluster")

# Display prior chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Capture user input and generate responses
if user_prompt := st.chat_input("Ask a question about the cluster analysis..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat.send_message(user_prompt)
            st.markdown(response.text)

    st.session_state.messages.append(
        {"role": "assistant", "content": response.text}
    )