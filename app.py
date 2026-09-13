import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Updated LangChain Partner Package Imports
from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load API keys from .env
load_dotenv()

st.set_page_config(page_title="Country Segmentation & Economic AI Assistant", layout="wide")

# --------------------------------------------------------------------------------------------
# 1. CACHED ASSETS (ML MODELS & RAG CHAIN)
# --------------------------------------------------------------------------------------------
@st.cache_resource
def load_ml_assets():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    notebooks_path = os.path.join(base_dir, "notebooks")

    scaler = joblib.load(os.path.join(notebooks_path, "scaler.pkl"))
    pca = joblib.load(os.path.join(notebooks_path, "pca.pkl"))
    kmeans = joblib.load(os.path.join(notebooks_path, "kmeans.pkl"))
    features = joblib.load(os.path.join(notebooks_path, "features.pkl"))
    return scaler, pca, kmeans, features

@st.cache_resource
def load_rag_chain():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check for either vector directory naming standard
    vectorstore_path = os.path.join(base_dir, "vectorstore_db")
    if not os.path.exists(vectorstore_path):
        vectorstore_path = os.path.join(base_dir, "chroma_db")
    
    embedding_model = FastEmbedEmbeddings()
    vectorstore = Chroma(
        persist_directory=vectorstore_path,
        embedding_function=embedding_model
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    def format_docs(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    prompt_template = """
You are an expert Economic Data Analyst and Political Economist.
Use the retrieved context documents from the World Bank RAG knowledge base alongside any active user country metrics to provide complete, insightful answers.

Guidelines:
- Base your analysis directly on the provided cluster profiles, feature metadata, and methodology documents.
- Clearly mention relevant cluster labels, PCA coordinates, or macro metrics when applicable.
- If you don't know the answer or if it's not present in the context, state that clearly.

Retrieved Context Documents:
{context}

User Question:
{question}

Answer:
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.2
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# Load cached pipeline assets
scaler, pca, kmeans, features = load_ml_assets()
rag_chain = load_rag_chain()

# --------------------------------------------------------------------------------------------
# 2. STREAMLIT UI: PREDICTION FORM
# --------------------------------------------------------------------------------------------
st.title("🌍 Country Segmentation & Economic AI Assistant")
st.write("Enter country macroeconomic indicators below to predict its cluster, then chat with Gemini using your RAG knowledge base.")

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
        school_enroll = st.number_input("School Enrollment, Primary (% Gross)", format="%.6f")
        services_val = st.number_input("Services Value Added (% of GDP)", format="%.6f")
        unemployment = st.number_input("Unemployment Total (%)", format="%.6f")
        socio_idx = st.number_input("Socioeconomic Development Index", format="%.6f")
        energy_idx = st.number_input("Energy & Environmental Index", format="%.6f")
        
        fdi_raw = st.number_input("Foreign Direct Investment", format="%.6f")
        inflation_raw = st.number_input("Inflation", format="%.6f")
        gdp_capita_raw = st.number_input("GDP per Capita", format="%.6f")

    submit_button = st.form_submit_button(label="Predict Cluster")

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if submit_button:
    fdi_log = np.sign(fdi_raw) * np.log1p(np.abs(fdi_raw))
    inflation_log = np.sign(inflation_raw) * np.log1p(np.abs(inflation_raw))
    gdp_capita_log = np.sign(gdp_capita_raw) * np.log1p(np.abs(gdp_capita_raw))

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
    
    X_new = pd.DataFrame(input_dict)[features]
    X_new_scaled = scaler.transform(X_new)
    X_new_pca = pca.transform(X_new_scaled)
    cluster = kmeans.predict(X_new_pca)[0] + 1
    
    st.session_state.last_prediction = {
        "cluster": cluster,
        "gdp_per_capita": gdp_capita_raw,
        "inflation": inflation_raw,
        "unemployment": unemployment,
        "socio_idx": socio_idx,
        "energy_idx": energy_idx,
        "pca_1": X_new_pca[0][0],
        "pca_2": X_new_pca[0][1]
    }
    
    st.success(f"**Prediction Complete:** Country maps to **Cluster {cluster}** (PCA: PC1={X_new_pca[0][0]:.2f}, PC2={X_new_pca[0][1]:.2f})")

# --------------------------------------------------------------------------------------------
# 3. STREAMLIT UI: RAG CHAT INTERFACE
# --------------------------------------------------------------------------------------------
st.divider()
st.subheader("💬 Ask Gemini About Country Profiles, Clusters & Methodology")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing chat conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Interactive chat input box
if user_prompt := st.chat_input("Ask a question about your inputs, cluster profiles, or methodology..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Prepend user's submitted input context to query if form was filled
    augmented_query = user_prompt
    if st.session_state.last_prediction:
        pred = st.session_state.last_prediction
        augmented_query = (
            f"[Active Form Input Context: Predicted Cluster {pred['cluster']}, "
            f"GDP per Capita: ${pred['gdp_per_capita']:,.2f}, Inflation: {pred['inflation']}%, "
            f"Unemployment: {pred['unemployment']}%, Socioeconomic Index: {pred['socio_idx']}, "
            f"Energy Index: {pred['energy_idx']}, PC1: {pred['pca_1']:.2f}, PC2: {pred['pca_2']:.2f}]\n\n"
            f"User Question: {user_prompt}"
        )

    with st.chat_message("assistant"):
        with st.spinner("Analyzing documents & generating response..."):
            response_text = rag_chain.invoke(augmented_query)
            st.markdown(response_text)

    st.session_state.messages.append({"role": "assistant", "content": response_text})