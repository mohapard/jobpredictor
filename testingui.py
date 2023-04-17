import streamlit as st
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
from scipy.sparse import hstack

def authenticate(password):
    return password == "wipo"

st.set_page_config(page_title="Job Category Predictor", page_icon=None, layout="centered", initial_sidebar_state="auto")

st.title("Welcome to Job Category Predictor")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    password = st.text_input("Enter password:", type="password")

    if st.button("Authenticate"):
        if authenticate(password):
            st.session_state.authenticated = True
            st.success("Authentication successful! You can now access the app.")
         
        else:
            st.error("Authentication failed. Please try again.")

if st.session_state.authenticated:
    # Loading the model for prediction
    #loaded_model, loaded_vectorizer = joblib.load('job_category_model.joblib')
    
    # Loading the model for prediction
    loaded_model, loaded_vectorizer_title, loaded_vectorizer_lang = joblib.load('job_category_model2.joblib')
    st.header("Job Category Predictor")

    job_title = st.text_input("Job title:")
    job_language = st.text_input("Language:")

    if st.button("Predict Job Category"):

        #job_title_vector = loaded_vectorizer.transform([job_title])
        
        predicted_prob = loaded_model.predict_proba(hstack([loaded_vectorizer_title.transform([job_title]), loaded_vectorizer_lang.transform([job_language])]))[0]
        #predicted_prob = loaded_model.predict_proba(job_title_vector)[0]
        
        predicted_category = loaded_model.predict(hstack([loaded_vectorizer_title.transform([job_title]), loaded_vectorizer_lang.transform([job_language])]))[0]
        confidence = predicted_prob.max()
        response = f"The predicted job category for '{job_title}' is '{predicted_category}' with {confidence:.2%} confidence."
        if confidence< 0.78:
            response +="It seems like this job title is not an educational professional."
        st.write(response)
