from joblib import load
import streamlit as st

# Load the model
model = load('emotion_classifier_model.pkl')

st.title("Emotion Analysis App")

user_input = st.text_area("Enter the text to analyze emotions:")

# Predict button
if st.button("Analyze Emotion"):
    if user_input:
        # Use the model to make predictions
        prediction = model.predict([user_input]) 
        st.write("Predicted Emotion:", prediction[0])
    else:
        st.warning("Please enter some text!")

