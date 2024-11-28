import streamlit as st
from joblib import load

# Load the model
with open('emotion_classifier_model.pkl', 'rb') as file:
    model = load(file)

st.title("Emotion Analysis App")

user_input = st.text_area("Enter the text to analyze emotions:")

# Predict button
if st.button("Analyze Emotion"):
    if user_input:
        # Ensure input is in the correct format for the model (likely a string)
        prediction = model.predict([user_input])  # Predict requires the text in a list, but as a string
        st.write("Predicted Emotion:", prediction[0])
    else:
        st.warning("Please enter some text!")
