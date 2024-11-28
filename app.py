import streamlit as st
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences 

from joblib import load

# Load the model and tokenizer
with open('emotion_classifier_model.pkl', 'rb') as file:
    model = load(file)
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

st.title("Emotion Analysis App")

user_input = st.text_area("Enter the text to analyze emotions:")

if st.button("Analyze Emotion"):
    if user_input:
        try:
            # Tokenize the input text before passing it to the model
            tokenized_input = tokenizer.texts_to_sequences([user_input])
            padded_input = tokenizer.pad_sequences(tokenized_input, padding='post')

            # Predict the emotion from the model
            prediction = model.predict(padded_input)  # Ensure model is expecting tokenized and padded input
            st.write("Predicted Emotion:", prediction[0])
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
    else:
        st.warning("Please enter some text!")
