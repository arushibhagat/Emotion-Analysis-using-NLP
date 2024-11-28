import streamlit as st
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences 
import numpy as np
from joblib import load

# Load the model and tokenizer
with open('emotion_classifier_model.pkl', 'rb') as file:
    model = load(file)
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Emotion Labels
emotion_labels = ['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise']

st.title("Emotion Analysis App")

user_input = st.text_area("Enter the text to analyze the emotion:")

if st.button("Analyze Emotion"):
    if user_input:
        try:
            # Tokenize the input text before passing it to the model
            tokenized_input = tokenizer.texts_to_sequences([user_input])
            # Pad the tokenized input to the expected input shape for the model
            padded_input = pad_sequences(tokenized_input, padding='post')

            # Predict the emotion from the model
            prediction = model.predict(padded_input)  # Ensure model is expecting tokenized and padded input

            # Assuming the model returns probabilities, take the index of the max probability
            predicted_index = np.argmax(prediction)  # Get the index of the highest probability
            predicted_emotion = emotion_labels[predicted_index]  # Map the index to the emotion label

            st.write(f"Predicted Emotion: {predicted_emotion}")
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
    else:
        st.warning("Please enter some text!")
