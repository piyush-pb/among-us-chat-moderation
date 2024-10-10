import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import word_tokenize
import pickle

# Load the classifier and TF-IDF vectorizer from files
with open('naive_bayes_classifier.pkl', 'rb') as file:
    model = pickle.load(file)

with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

# Set page title and layout
st.set_page_config(page_title="Hate Speech Detection", page_icon=":angry:", layout="wide")

# Set title and subtitle
st.title('Hate Speech Detection using TF-IDF')
st.write("This application uses a TF-IDF vectorizer and a trained Multinomial Naive Bayes classifier to detect hate speech.")

# Text input for user to input their text
user_text = st.text_input('Enter your text:')

# Button to trigger text classification
if st.button('Classify'):
    # Tokenize the user input
    tokenized_text = word_tokenize(user_text.lower())

    # Convert tokenized text into TF-IDF vectors
    text_vector = tfidf_vectorizer.transform([' '.join(tokenized_text)])

    # Make prediction
    prediction = model.predict(text_vector)

    if prediction == 1:
        st.write("Hate Speech")
    elif prediction == 0:
        st.write("Offence Not Allowed Baby!")
    else:
        st.write("Everything is Normal Bitch")
