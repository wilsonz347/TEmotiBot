import streamlit as st
import pickle
import pandas as pd
import spacy

with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

nlp = spacy.load("en_core_web_sm")

df = pd.read_csv('grouped_intents.csv')


def spacy_tokenize(text):
    if not text or len(str(text).strip()) < 1:
        return ""

    doc = nlp(str(text).lower())

    new_text = [
        token.lemma_ for token in doc
        if (len(token.text) > 0 and (token.is_alpha or token.text.isalnum())) and
           not token.is_punct
    ]

    processed_text = " ".join(new_text) if new_text else text
    return processed_text


def get_response(tag):
    # Get a random response for the predicted tag
    responses = df[df['grouped_tag'] == tag]['response'].tolist()
    return responses[0] if responses else "I'm not sure how to respond to that."


st.title('Intent Classification Chatbot')

user_input = st.text_input("You:", "")

if user_input:
    processed_input = spacy_tokenize(user_input)

    prediction = model.predict([processed_input])[0]
    predicted_tag = label_encoder.inverse_transform([prediction])[0]

    response = get_response(predicted_tag)

    st.text_area("Bot:", value=response, height=100, max_chars=None, key=None)

st.write("---")
st.write("This chatbot uses a machine learning model to classify intents and provide appropriate responses.")

