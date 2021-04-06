import streamlit as st 

import spacy
import spacy_streamlit
nlp = spacy.blank('en')

import os
from PIL import Image

st.title("Spacy-Streamlit NLP App")

menu = ['Home', 'NER']
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
    st.subheader("Tokenization")
    raw_text = st.text_area("Your Text", "Enter Text Here")
    docx = nlp(raw_text)
    if st.button("Tokenize"):
        spacy_streamlit.visualize_tokens(docx, attrs=['text'])

elif choice == "NER":
    st.subheader("Named Entity Recognition")
    raw_text = st.text_area("Your Text", "Enter Text Here")
    docx = nlp(raw_text)
    
    spacy_streamlit.visualize_ner(docx, labels=nlp.get_pipe('ner').labels)
                    
