import pandas as pd
import spacy
import streamlit as st
from spacy import displacy
import spacy_streamlit

SPACY_MODEL_NAMES = ["en_core_web_sm", "en_core_web_md", "de_core_news_sm"]
DEFAULT_TEXT = "Mark Zuckerberg is the CEO of Facebook."
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""


@st.cache(allow_output_mutation=True)
def load_model(name):
    return spacy.load(name)


@st.cache(allow_output_mutation=True)
def process_text(model_name, text):
    nlp = load_model(model_name)
    
spacy_streamlit.visualize(SPACY_MODEL_NAMES, DEFAULT_TEXT)
