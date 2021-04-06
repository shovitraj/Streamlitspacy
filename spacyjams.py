

import pandas as pd
import spacy
import streamlit as st
from spacy import displacy
import spacy_streamlit


from sentence_transformers import SentenceTransformer, util
import numpy as np

import plotly
import plotly.express as px 

import plotly.graph_objects as go
model = SentenceTransformer('stsb-roberta-large')

SPACY_MODEL_NAMES = ["en_core_web_sm", "en_core_web_md", "de_core_news_sm"]
DEFAULT_TEXT = "Applicants are highly engouraged to use JAMS assist."
COMPARE_TEXT = "Recruiters, we will find the best candidate for you."
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""


@st.cache(allow_output_mutation=True)
def load_model(name):
    return spacy.load(name)


@st.cache(allow_output_mutation=True)
def process_text(model_name, text):
    nlp = load_model(model_name)
    return nlp(text)


st.sidebar.title("Interactive spaCy visualizer")
st.sidebar.markdown(
    """
JAMS Spacy
"""
)

spacy_model = st.sidebar.selectbox("Model name", SPACY_MODEL_NAMES)
model_load_state = st.info(f"Loading model '{spacy_model}'...")
nlp = load_model(spacy_model)
model_load_state.empty()

text1 = st.text_area("Text to analyze", DEFAULT_TEXT)
text1 = ''.join(c for c in text1 if not c.isnumeric())
text1 = ''.join([x.replace('●','') for x in text1])
text1 = ''.join([x.replace('•','') for x in text1])
text1 = ''.join([x.replace('_','') for x in text1])
doc1 = process_text(spacy_model, text1)

sentences1 = [sent.text for sent in doc1.sents]
sentences1 = [x.replace('\n','') for x in sentences1]

embedding1 = model.encode(sentences1, convert_to_tensor=True)

text2 = st.text_area("Text to compare", COMPARE_TEXT)
text2 = ''.join(c for c in text2 if not c.isnumeric())
text2 = ''.join([x.replace('●','') for x in text2])
text2 = ''.join([x.replace('•','') for x in text2])
text2 = ''.join([x.replace('_','') for x in text2])
doc2 = process_text(spacy_model, text2)

sentences2 = [sent.text for sent in doc2.sents]
sentences2 = [x.replace('\n','') for x in sentences2]

embedding2 = model.encode(sentences2, convert_to_tensor=True)

# compute similarity scores of two embeddings
cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
score=[]
for i in range(len(sentences1)):
    for j in range(len(sentences2)):
        if cosine_scores[i][j] > 0.0:
            score.append(cosine_scores[i][j].item())
#             st.write("Sentence 1:", sentences1[i])
#             st.write("Sentence 2:", sentences2[j])
#             st.write("Similarity Score:", cosine_scores[i][j].item())
            
score_filtered = [i for i in score if i > 0.4]
          
average = np.round((sum(score_filtered)/len(score_filtered)), 2)
matchPercentage = average * 100

st.write("You resume matches", matchPercentage, "% with the job description.")

st.subheader("Distribution of the Score")
import matplotlib.pyplot as plt
import seaborn as sns
fig = plt.figure(figsize=(10,5))
ax = sns.distplot(score, bins=10)
ax.set_xlabel('Similarity Score of out 1')
ax.set_ylabel('Density')
st.pyplot(fig)


# st.subheader("Histogram of the Score")
# fig1 = plt.figure(figsize=(10,5))
# ax1 = sns.histplot(score, bins=10, stat='density')
# ax1.set_xlabel('Similarity Score of out 1')
# ax1.set_ylabel('Density')
# st.pyplot(fig1)
# x = np.array(score)
# fig = go.Figure(data=[go.Histogram(x=x, histnorm='probability')])
# fig.show()

# st.plotly_chart(fig)

# # st.pyplot(fig)
# st.bar_chart(fig)

# hist_values = np.histogram(
#     x)

# st.bar_chart(hist_values)

# st.markdown(f"> {sentences}")
# st.write('1st', sentences[0])



# sentences= []
# for sent in doc.sents:
#     sentences=sentences.append(sent.text)
#     st.write(sentences)

# sentences = list(doc.sents)
# st.write(sentences)


# sentences = list(doc.sents)

# 



# for sent in doc.sents:
#     html = displacy.render(sent)
# #     sentences = list(sent.text)
#     html = html.replace("\n\n", "\n")
#     if len(doc)>1:
#         sentences=list(doc.sents)
# #         st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
#         html1 = displacy.render(sentences)
#     st.write(sentences)

    
# st.write('the first sentence', sentences[0])
    
    
#     docs = [span.as_doc() for span in doc.sents] 
# #     if split_sents else [doc]
#     for sent in docs:
#         html = displacy.render(sent)
#         # Double newlines seem to mess with the rendering
#         html = html.replace("\n\n", "\n")
# #         if split_sents and len(docs) > 1:
# #             st.markdown(f"> {sent.text}")
# #         st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
#         st.write(sent)

    
# st.write(sentences[0])
