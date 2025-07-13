import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sentence_transformers import SentenceTransformer

# ----------------------------
# 🌟 App Config
# ----------------------------
st.set_page_config(page_title="Sentence to Emoji Predictor (Word Embedding)", page_icon="🤖")
st.title("🤖 Sentence to Emoji Predictor with Embeddings")
st.write("Type a sentence and see which emoji best fits it, using word embeddings and trying different models!")

