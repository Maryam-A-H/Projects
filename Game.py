import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Sentence to Emoji Predictor", page_icon="🤖")
st.title("🤖 Sentence to Emoji Predictor with Embeddings")

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("Train.csv")
    data = data.rename(columns={"French macaroon is so tasty":"Text","4":"Emoji"})
    data = data.iloc[:, 1:3].reset_index(drop=True)
    return data

data = load_data()
st.write("Data loaded:", data.shape)
st.write(data.head())

# Load mapping
@st.cache_data
def load_mapping():
    mapping = pd.read_csv("Mapping.csv").iloc[:,1:2]
    return mapping.squeeze().to_dict()

emoji_mapping = load_mapping()
st.write("### Emoji Mapping Sample")
# Display a sample of the mapping with emojis visible
mapping_sample = {k: v for k, v in list(emoji_mapping.items())[:20]}
st.write(pd.DataFrame(list(mapping_sample.items()), columns=["Label", "Emoji"]))

# Embed sentences
@st.cache_resource
def get_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = get_embedder()

@st.cache_data
def embed_texts(texts):
    return embedder.encode(texts, convert_to_tensor=False)

X_embeddings = embed_texts(data['TEXT'])
st.write("Embeddings shape:", np.array(X_embeddings).shape)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(data['Emoji'])

# Model selection and training
model_option = st.selectbox("Select model:", ["Logistic Regression", "Random Forest", "Support Vector Machine"])

@st.cache_resource
def train_model(X, y, model_name):
    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "Random Forest":
        model = RandomForestClassifier()
    else:
        model = SVC()
    model.fit(X, y)
    return model

model = train_model(X_embeddings, y_encoded, model_option)

# Input form & prediction
sentence = st.text_input("Write a sentence:", "I love pizza")

@st.cache_data
def embed_sentence(sentence):
    return embedder.encode([sentence], convert_to_tensor=False)

if sentence:
    sentence_embedding = embed_sentence(sentence)
    prediction_encoded = model.predict(sentence_embedding)[0]
    prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]
    
    predicted_emoji = emoji_mapping.get(int(prediction_label), prediction_label)
    st.markdown(f"### Predicted Emoji: {predicted_emoji}")
    st.write(f"Model used: **{model_option}**")

st.markdown("---")
st.markdown("### 🧠 How does this teach Data Science?")
st.write("""
- Uses sentence embeddings for semantic understanding.
- Shows different classifier performance.
- Maps text inputs to emojis as a prediction task.
""")

