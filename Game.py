import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sentence_transformers import SentenceTransformer
import joblib
import altair as alt

# Page config
st.set_page_config(page_title="Sentence to Emoji Predictor", page_icon="🤖")

# Title
st.markdown(
    """
    <h1 style='
        text-align: center; 
        color: green; 
        text-shadow: 2px 2px 4px #000000;
        font-family: "Arial Black", Gadget, sans-serif;
        letter-spacing: 2px;
    '>
        🤖 Sentence to Emoji Predictor
    </h1>
    """,
    unsafe_allow_html=True
)

st.write("### Pre-Processing Data")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("Train_processed_sampled.csv")

data_full = load_data()

# Load emoji mapping
@st.cache_data
def load_mapping():
    mapping = pd.read_csv("Mapping.csv").iloc[:, 1:2]
    return mapping.squeeze().to_dict()

emoji_mapping = load_mapping()

# Map labels to emojis
data_full["Emoji"] = data_full["Label"].map(emoji_mapping)

st.write("Full dataset loaded:", data_full.shape)
st.write("### Sample Twitter Text with Emoji")
st.write(data_full.sample(15)[["TEXT", "Label", "Emoji"]])

st.write(data_full["Emoji"].unique())

# Sampling functions
@st.cache_data
def stratified_sample(data, frac):
    return data.groupby('Label', group_keys=False).apply(
        lambda x: x.sample(frac=frac, random_state=42)
    ).reset_index(drop=True)

@st.cache_data
def balanced_sample(data):
    n_samples = 500
    def sample_group(group):
        if len(group) >= n_samples:
            return group.sample(n=n_samples, random_state=42)
        else:
            return group.sample(n=n_samples, replace=True, random_state=42)
    return data.groupby('Label', group_keys=False).apply(sample_group).reset_index(drop=True)

@st.cache_data
def simple_random_sample(data, frac):
    return data.sample(frac=frac, random_state=42).reset_index(drop=True)
# Sampling controls with session state to remember choice
sample_frac = st.slider("Select sample fraction", 0.01, 1.0, 0.1, 0.01)

# Initialize session_state keys if not exist
if 'sampling_type' not in st.session_state:
    st.session_state['sampling_type'] = None
if 'model_option' not in st.session_state:
    st.session_state['model_option'] = "Logistic Regression"

col1, col2, col3 = st.columns(3)

if col1.button("Stratified Sampling"):
    st.session_state['sampling_type'] = "stratified"

if col2.button("Balanced Sampling"):
    st.session_state['sampling_type'] = "balanced"

if col3.button("Simple Random Sampling"):
    st.session_state['sampling_type'] = "simple"

# Default sampling if none selected
if st.session_state['sampling_type'] is None:
    st.write("No sampling method selected yet. Using balanced sampling as default.")
    st.session_state['sampling_type'] = "balanced"

sampling_type = st.session_state['sampling_type']

# Load sampled data based on sampling_type
if sampling_type == "stratified":
    data = stratified_sample(data_full, sample_frac)
elif sampling_type == "balanced":
    data = balanced_sample(data_full)
else:
    data = simple_random_sample(data_full, sample_frac)

st.write(f"Using {sampling_type} sampled data: {data.shape}")

# Label counts and chart (unchanged)
label_counts = data['Label'].value_counts().reset_index()
label_counts.columns = ['Label', 'Count']
label_counts['Emoji'] = label_counts['Label'].map(emoji_mapping).fillna(label_counts['Label'].astype(str))

chart = (
    alt.Chart(label_counts)
    .mark_bar(color='skyblue')
    .encode(
        x=alt.X('Emoji:N', title='Emoji', sort=None),
        y=alt.Y('Count:Q', title='Count'),
        tooltip=[alt.Tooltip('Label:N'), alt.Tooltip('Count:Q')]
    )
    .properties(title="Sampled Data Label Distribution", width=600, height=350)
)
st.altair_chart(chart, use_container_width=True)

st.write("### Whats Embeddings? ")

import streamlit as st
from PIL import Image

import os

image_path = "1HOvcH2lZXWyOtmcqwniahQ.png"
if os.path.exists(image_path):
    image = Image.open(image_path)
    st.image(image, caption="This is a PNG image", use_container_width=True)
else:
    st.error(f"Image file not found: {image_path}")


st.write("### Give it a try? ")
# Load embeddings according to sampling_type
@st.cache_data
def load_embeddings(sampling_type):
    if sampling_type == "stratified":
        return np.load("train_embeddings_sampled.npy")
    elif sampling_type == "balanced":
        return np.load("train_embeddings_balanced_sampled.npy")
    else:
        return np.load("train_embeddings_sampled.npy")

X_embeddings_full = load_embeddings(sampling_type)
sampled_indices = data.index.tolist()
X_embeddings = X_embeddings_full[sampled_indices]
st.write("Embeddings shape (sampled):", X_embeddings.shape)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(data['Label'])

# Model selection with session_state persistence
model_option = st.selectbox(
    "Select model:",
    ["Logistic Regression", "Random Forest", "Support Vector Machine"],
    index=["Logistic Regression", "Random Forest", "Support Vector Machine"].index(st.session_state['model_option'])
)

st.session_state['model_option'] = model_option

# Load or train model
@st.cache_resource
def get_model(name, X, y, sampling_type):
    if sampling_type == "balanced":
        filename = f"{name.lower().replace(' ', '_')}_{sampling_type}_model.joblib"
    else:
        filename = f"{name.lower().replace(' ', '_')}_model.joblib"

    try:
        model = joblib.load(filename)
    except Exception:
        if name == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif name == "Random Forest":
            model = RandomForestClassifier()
        else:
            model = SVC(probability=True)
        model.fit(X, y)
        joblib.dump(model, filename)
    return model

model = get_model(model_option, X_embeddings, y_encoded, sampling_type)

# Embedder & Prediction input (unchanged)
@st.cache_resource
def get_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = get_embedder()

@st.cache_data
def embed_sentence(sentence):
    return embedder.encode([sentence], convert_to_tensor=False)

sentence = st.text_input("Write a sentence:", "I love pizza")

if sentence:
    sentence_embedding = embed_sentence(sentence)
    prediction_encoded = model.predict(sentence_embedding)[0]
    prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]
    predicted_emoji = emoji_mapping.get(int(prediction_label), prediction_label)
    st.markdown(f"### Predicted Emoji: {predicted_emoji}")
    st.write(f"Model used: **{model_option}**")


