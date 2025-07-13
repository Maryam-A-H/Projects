import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sentence_transformers import SentenceTransformer
import joblib  # for loading saved models

st.set_page_config(page_title="Sentence to Emoji Predictor", page_icon="🤖")
st.markdown(
    """
    <h1 style='
        text-align: center; 
        color: green; 
        text-shadow: 2px 2px 4px #000000;
        font-family: "Arial Black", Gadget, sans-serif;
        letter-spacing: 2px;
    '>
        🤖 Sentence to Emoji Predictor with Precomputed Embeddings
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h3 style="color: black; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
               text-align: center; margin-bottom: 0;">
        Pre-Processed Data Sample
    </h3>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h6 style="color: black; font-style: italic; font-family: 'Georgia', serif; 
               text-align: center; margin-top: 5px;">
        Really dig into your dataset.
    </h6>
    """,
    unsafe_allow_html=True
)


# Load processed sampled data
@st.cache_data
def load_data():
    data = pd.read_csv("Train_processed_sampled.csv")  # sampled CSV with 'Label' and 'TEXT'
    return data

data = load_data()
st.write("Data loaded:", data.shape)
st.write(data.head())

# Load emoji mapping
@st.cache_data
def load_mapping():
    mapping = pd.read_csv("Mapping.csv").iloc[:, 1:2]
    return mapping.squeeze().to_dict()

emoji_mapping = load_mapping()
st.write("### Emoji Mapping Sample")
mapping_sample = {k: v for k, v in list(emoji_mapping.items())[:20]}
st.write(pd.DataFrame(list(mapping_sample.items()), columns=["Label", "Emoji"]))

# Load precomputed embeddings (aligned with data rows)
@st.cache_data
def load_embeddings():
    return np.load("train_embeddings_sampled.npy")

X_embeddings = load_embeddings()
st.write("Embeddings shape:", X_embeddings.shape)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(data['Label'])

# Model selection dropdown
model_option = st.selectbox("Select model:", ["Logistic Regression", "Random Forest", "Support Vector Machine"])

# Load saved models if available, else train and save them
@st.cache_resource
def get_model(name):
    filename = f"{name.lower().replace(' ', '_')}_model.joblib"
    try:
        model = joblib.load(filename)
        st.write(f"Loaded saved {name} model.")
    except Exception:
        # Train model if not found
        if name == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif name == "Random Forest":
            model = RandomForestClassifier()
        else:
            model = SVC(probability=True)
        model.fit(X_embeddings, y_encoded)
        joblib.dump(model, filename)
        st.write(f"Trained and saved {name} model.")
    return model

model = get_model(model_option)

# Embedder for new user inputs
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

st.markdown("---")
st.markdown("### 🧠 How does this teach Data Science?")
st.write("""
- Uses precomputed sentence embeddings for fast training.
- Loads or trains multiple classifiers.
- Maps text inputs to emojis as a prediction task.
""")


