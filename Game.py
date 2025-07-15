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
from PIL import Image
import os

# Page config
st.set_page_config(page_title="Sentence to Emoji Predictor", page_icon="🤖", layout="wide")

# Title
st.markdown(
    """
    <h1 style='text-align: center; color: green; text-shadow: 2px 2px 4px #000000; font-family: "Arial Black"; letter-spacing: 2px;'>
        🤖 Sentence to Emoji Predictor
    </h1>
    """,
    unsafe_allow_html=True
)

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
data_full["Emoji"] = data_full["Label"].map(emoji_mapping)


# Sampling functions
@st.cache_data
def stratified_sample(data, frac):
    return data.groupby('Label', group_keys=False).apply(
        lambda x: x.sample(frac=frac, random_state=42)
    ).reset_index(drop=True)

@st.cache_data
def balanced_sample(data):
    n_samples = 500*(sample_frac)
    def sample_group(group):
        if len(group) >= n_samples:
            return group.sample(n=n_samples, random_state=42)
        else:
            return group.sample(n=n_samples, replace=True, random_state=42)
    return data.groupby('Label', group_keys=False).apply(sample_group).reset_index(drop=True)

@st.cache_data
def simple_random_sample(data, frac):
    return data.sample(frac=frac, random_state=42).reset_index(drop=True)

# Apply sampling
with st.spinner("Sampling data..."):
    if sampling_type == "Stratified":
        data = stratified_sample(data_full, sample_frac)
    elif sampling_type == "Balanced":
        data = balanced_sample(data_full)
    else:
        data = simple_random_sample(data_full, sample_frac)

st.markdown(f"### 📊 Using {sampling_type} sampled data: {data.shape}")

# Display dataset sample
st.markdown("#### ✨ Sample Tweets with Emojis")
st.dataframe(data.sample(10)[["TEXT", "Label", "Emoji"]], use_container_width=True)

# Label distribution chart
label_counts = data['Label'].value_counts().reset_index()
label_counts.columns = ['Label', 'Count']
label_counts['Emoji'] = label_counts['Label'].map(emoji_mapping).fillna(label_counts['Label'].astype(str))

chart = (
    alt.Chart(label_counts)
    .mark_bar(color='lightgreen')
    .encode(
        x=alt.X('Emoji:N', title='Emoji', sort=None),
        y=alt.Y('Count:Q', title='Count'),
        tooltip=[alt.Tooltip('Label:N'), alt.Tooltip('Count:Q')]
    )
    .properties(title="Sampled Data Label Distribution", width=600, height=350)
)
st.altair_chart(chart, use_container_width=True)

# Embeddings explanation with image
st.markdown("## 🤔 What are Embeddings?")
image_path = "1HOvcH2lZXWyOtmcqwniahQ.png"
if os.path.exists(image_path):
    image = Image.open(image_path)
    st.image(image, caption="This is a PNG image", use_container_width=True)
else:
    st.warning("Embedding explanation image not found.")

# Load embeddings
@st.cache_data
def load_embeddings(sampling_type):
    if sampling_type == "Stratified":
        return np.load("train_embeddings_sampled.npy")
    elif sampling_type == "Balanced":
        return np.load("train_embeddings_balanced_sampled.npy")
    else:
        return np.load("train_embeddings_sampled.npy")

with st.spinner("Loading embeddings..."):
    X_embeddings_full = load_embeddings(sampling_type)
    sampled_indices = data.index.tolist()
    X_embeddings = X_embeddings_full[sampled_indices]

st.write("✅ Embeddings shape (sampled):", X_embeddings.shape)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(data['Label'])

# Load or train model
@st.cache_resource
def get_model(name, X, y, sampling_type):
    if sampling_type == "Balanced":
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

with st.spinner(f"Loading {model_option} model..."):
    model = get_model(model_option, X_embeddings, y_encoded, sampling_type)

# Embedder
@st.cache_resource
def get_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = get_embedder()

@st.cache_data
def embed_sentence(sentence):
    return embedder.encode([sentence], convert_to_tensor=False)
    
# Controls within main page
st.markdown("## ⚙️ Sampling & Model Selection")

col1, col2, col3 = st.columns(3)
with col1:
    sample_frac = st.slider("Sample fraction", 0.01, 1.0, 0.1, 0.01)
with col2:
    sampling_type = st.radio("Sampling Method", ["Stratified", "Balanced", "Simple Random"])
with col3:
    model_option = st.selectbox("Choose Model", ["Logistic Regression", "Random Forest", "Support Vector Machine"])



# Prediction input
st.markdown("## ✍️ Write a sentence to predict its emoji")
sentence = st.text_input("Your sentence here:", "I love pizza")

if sentence:
    with st.spinner("Predicting emoji..."):
        sentence_embedding = embed_sentence(sentence)
        prediction_encoded = model.predict(sentence_embedding)[0]
        prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]
        predicted_emoji = emoji_mapping.get(int(prediction_label), prediction_label)
        probs = model.predict_proba(sentence_embedding)[0]
        max_prob = np.max(probs)

    st.success(f"### 🎯 Predicted Emoji: {predicted_emoji}")
    st.write(f"Confidence: {max_prob:.2f}")
    st.caption(f"Model used: **{model_option}**")

# Footer
st.markdown("""
<hr>
<p style='text-align: center;'>
Built by [Your Name] | 🤖 AI Emoji Predictor | 📝 [GitHub](your-repo-link)
</p>
""", unsafe_allow_html=True)



