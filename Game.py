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
    <h6 style="color: black; font-family: 'Georgia', serif; 
               text-align: center; margin-top: 5px;">
        Really dig into your dataset. Find out what's there? What's the distribution of the dataset? <b>Most Important Step</b>
    </h6>
    """,
    unsafe_allow_html=True
)

# Sidebar: Number of samples per class slider
n_samples_per_class = st.sidebar.slider(
    "Number of samples per class", min_value=1, max_value=1000, value=100, step=10
)

# Load full dataset
@st.cache_data
def load_data():
    data = pd.read_csv("Train_processed_sampled.csv")  # full dataset
    return data

data_full = load_data()

st.write("Full dataset loaded:", data_full.shape)
st.write(data_full.head())

# Load emoji mapping
@st.cache_data
def load_mapping():
    mapping = pd.read_csv("Mapping.csv").iloc[:, 1:2]
    return mapping.squeeze().to_dict()

emoji_mapping = load_mapping()
st.write("### Emoji Mapping Sample")
mapping_sample = {k: v for k, v in list(emoji_mapping.items())[:20]}
st.write(pd.DataFrame(list(mapping_sample.items()), columns=["Label", "Emoji"]))

# Balanced sampling with resampling if needed
@st.cache_data
def balanced_sample(data, n_samples):
    def sample_group(group):
        if len(group) >= n_samples:
            return group.sample(n=n_samples, random_state=42)
        else:
            return group.sample(n=n_samples, replace=True, random_state=42)
    sampled = data.groupby('Label', group_keys=False).apply(sample_group).reset_index(drop=True)
    return sampled

data = balanced_sample(data_full, n_samples_per_class)

st.write(f"Using balanced sampled data: {data.shape}")
st.write(data['Label'].value_counts())

# Bar chart with emojis using Altair
label_counts = data['Label'].value_counts().reset_index()
label_counts.columns = ['Label', 'Count']
label_counts['Emoji'] = label_counts['Label'].map(emoji_mapping)

chart = (
    alt.Chart(label_counts)
    .mark_bar(color='skyblue')
    .encode(
        x=alt.X('Emoji:N', title='Emoji', sort=None),
        y=alt.Y('Count:Q', title='Count'),
        tooltip=[alt.Tooltip('Label:N'), alt.Tooltip('Count:Q')]
    )
    .properties(title="Balanced Sampled Data Label Distribution", width=600, height=350)
)

st.altair_chart(chart, use_container_width=True)

# Load embeddings (for full dataset)
@st.cache_data
def load_embeddings():
    return np.load("train_embeddings_sampled.npy")  # Make sure path matches

X_embeddings_full = load_embeddings()

# Get sampled indices and select corresponding embeddings
sampled_indices = data.index.tolist()
X_embeddings = X_embeddings_full[sampled_indices]

st.write("Embeddings shape (sampled):", X_embeddings.shape)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(data['Label'])

# Model selection dropdown
model_option = st.selectbox("Select model:", ["Logistic Regression", "Random Forest", "Support Vector Machine"])

# Load or train model
@st.cache_resource
def get_model(name, X, y):
    filename = f"{name.lower().replace(' ', '_')}_model.joblib"
    try:
        model = joblib.load(filename)
        st.write(f"Loaded saved {name} model.")
    except Exception:
        if name == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif name == "Random Forest":
            model = RandomForestClassifier()
        else:
            model = SVC(probability=True)
        model.fit(X, y)
        joblib.dump(model, filename)
        st.write(f"Trained and saved {name} model.")
    return model

model = get_model(model_option, X_embeddings, y_encoded)

# Embedder for new inputs
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
- Lets you dynamically sample your dataset for fast experimentation.
""")



