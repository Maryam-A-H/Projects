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
        🤖 Sentence to Emoji Predictor
    </h1>
    """,
    unsafe_allow_html=True
)

st.write("## Pre-Processing Data")
st.write("##### Really dig into your dataset. Find out what's there? What's the distribution of the dataset? * Most Important*")
# Sidebar controls
sample_frac = st.sidebar.slider(
    "Sampling fraction per class (percentage)", min_value=1, max_value=100, value=10, step=1
) / 100.0  # convert to fraction

balanced_sampling = st.sidebar.checkbox(
    "Use balanced sampling (equal samples per class with resampling)", value=True
)

import streamlit as st
import pandas as pd

# Load full dataset
@st.cache_data
def load_data():
    data = pd.read_csv("Train_processed_sampled.csv")  # full dataset
    return data

data_full = load_data()

# Load emoji mapping
@st.cache_data
def load_mapping():
    mapping = pd.read_csv("Mapping.csv").iloc[:, 1:2]
    return mapping.squeeze().to_dict()

emoji_mapping = load_mapping()

# Map labels to emojis before displaying
data_full["Emoji"] = data_full["Label"].map(emoji_mapping)

st.write("Full dataset loaded:", data_full.shape)
st.write("### Sample with Emoji Mapped")
st.write(data_full.sample(15)[["TEXT", "Label", "Emoji"]])  # adjust column names

# Show emoji mapping sample
st.write("### Emoji Mapping Dataset")
mapping_sample = {k: v for k, v in list(emoji_mapping.items())[:20]}
st.write(pd.DataFrame(list(mapping_sample.items()), columns=["Label", "Emoji"]))

### NEW 

import streamlit as st
import pandas as pd

st.write("### Sampling")

# Load full dataset
@st.cache_data
def load_data():
    return pd.read_csv("Train_processed_sampled.csv")

data_full = load_data()

# Define sampling functions
@st.cache_data
def stratified_sample(data, frac):
    return data.groupby('Label', group_keys=False).apply(
        lambda x: x.sample(frac=frac, random_state=42)
    ).reset_index(drop=True)

@st.cache_data
def balanced_sample(data, frac):
    min_class_size = data['Label'].value_counts().min()
    n_samples = max(1, int(min_class_size * frac))
    
    def sample_group(group):
        if len(group) >= n_samples:
            return group.sample(n=n_samples, random_state=42)
        else:
            return group.sample(n=n_samples, replace=True, random_state=42)
    
    return data.groupby('Label', group_keys=False).apply(sample_group).reset_index(drop=True)

@st.cache_data
def simple_random_sample(data, frac):
    return data.sample(frac=frac, random_state=42).reset_index(drop=True)

# Sampling fraction input
sample_frac = st.slider("Select sample fraction", 0.01, 1.0, 0.1, 0.01)
# Initialize data to None before sampling
data = None

# Buttons for sampling methods
col1, col2, col3 = st.columns(3)

# Always show buttons and determine sampling method
if col1.button("Stratified Sampling"):
    data = stratified_sample(data_full, sample_frac)
    st.write(f"Using stratified sampled data ({sample_frac*100:.1f}% per class): {data.shape}")

if col2.button("Balanced Sampling"):
    data = balanced_sample(data_full, sample_frac)
    st.write(f"Using balanced sampled data ({sample_frac*100:.1f}% of smallest class size per class): {data.shape}")

if col3.button("Simple Random Sampling"):
    data = simple_random_sample(data_full, sample_frac)
    st.write(f"Using simple random sampled data ({sample_frac*100:.1f}% of total data): {data.shape}")

# If no sampling button clicked yet
if data is None:
    st.write("No sampling method selected yet.")
    data = balanced_sample(data_full, sample_frac)
    
else:
    # Show sample only if data is available
    st.write("### Sampled data preview")
    
# Show bar chart of label distribution with emojis
st.write("### Sampled Data Label Distribution")

# Prepare label counts with emoji mapping
label_counts = data['Label'].value_counts().reset_index()
label_counts.columns = ['Label', 'Count']
label_counts['Emoji'] = label_counts['Label'].map(emoji_mapping)

# Handle missing emojis if any label not in mapping
label_counts['Emoji'] = label_counts['Emoji'].fillna(label_counts['Label'].astype(str))

# Display bar chart with emojis
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

##N#@@@
# Load embeddings (full dataset)
@st.cache_data
def load_embeddings():
    return np.load("train_embeddings_sampled.npy")  # Make sure path matches

X_embeddings_full = load_embeddings()

# Select embeddings corresponding to sampled data rows
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



