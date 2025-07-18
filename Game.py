import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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

# 1. Show dataset sample (5 rows)
st.markdown("## ✨ Sample of the Dataset")
st.dataframe(data_full.sample(5)[["TEXT", "Label", "Emoji"]], use_container_width=True)

# 2. Show all emojis in the dataset
st.markdown("## All Emojis in Dataset! ")
unique_emojis = pd.DataFrame(data_full['Emoji'].unique(), columns=["Emoji"])
st.dataframe(unique_emojis, use_container_width=True)

# 3. Explain embeddings with image
st.markdown("""
### 🧠💬 How does the computer understand text?

Computers can’t read words like we do. They turn sentences into numbers, kind of like secret codes (using embedding), so they can understand what the words *mean* and find patterns. It’s like giving the computer a way to feel what the sentence is saying, even though it doesn’t speak our language.
""")

image_path = "1HOvcH2lZXWyOtmcqwniahQ.png"
if os.path.exists(image_path):
    image = Image.open(image_path)
    st.image(image, caption="Embeddings explained visually", use_container_width=True)
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

# 4. Sampling controls
st.markdown("## ⚙️ Sampling")

st.markdown("""
Imagine you’re trying to learn about animals, but your classroom has 90 cats and only 10 dogs.  
If you only look at what most kids are doing (playing with cats), you might forget how dogs behave!

Sampling is like making sure you have enough dogs *and* cats to learn about both.  
This way, you don’t just get really good at recognizing cats but also recognize the dogs.
It helps the computer not to be lazy and guess only the most common thing.
""")

col1, col2 = st.columns(2)
with col1:
    sample_frac = st.slider("Sample fraction", 0.01, 1.0, 0.1, 0.01)
with col2:
    sampling_type = st.radio("Sampling Method", ["Stratified", "Balanced"])

# Sampling functions without simple random
@st.cache_data
def stratified_sample(data, frac):
    return data.groupby('Label', group_keys=False).apply(
        lambda x: x.sample(frac=frac, random_state=42)
    ).reset_index(drop=True)

@st.cache_data
def balanced_sample(data):
    n_samples = int(500 * sample_frac)
    def sample_group(group):
        if len(group) >= n_samples:
            return group.sample(n=n_samples, random_state=42)
        else:
            return group.sample(n=n_samples, replace=True, random_state=42)
    return data.groupby('Label', group_keys=False).apply(sample_group).reset_index(drop=True)

# Apply sampling
with st.spinner("Sampling data..."):
    if sampling_type == "Stratified":
        data = stratified_sample(data_full, sample_frac)
    elif sampling_type == "Balanced":
        data = balanced_sample(data_full)

st.markdown(f"### 📊 Sampled Data: {data.shape}")

# Load embeddings for sampled data
with st.spinner("Loading embeddings..."):
    X_embeddings_full = load_embeddings(sampling_type)
    sampled_indices = data.index.tolist()
    X_embeddings = X_embeddings_full[sampled_indices]

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

# 5. Model selection (only Logistic Regression and Random Forest)
st.markdown("## Pick a Model!")

# Kid-friendly explanations
model_explanations = {
    "Logistic Regression": """
    **Logistic Regression** is like a magic sorting hat.  
    It looks at features and guesses if something is one thing or another, like deciding if a pet is a cat or a dog by drawing a line between groups.
    """,
    "Random Forest": """
    **Random Forest** is like asking a group of friends to guess.  
    Each friend looks at the features and votes, and the most popular guess wins!
    """
}

model_option = st.selectbox("Choose Model", ["Logistic Regression", "Random Forest"])

st.markdown("### How the model works:")
st.info(model_explanations[model_option])


# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(data['Label'])

# Load or train model with caching
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
            model = LogisticRegression(max_iter=500, solver='liblinear')  # faster solver & fewer iterations
        else:
            model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)  # smaller forest

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

# 6. Sentence input and prediction
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





