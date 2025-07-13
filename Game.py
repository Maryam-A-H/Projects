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

# ----------------------------
# 📥 Load Dataset and Mapping
# ----------------------------
data = pd.read_csv("Train.csv")
data = data.rename(columns={"French macaroon is so tasty":"Text","4":"Emoji"})
data = data.iloc[:, 1:3]
data = data.reset_index(drop=True)

mapping = pd.read_csv("Mapping.csv")
mapping = mapping.iloc[:,1:2]

# Create mapping dictionary
emoji_mapping = mapping.squeeze().to_dict()

# ----------------------------
# 🔬 Prepare embeddings
# ----------------------------
model_name = 'all-MiniLM-L6-v2'
embedder = SentenceTransformer(model_name)

X_embeddings = embedder.encode(data['Text'], convert_to_tensor=False)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(data['Emoji'])

# ----------------------------
# 🤖 Model Selection
# ----------------------------
model_option = st.selectbox("Select model:", ["Logistic Regression", "Random Forest", "Support Vector Machine"])

if model_option == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
elif model_option == "Random Forest":
    model = RandomForestClassifier()
else:
    model = SVC()

model.fit(X_embeddings, y_encoded)

# ----------------------------
# 📝 Input Form
# ----------------------------
sentence = st.text_input("Write a sentence:", "I love pizza")

if sentence:
    sentence_embedding = embedder.encode([sentence], convert_to_tensor=False)
    prediction_encoded = model.predict(sentence_embedding)[0]
    prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]
    
    # Map prediction to emoji using mapping.csv if needed
    predicted_emoji = emoji_mapping.get(int(prediction_label), prediction_label)

    st.markdown(f"### Predicted Emoji: {predicted_emoji}")

    st.write(f"This prediction uses **{model_option} + sentence embeddings** trained on your dataset to map sentences to emojis.")

# ----------------------------
# 💡 Teaching Note
# ----------------------------
st.markdown("---")
st.markdown("### 🧠 How does this teach Data Science?")
st.write("""
Here, we:
- Used **sentence embeddings** for semantic understanding.
- Allowed switching between **Logistic Regression, Random Forest, and SVM** classifiers.
- Predicted the best matching emoji for new sentences.

This shows how different models perform on the same feature representations.
""")

# ----------------------------
# ✅ Next Steps
# ----------------------------
# Expand to fine-tune transformer models for production.

# End of Script

# Save as app.py and run with:
# pip install streamlit pandas scikit-learn sentence-transformers
# streamlit run app.py

# Let me know if you want it extended for full emoji decoding explanations for your kids’ data science workshops.
