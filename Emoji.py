import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# ----------------------------
# 🌟 App Config
# ----------------------------
st.set_page_config(page_title="Sentence to Emoji Predictor", page_icon="🤖")
st.title("🤖 Sentence to Emoji Predictor")
st.write("Type a sentence and see which emoji best fits it!")

# ----------------------------
# 📥 Load Dataset
# ----------------------------
@st.cache
def load_data():
    return pd.read_csv("emojify_data.csv")
df = load_data()
df  = df.iloc[:, 0:2]
df = df.rename(columns= {"French macaroon is so tasty":"Text","4":"Emoji"})
df
# ----------------------------
# 🔬 Train Classifier
# ----------------------------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['Text'])
y = df['Emoji']

model = MultinomialNB()
model.fit(X, y)
# ----------------------------
# 📝 Input Form
# ----------------------------
sentence = st.text_input("Write a sentence:")

if sentence:
    X_test = vectorizer.transform([sentence])
    prediction = model.predict(X_test)[0]
    st.markdown(f"### Predicted Emoji: {prediction}")
    st.write("This prediction uses a trained model from the Emojify dataset to map sentences to emojis.")

# ----------------------------
# 💡 Teaching Note
# ----------------------------
st.markdown("---")
st.markdown("### 🧠 How does this teach Data Science?")
st.write("""
In this app, we:
- **Load** a dataset of text-emoji pairs.
- **Vectorize** the text into numerical features.
- **Train** a Naive Bayes classifier to predict emojis.
- **Predict** the emoji for new sentences.

This demonstrates how machine learning models can learn from data and make predictions.
""")

