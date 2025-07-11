# Save as app.py and run with 'streamlit run app.py'

import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("👗 Fashion AI Lab")

# Load data
df = pd.read_csv("Womens Clothing E-Commerce Reviews.csv")
df = df.dropna(subset=['Rating'])

# Feature selection
features = ['Age', 'Department Name', 'Class Name']
selected_features = st.multiselect("Select features to use:", features, default=['Age'])

X = df[selected_features]
y = (df['Rating'] >= 4).astype(int)

X = pd.get_dummies(X, drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model hyperparameter
max_depth = st.slider("Select max depth for Decision Tree:", 1, 10, 5)

# Train model button
if st.button("Train AI Model"):
    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.success(f"Model Accuracy: {acc:.2f}")

