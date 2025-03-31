import numpy as np
import pandas as pd
import re
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Download stopwords
nltk.download('stopwords', quiet=True)

# Load dataset
df = pd.read_csv('WELFake_Dataset.csv').fillna(' ')

# Text Preprocessing (Stemming)
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower().split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

df['title'] = df['title'].astype(str).apply(preprocess_text)

# Prepare dataset
X = df['title']
y = df['label']

# Convert text to numerical vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.title("📰 Fake News Detector")
st.markdown("### Enter a news headline to check its authenticity")

# Text input
input_text = st.text_input("Enter News Headline:", "")

# Prediction function
def predict_fake_news(text):
    text = preprocess_text(text)  # Preprocess input text
    input_data = vectorizer.transform([text])
    return model.predict(input_data)[0]

# Check news when button is clicked
if st.button("Check News"):
    if not input_text.strip():
        st.warning("⚠️ Please enter a valid news headline!")
    else:
        pred = predict_fake_news(input_text)
        if pred == 1:
            st.error("🚨 This news article is **Fake**!")
        else:
            st.success("✅ This news article is **Real**!")

   

    
    



            
 
