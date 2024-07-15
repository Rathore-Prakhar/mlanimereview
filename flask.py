from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

best_model = joblib.load('best_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens) 

def predict_sentiment(review):
    processed_review = preprocess_text(review)
    vectorized_review = vectorizer.transform([processed_review])
    prediction = best_model.predict(vectorized_review)[0]
    probability = best_model.predict_proba(vectorized_review)[0]
    return prediction, probability

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    review = data['review']
    prediction, probability = predict_sentiment(review)
    
    response = {
        'review': review,
        'sentiment': 'Positive' if prediction == 1 else 'Negative',
        'confidence': float(max(probability))
    }
    
    return jsonify(response)

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    data = request.json
    reviews = data['reviews']
    
    results = []
    for review in reviews:
        prediction, probability = predict_sentiment(review)
        results.append({
            'review': review,
            'sentiment': 'Positive' if prediction == 1 else 'Negative',
            'confidence': float(max(probability))
        })
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
