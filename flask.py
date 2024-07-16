from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
import re
import nltk
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import bigrams
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import io
import base64
import seaborn as sns
from transformers import pipeline
import torch

plt.switch_backend('Agg')

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

best_model = joblib.load('best_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')
data = pd.read_csv('reviews.csv')

device = 0 if torch.cuda.is_available() else -1
summarizer = pipeline('summarization', model='sshleifer/distilbart-cnn-12-6', device=device)

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

def categorize_tags(tag):
    if 'Not Recommended' in tag:
        return 'Not Recommended'
    elif 'Recommended' in tag:
        return 'Recommended'
    elif 'Mixed Feelings' in tag:
        return 'Mixed Feelings'
    else:
        return 'Uncategorized'

data['processed_review'] = data['review'].apply(preprocess_text)
data['category'] = data['tags'].apply(categorize_tags)
data = data[data['category'] != 'Uncategorized']

analyzer = SentimentIntensityAnalyzer()
data['vader_score'] = data['processed_review'].apply(lambda x: analyzer.polarity_scores(x))
data['vader_compound'] = data['vader_score'].apply(lambda score_dict: score_dict['compound'])
data['vader_negative'] = data['vader_score'].apply(lambda score_dict: score_dict['neg'])
data['vader_neutral'] = data['vader_score'].apply(lambda score_dict: score_dict['neu'])
data['vader_positive'] = data['vader_score'].apply(lambda score_dict: score_dict['pos'])

data['review_length'] = data['processed_review'].apply(lambda x: len(x.split()))
data['average_word_length'] = data['processed_review'].apply(lambda x: sum(len(word) for word in x.split()) / len(x.split()))
data['review_polarity'] = data['vader_compound'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))

@app.route('/')
def home():
    return render_template('index.html')

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

def chunk_text(text, max_length=1024):
    words = text.split()
    for i in range(0, len(words), max_length):
        yield ' '.join(words[i:i + max_length])

@app.route('/anime/<title>', methods=['GET'])
def anime_details(title):
    anime_data = data[data['title'].str.lower() == title.lower()]
    if anime_data.empty:
        return jsonify({'error': 'Anime title not found'}), 404

    average_vader = anime_data['vader_compound'].mean()
    
    plt.figure(figsize=(10, 5))
    plt.hist(anime_data['review_length'], bins=20, color='blue', edgecolor='black')
    plt.xlabel('Review Length (words)')
    plt.ylabel('Number of Reviews')
    plt.title(f'Distribution of Review Lengths for {title}')
    plt.grid(axis='y')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    review_length_dist_url = base64.b64encode(img.getvalue()).decode()

    plt.figure(figsize=(10, 5))
    plt.hist(anime_data['vader_compound'], bins=20, color='green', edgecolor='black')
    plt.xlabel('VADER Compound Score')
    plt.ylabel('Number of Reviews')
    plt.title(f'Sentiment Scores Distribution for {title}')
    plt.grid(axis='y')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    sentiment_scores_dist_url = base64.b64encode(img.getvalue()).decode()

    polarity_counts = anime_data['review_polarity'].value_counts().to_dict()
    
    bigram_counter = Counter(bigram for review in anime_data['processed_review'] for bigram in bigrams(review.split()))
    most_common_bigrams = bigram_counter.most_common(10)
    
    y_true = anime_data['category'].apply(lambda x: 1 if x == 'Recommended' else 0).tolist()
    y_pred = anime_data['processed_review'].apply(lambda x: predict_sentiment(x)[0]).tolist()
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    confusion_matrix_url = base64.b64encode(img.getvalue()).decode()

    top_reviews = anime_data[['review', 'vader_compound']].sort_values(by='vader_compound', ascending=False).head(5).to_dict(orient='records')
    top_reviews_text = " ".join([review['review'] for review in top_reviews])

    summaries = []
    for chunk in chunk_text(top_reviews_text, max_length=200):
        try:
            summary = summarizer(chunk, max_length=100, min_length=50, do_sample=False)[0]['summary_text']
            summaries.append(summary)
        except IndexError as e:
            summaries.append("Summarization failed due to input length issues.")

    combined_summary = ' '.join(summaries)

    response = {
        'title': title,
        'average_vader': average_vader,
        'review_length_dist_url': review_length_dist_url,
        'sentiment_scores_dist_url': sentiment_scores_dist_url,
        'confusion_matrix_url': confusion_matrix_url,
        'total_reviews': len(anime_data),
        'top_reviews': top_reviews,
        'summary': combined_summary,
        'word_count': Counter(' '.join(anime_data['processed_review']).split()).most_common(10),
        'polarity_counts': polarity_counts,
        'most_common_bigrams': most_common_bigrams,
        'average_word_length': anime_data['average_word_length'].mean()
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
