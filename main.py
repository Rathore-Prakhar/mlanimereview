import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import joblib
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download('punkt')
nltk.download('stopwords')


data = pd.read_csv('reviews.csv')


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)

data['processed_review'] = data['review'].apply(preprocess_text)

data['review_length'] = data['review'].str.len()

data['title_length'] = data['title'].str.len()

data['num_tags'] = data['tags'].str.count(',') + 1


N = 20
top_tags = data['tags'].str.split(',', expand=True).stack().value_counts().nlargest(N).index
for tag in top_tags:
    tag_escaped = re.escape(tag)  # Escape special characters
    data[f'tag_{tag}'] = data['tags'].str.contains(tag_escaped).astype(int)

vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(data['processed_review'])


X = np.hstack((
    X_tfidf.toarray(),
    data[['review_length', 'title_length', 'num_tags']].values,
    data[[f'tag_{tag}' for tag in top_tags]].values
))

y = data['sentiment']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Model accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Random Forest)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


feature_importance = rf_model.feature_importances_
feature_names = np.array(vectorizer.get_feature_names_out().tolist() + 
                         ['review_length', 'title_length', 'num_tags'] + 
                         [f'tag_{tag}' for tag in top_tags])

top_features = sorted(zip(feature_importance, feature_names), reverse=True)[:20]
importance, names = zip(*top_features)

plt.figure(figsize=(12, 6))
plt.bar(names, importance)
plt.title('Top 20 Most Important Features (Random Forest)')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


joblib.dump(rf_model, 'rf_model.joblib')

joblib.dump(vectorizer, 'vectorizer.joblib')

print("\nRandom Forest model and vectorizer saved.")

def predict_sentiment_rf(review, title, tags):
    processed_review = preprocess_text(review)
    
    review_length = len(review)
    
    title_length = len(title)
    
    num_tags = len(tags.split(','))
    
    tag_features = {f'tag_{tag}': 1 if tag in tags else 0 for tag in top_tags}
    
    vectorized_review = vectorizer.transform([processed_review]).toarray()
    
    additional_features = np.array([[review_length, title_length, num_tags]])
    
    tag_features_array = np.array([list(tag_features.values())])
    
    features = np.hstack((vectorized_review, additional_features, tag_features_array))
    
    prediction = rf_model.predict(features)[0]
    
    probability = rf_model.predict_proba(features)[0]
    return prediction, probability

# EXAMPLE
print("\nExample predictions with Random Forest model:")
example_reviews = [
    ("This anime was absolutely amazing! The character development was superb.", "Great Anime", "action,adventure"),
    ("I found the plot to be confusing and the pacing too slow.", "Slow Anime", "drama,slice of life"),
    ("While the animation was beautiful, the story left much to be desired.", "Beautiful but Lacking", "fantasy,romance")
]

for review, title, tags in example_reviews:
    prediction, probability = predict_sentiment_rf(review, title, tags)
    print(f"Review: {review}")
    print(f"Title: {title}")
    print(f"Tags: {tags}")
    print(f"Sentiment: {'Positive' if prediction == 1 else 'Negative'}")
    print(f"Confidence: {max(probability):.2f}\n")
