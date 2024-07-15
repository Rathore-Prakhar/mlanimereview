import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')

data = pd.read_csv('reviews.csv')

print("Original data shape:", data.shape)
print("\nSample data:")
print(data.head())

print("\nDataset Info:")
data.info()

print("\nMissing values:")
print(data.isnull().sum())

print("\nDataset statistics:")
print(data.describe())

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)

data['processed_review'] = data['review'].apply(preprocess_text)

data['sentiment'] = data['processed_review'].apply(lambda x: 1 if 'good' in x or 'great' in x else 0)

plt.figure(figsize=(8, 6))
data['sentiment'].value_counts().plot(kind='bar')
plt.title('Distribution of Sentiments')
plt.xlabel('Sentiment (0: Negative, 1: Positive)')
plt.ylabel('Count')
plt.show()

data['review_length'] = data['processed_review'].str.len()
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='review_length', bins=50)
plt.title('Distribution of Processed Review Lengths')
plt.xlabel('Review Length')
plt.ylabel('Count')
plt.show()

X = data['processed_review']
y = data['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

print("\nProcessed data split and saved to CSV files.")

def get_most_common_words(texts, n=20):
    all_words = [word for text in texts for word in text.split()]
    word_counts = Counter(all_words)
    return word_counts.most_common(n)

positive_reviews = X_train[y_train == 1]
negative_reviews = X_train[y_train == 0]

print("\nMost common words in positive reviews:")
print(get_most_common_words(positive_reviews))

print("\nMost common words in negative reviews:")
print(get_most_common_words(negative_reviews))
