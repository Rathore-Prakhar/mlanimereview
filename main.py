import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

X_train = train_data['processed_review']
y_train = train_data['sentiment']
X_test = test_data['processed_review']
y_test = test_data['sentiment']

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

print(f"Number of features: {X_train_vectorized.shape[1]}")

model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

y_pred = model.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

feature_importance = model.feature_log_prob_[1] - model.feature_log_prob_[0]
feature_names = vectorizer.get_feature_names_out()

top_features = sorted(zip(feature_importance, feature_names), reverse=True)[:20]
importance, names = zip(*top_features)

plt.figure(figsize=(12, 6))
plt.bar(names, importance)
plt.title('Top 20 Most Important Features')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

joblib.dump(model, 'initial_model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')

print("\nInitial model and vectorizer saved.")
