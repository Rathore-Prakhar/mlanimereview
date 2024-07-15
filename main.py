import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, learning_curve, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from collections import Counter
import time
import joblib

# Load the data
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

X_train = train_data['processed_review']
y_train = train_data['sentiment']
X_test = test_data['processed_review']
y_test = test_data['sentiment']

# Vectorize the text data
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Create and train the model
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Cross-validation
start_time = time.time()
cv_scores = cross_val_score(model, X_train_vectorized, y_train, cv=5)
end_time = time.time()

print("Cross-validation scores:", cv_scores)
print(f"Mean CV score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
print(f"Time taken for cross-validation: {end_time - start_time:.2f} seconds")

plt.figure(figsize=(8, 6))
plt.boxplot(cv_scores)
plt.title('Cross-validation Scores')
plt.ylabel('Accuracy')
plt.show()

# Learning curve
start_time = time.time()
train_sizes, train_scores, test_scores = learning_curve(
    model, X_train_vectorized, y_train, cv=5, 
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1)
end_time = time.time()

print(f"Time taken for learning curve generation: {end_time - start_time:.2f} seconds")

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(12, 6))
plt.plot(train_sizes, train_mean, label="Training score", color="r")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
plt.plot(train_sizes, test_mean, label="Cross-validation score", color="g")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy Score")
plt.title("Learning Curves")
plt.legend(loc="best")
plt.tight_layout()
plt.show()

gap = train_mean[-1] - test_mean[-1]
print(f"Gap between training and cross-validation scores: {gap:.2f}")

if gap > 0.1:
    print("The model might be overfitting.")
elif test_mean[-1] < 0.6:
    print("The model might be underfitting.")
else:
    print("The model seems to be fitting well.")

learning_curve_data = {
    'train_sizes': train_sizes,
    'train_mean': train_mean,
    'train_std': train_std,
    'test_mean': test_mean,
    'test_std': test_std
}
joblib.dump(learning_curve_data, 'learning_curve_data.joblib')

print("\nLearning curve data saved.")
