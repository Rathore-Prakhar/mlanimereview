import pandas as pd
import re
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud

nltk.download('punkt')
nltk.download('stopwords')

data = pd.read_csv('reviews.csv')

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)

data['processed_review'] = data['review'].apply(preprocess_text)

def categorize_tags(tag):
    if 'Not Recommended' in tag:
        return 'Not Recommended'
    elif 'Recommended' in tag:
        return 'Recommended'
    elif 'Mixed Feelings' in tag:
        return 'Mixed Feelings'
    else:
        return 'Uncategorized'

data['category'] = data['tags'].apply(categorize_tags)

data = data[data['category'] != 'Uncategorized']

analyzer = SentimentIntensityAnalyzer()

data['vader_score'] = data['processed_review'].apply(lambda x: analyzer.polarity_scores(x))
data['vader_compound'] = data['vader_score'].apply(lambda score_dict: score_dict['compound'])
data['vader_negative'] = data['vader_score'].apply(lambda score_dict: score_dict['neg'])
data['vader_neutral'] = data['vader_score'].apply(lambda score_dict: score_dict['neu'])
data['vader_positive'] = data['vader_score'].apply(lambda score_dict: score_dict['pos'])

average_compound = data.groupby('title')['vader_compound'].mean().reset_index()

top_10 = average_compound.sort_values('vader_compound', ascending=False).head(10)

top_10_info = pd.merge(top_10, data[['title', 'link']].drop_duplicates(), on='title')

print("Top 10 shows based on Reviews:")
print(top_10_info[['title', 'vader_compound', 'link']])

plt.figure(figsize=(12, 8))
plt.barh(top_10_info['title'], top_10_info['vader_compound'], color='green')
plt.xlabel('Average VADER Compound Score')
plt.title('Top 10 shows by Average VADER Compound Scores')
plt.grid(axis='x')
plt.gca().invert_yaxis()
plt.show()

positive_reviews = ' '.join(data[data['vader_compound'] > 0.5]['processed_review'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_reviews)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Positive Reviews')
plt.show()

num_reviews = len(data)
num_titles = data['title'].nunique()
average_score = data['vader_compound'].mean()

print(f"Total number of reviews: {num_reviews}")
print(f"Total number of unique titles: {num_titles}")
print(f"Average VADER compound score: {average_score:.2f}")

plt.figure(figsize=(10, 5))
plt.hist(data['vader_compound'], bins=20, color='blue', edgecolor='black')
plt.xlabel('VADER Compound Score')
plt.ylabel('Number of Reviews')
plt.title('Distribution of VADER Compound Scores')
plt.grid(axis='y')
plt.show()

negative_reviews = ' '.join(data[data['vader_compound'] < -0.5]['processed_review'])
wordcloud_neg = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(negative_reviews)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_neg, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Negative Reviews')
plt.show()

neutral_reviews = ' '.join(data[(data['vader_compound'] >= -0.5) & (data['vader_compound'] <= 0.5)]['processed_review'])
wordcloud_neu = WordCloud(width=800, height=400, background_color='gray', colormap='Blues').generate(neutral_reviews)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_neu, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Neutral Reviews')
plt.show()
