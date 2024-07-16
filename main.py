import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from textstat import flesch_kincaid_grade

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

def tfidf_similarity_ranking(data):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['processed_review'])
    
    cosine_sim = cosine_similarity(tfidf_matrix.mean(axis=0).reshape(1, -1), tfidf_matrix)
    
    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    top_shows = [(data['title'].iloc[i], score) for i, score in sim_scores[:10]]
    return top_shows

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

data['review_length'] = data['processed_review'].apply(lambda x: len(x.split()))

average_length = data.groupby('title')['review_length'].mean().reset_index()
top_10_length = average_length.sort_values('review_length', ascending=False).head(10)

print("Top 10 shows based on Review Length:")
print(top_10_length[['title', 'review_length']])

plt.figure(figsize=(12, 8))
plt.barh(top_10_length['title'], top_10_length['review_length'], color='purple')
plt.xlabel('Average Review Length (words)')
plt.title('Top 10 shows by Average Review Length')
plt.grid(axis='x')
plt.gca().invert_yaxis()
plt.show()

plt.figure(figsize=(10, 5))
plt.hist(data['review_length'], bins=20, color='orange', edgecolor='black')
plt.xlabel('Review Length (words)')
plt.ylabel('Number of Reviews')
plt.title('Distribution of Review Lengths')
plt.grid(axis='y')
plt.show()

category_sentiment = data.groupby('category')['vader_compound'].mean().reset_index()

print("Average VADER compound score by Category:")
print(category_sentiment)

plt.figure(figsize=(10, 5))
plt.bar(category_sentiment['category'], category_sentiment['vader_compound'], color='cyan')
plt.xlabel('Category')
plt.ylabel('Average VADER Compound Score')
plt.title('Average Sentiment Score by Category')
plt.grid(axis='y')
plt.show()

if 'date' in data.columns:
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)

    sentiment_over_time = data['vader_compound'].resample('M').mean()

    plt.figure(figsize=(12, 6))
    plt.plot(sentiment_over_time, color='red')
    plt.xlabel('Date')
    plt.ylabel('Average VADER Compound Score')
    plt.title('Sentiment Score Trend Over Time')
    plt.grid(True)
    plt.show()

top_5_positive = data.nlargest(5, 'vader_compound')[['title', 'review', 'vader_compound']]
top_5_negative = data.nsmallest(5, 'vader_compound')[['title', 'review', 'vader_compound']]

print("Top 5 Positive Reviews:")
print(top_5_positive)

print("Top 5 Negative Reviews:")
print(top_5_negative)

all_words = ' '.join(data['processed_review']).split()
word_freq = Counter(all_words)

most_common_words = word_freq.most_common(10)

print("Most Common Words in Reviews:")
print(most_common_words)

words, counts = zip(*most_common_words)

plt.figure(figsize=(10, 5))
plt.bar(words, counts, color='magenta')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Most Common Words in Reviews')
plt.show()

category_counts = data['category'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', colors=['lightblue', 'lightgreen', 'lightcoral'])
plt.title('Review Categories Distribution')
plt.show()


title_sentiment_distribution = data.groupby('title')['vader_compound'].describe().reset_index()
print("Sentiment Score Distribution by Title:")
print(title_sentiment_distribution)

plt.figure(figsize=(12, 8))
data.boxplot(column='vader_compound', by='title', grid=False, rot=90)
plt.xlabel('Title')
plt.ylabel('VADER Compound Score')
plt.title('Sentiment Scores Distribution by Title')
plt.suptitle('')
plt.show()


if 'date' in data.columns:
    reviews_per_year = data.resample('Y').size()

    plt.figure(figsize=(12, 6))
    plt.plot(reviews_per_year, marker='o', linestyle='-', color='blue')
    plt.xlabel('Year')
    plt.ylabel('Number of Reviews')
    plt.title('Number of Reviews per Year')
    plt.grid(True)
    plt.show()

if 'date' in data.columns:
    avg_sentiment_per_year = data['vader_compound'].resample('Y').mean()

    plt.figure(figsize=(12, 6))
    plt.plot(avg_sentiment_per_year, marker='o', linestyle='-', color='green')
    plt.xlabel('Year')
    plt.ylabel('Average VADER Compound Score')
    plt.title('Average Sentiment Score per Year')
    plt.grid(True)
    plt.show()

data['unique_word_count'] = data['processed_review'].apply(lambda x: len(set(x.split())))
average_unique_words = data.groupby('title')['unique_word_count'].mean().reset_index()

print("Average Unique Words per Review by Title:")
print(average_unique_words)

plt.figure(figsize=(12, 8))
plt.barh(average_unique_words['title'], average_unique_words['unique_word_count'], color='brown')
plt.xlabel('Average Unique Words per Review')
plt.title('Average Unique Words per Review by Title')
plt.grid(axis='x')
plt.gca().invert_yaxis()
plt.show()

tfidf_top_shows = tfidf_similarity_ranking(data)
print("Top 10 shows based on TF-IDF similarity:")
for show, score in tfidf_top_shows:
    print(f"{show}: {score:.4f}")

data['engagement_score'] = data['review_length'] * data['unique_word_count']
engagement_ranking = data.groupby('title')['engagement_score'].mean().sort_values(ascending=False)

print("\nTop 10 shows based on review engagement:")
print(engagement_ranking.head(10))

sentiment_consistency = data.groupby('title')['vader_compound'].std().sort_values()

print("\nTop 10 shows with most consistent sentiment:")
print(sentiment_consistency.head(10))

data['readability_score'] = data['review'].apply(lambda x: flesch_kincaid_grade(x))
readability_ranking = data.groupby('title')['readability_score'].mean().sort_values()

print("\nTop 10 shows with most readable reviews:")
print(readability_ranking.head(10))

def create_show_network(data):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data.groupby('title')['processed_review'].sum())
    
    cosine_sim = cosine_similarity(tfidf_matrix)
    np.fill_diagonal(cosine_sim, 0)
    
    G = nx.from_numpy_array(cosine_sim)
    G = nx.relabel_nodes(G, lambda x: data['title'].unique()[x])
    
    pagerank = nx.pagerank(G)
    return G, pagerank

G, pagerank = create_show_network(data)

print("\nTop 10 shows based on PageRank:")
top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
for show, score in top_pagerank:
    print(f"{show}: {score:.4f}")


plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=1000, node_color='lightblue', 
        font_size=8, font_weight='bold')
plt.title("Network of Show Similarities")
plt.axis('off')
plt.tight_layout()
plt.show()

def plot_sentiment_trends(data, top_n=5):
    top_shows = data.groupby('title')['vader_compound'].mean().nlargest(top_n).index
    
    plt.figure(figsize=(12, 6))
    for show in top_shows:
        show_data = data[data['title'] == show]
        show_data.set_index('date')['vader_compound'].resample('M').mean().plot(label=show)
    
    plt.xlabel('Date')
    plt.ylabel('Average Sentiment Score')
    plt.title(f'Sentiment Trends for Top {top_n} Shows')
    plt.legend()
    plt.grid(True)
    plt.show()

if 'date' in data.columns:
    plot_sentiment_trends(data)

if 'helpfulness' in data.columns:
    data['helpfulness_score'] = data['helpfulness'].str.split('/').str[0].astype(float) / \
                                data['helpfulness'].str.split('/').str[1].astype(float)
    
    helpfulness_ranking = data.groupby('title')['helpfulness_score'].mean().sort_values(ascending=False)
    
    print("\nTop 10 shows based on review helpfulness:")
    print(helpfulness_ranking.head(10))
    
    plt.figure(figsize=(10, 6))
    plt.scatter(data['vader_compound'], data['helpfulness_score'], alpha=0.5)
    plt.xlabel('Sentiment Score')
    plt.ylabel('Helpfulness Score')
    plt.title('Sentiment vs Helpfulness')
    plt.grid(True)
    plt.show()

if 'verified' in data.columns:
    verified_ratio = data.groupby('title')['verified'].mean().sort_values(ascending=False)
    
    print("\nTop 10 shows with highest ratio of verified reviews:")
    print(verified_ratio.head(10))

if 'date' in data.columns:
    data['month'] = data['date'].dt.month
    monthly_sentiment = data.groupby('month')['vader_compound'].mean()
    
    plt.figure(figsize=(10, 6))
    monthly_sentiment.plot(kind='bar')
    plt.title('Average Sentiment by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Sentiment Score')
    plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.grid(axis='y')
    plt.show()

data['composite_score'] = (
    data['vader_compound'].rank(pct=True) +
    data['engagement_score'].rank(pct=True) +
    data['readability_score'].rank(pct=True, ascending=False)
) / 3

composite_ranking = data.groupby('title')['composite_score'].mean().sort_values(ascending=False)

print("\nTop 10 shows based on composite score:")
print(composite_ranking.head(10))
