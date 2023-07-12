# Kevin Gerges, Kgerges@usc.edu
# ITP 216, Spring 2023
# Section: 31883R
# Final Project
# Description:
# Describe what this program does in your own words such as:
# A web application that uses machine learning and Pandas to visualize and estimate tweets sentiment. 

from flask import Flask, request, send_file
import pickle
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import sqlite3

app = Flask(__name__)
print("---------------------------------------------------------------------------")
print("-please be patient this whole process take about 3-5 mins to read the data-")
print("---------------------------------------------------------------------------")
print("--------------------please be patient reading the data---------------------")


# Create a database connection
conn = sqlite3.connect('twitter_data.db')
c = conn.cursor()

# Create a table to hold the Twitter data
c.execute('CREATE TABLE IF NOT EXISTS tweets (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT, category INTEGER)')

# Load data into the database
df = pd.read_csv("Twitter_Data.csv")
df = df[df["clean_text"].notnull()]
df['contains_tags_or_links'] = 0
for index, row in df.iterrows():
    tweet = str(row['clean_text'])
    if "#" in tweet or "https" in tweet or "@" in tweet:
        df.at[index, 'contains_tags_or_links'] = 1
    else:
        df.at[index, 'contains_tags_or_links'] = 0
df = df.sort_values(by='contains_tags_or_links', ascending=False)
df_tweet_length = df['tweet_length'] = df['clean_text'].str.len()
df.dropna(inplace=True)

print("---------------------------------------------------------------------------")
print("------------------------------done reading---------------------------------")

print("---------------------------------------------------------------------------")
print("----------------------------seting up database-----------------------------")
print("---------------------------------------------------------------------------")

for index, row in df.iterrows():
    c.execute('INSERT INTO tweets (text, category) VALUES (?, ?)', (row['clean_text'], row['category']))
conn.commit()
# Train the model

print("--please be patient as this will take about 1.5 mins to finish proccesing--")
print("---------------------------------------------------------------------------")
print("--please be patient as this will take about 1.5 mins to finish proccesing--")
print("---------------------------------------------------------------------------")

# Train the model
vectorizer = CountVectorizer(max_features=5000)
print("---------------------------------vectorized--------------------------------")
print("---------------------------------------------------------------------------")
X = vectorizer.fit_transform(df['clean_text'])
print("---------------------------cleaninig vectoriztion--------------------------")
print("---------------------------------------------------------------------------")
X = X.toarray()
print("--------------------------finihsed vectoriezed-----------------------------")
print("---------------------------------------------------------------------------")
y = df['category']
print("-------------------------about to start training---------------------------")

print("---------------------------------------------------------------------------")
print("------------------starting training, est is 1 min exactly------------------")
rfc = SGDClassifier()
rfc.fit(X, y)

# with open('model.pickle', 'wb') as f:
#     pickle.dump(rfc, f)

# # Load the trained model from disk
# with open('model.pickle', 'rb') as f:
#     rfc = pickle.load(f)
    
print("---------------------------------------------------------------------------")
print("------------------------------Models Accuracy------------------------------")
y_pred = rfc.predict(X)
accuracy = accuracy_score(y, y_pred)
print(f"Accuracy: {accuracy}")
print("---------------------------------------------------------------------------")

print("-------------------------------good to go----------------------------------")
print("---------------------------------------------------------------------------")

print("-----------------------------starting server-------------------------------")
print("---------------------------------------------------------------------------")

#home 
@app.route('/')
def home():
    return '''
        <h1> Hello and welcome to my Api to analyze the sentiment of tweets</h1> 
        <p><a href="/visualizations">Click here to view visualizations of sentiment analysis</a> </p>
        <p>  <a href="/analyze_tweet">Click here to analyze a single tweet</a></p>
        '''

def generate_visualizations():
    
    # Heatmap of sentiment vs. tweet length
    pivot = pd.pivot_table(df, values='clean_text', index='tweet_length', columns='category', aggfunc='count')
    sns.heatmap(pivot, cmap='coolwarm')
    plt.title('Sentiment vs. Tweet Length')
    plt.xlabel('Sentiment Category')
    plt.ylabel('Tweet Length')
    heatmap = plt.savefig('heatmap.png')
    plt.clf()

    # Word cloud of all tweets
    text = ' '.join(df['clean_text'].dropna().astype(str))
    wordcloud = WordCloud(width=800, height=800, background_color='white').generate(text)
    plt.figure(figsize=(8, 8))
    plt.imshow(wordcloud)
    plt.axis('off')
    wordcloud = plt.savefig('wordcloud.png')
    plt.clf()

    # Bar chart of sentiment category distribution
    sentiments = df['category'].value_counts()
    plt.bar(sentiments.index.astype(str), sentiments.values)
    plt.xlabel('Sentiment')
    plt.ylabel('Frequency')
    plt.title('Distribution of Sentiment Categories')
    bar_chart = plt.savefig('bar_chart.png')
    plt.clf()

    # Pie chart of sentiment category distribution
    sentiments = df['category'].value_counts()
    plt.pie(sentiments.values, labels=sentiments.index.astype(str), autopct='%1.1f%%')
    plt.title('Distribution of Sentiment Categories')
    pie_chart = plt.savefig('pie_chart.png')
    plt.clf()

    return {'heatmap': 'heatmap.png', 'wordcloud': 'wordcloud.png', 'bar_chart': 'bar_chart.png', 'pie_chart': 'pie_chart.png'}

@app.route('/analyze_tweet', methods=['POST'])
def analyze_tweet():
    tweet = request.form['tweet']
    tweet_vectorized = vectorizer.transform([tweet]).toarray()
    sentiment = rfc.predict(tweet_vectorized)[0]
    return f'The sentiment of the tweet "{tweet}" is {int(sentiment)}.'


@app.route('/analyze_tweet', methods=['GET'])
def analyze_tweet_form():
    return '''
        <form method="post" action="/analyze_tweet">
        <h1> 0 = netural |||||| 1 = Postive ||||| -1 = negative </h1>
            <label for="tweet">Enter or paste a tweet:</label><br>
            <textarea id="tweet" name="tweet" rows="4" cols="50"></textarea><br><br>
            <input type="submit" value="Submit">
        </form>
    '''

@app.route('/visualizations', methods=['GET'])
def visualizations_form():
    return '''
        <form method="post" action="/visualizations">
            <label for="visualization">Select a visualization:</label>
            <select id="visualization" name="visualization">
                <option value="heatmap">Sentiment vs. Tweet Length (heatmap)</option>
                <option value="wordcloud">Word Cloud</option>
                <option value="bar_chart">Distribution of Sentiment Categories (bar chart)</option>
                <option value="pie_chart">Distribution of Sentiment Categories (pie chart)</option>
            </select><br><br>
            <input type="submit" value="Submit">
        </form>
    '''

@app.route('/visualizations', methods=['POST'])
def visualizations():
    try:
        generate_visualizations()
        visualization = request.form['visualization']
        if visualization == 'heatmap':
            return send_file('heatmap.png', mimetype='image/png')
        elif visualization == 'wordcloud':
            return send_file('wordcloud.png', mimetype='image/png')
        elif visualization == 'bar_chart':
            return send_file('bar_chart.png', mimetype='image/png')
        elif visualization == 'pie_chart':
            return send_file('pie_chart.png', mimetype='image/png')
        else:
            return "Invalid visualization type"
    except Exception as e:
        return f"An error occurred: {str(e)}"
