from sklearn import *
import numpy as np 
import pandas as panda
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import joblib


def predt():
    model = joblib.load('cb_sgd_final.sav')  # Path of classifier
    df = panda.read_csv("./livedata/real_time_tweets.csv")  # Path of real-time tweets
    count_vect = pickle.load(open('count_vect.sav', 'rb'))  # Path of count vectorizer

    df['text length'] = df['text'].apply(len)
    tweet = df.text
    stop_words = set(nltk.corpus.stopwords.words("english"))  # Use set for efficiency
    other_exclusions = {"#ff", "ff", "rt"}
    stop_words.update(other_exclusions)
    stemmer = PorterStemmer()

    def preprocess(tweet):  
        # Removal of extra spaces
        tweet_space = tweet.str.replace(r'\s+', ' ', regex=True)

        # Removal of @mentions
        tweet_name = tweet_space.str.replace(r'@[\w\-]+', '', regex=True)

        # Removal of links (URLs)
        giant_url_regex = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                                     r'[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        tweets = tweet_name.str.replace(giant_url_regex, '', regex=True)

        # Removal of punctuations and numbers
        punc_remove = tweets.str.replace(r"[^a-zA-Z]", " ", regex=True)

        # Remove whitespace with a single space
        newtweet = punc_remove.str.replace(r'\s+', ' ', regex=True)

        # Remove leading and trailing whitespace
        newtweet = newtweet.str.replace(r'^\s+|\s+?$', '', regex=True)

        # Replace normal numbers with "numbr"
        newtweet = newtweet.str.replace(r'\d+(\.\d+)?', 'numbr', regex=True)

        # Convert to lowercase
        tweet_lower = newtweet.str.lower()

        # Tokenizing
        tokenized_tweet = tweet_lower.apply(lambda x: x.split())

        # Removal of stopwords
        tokenized_tweet = tokenized_tweet.apply(lambda x: [item for item in x if item not in stop_words])

        # Stemming
        tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) 

        # Convert list back to string
        return tokenized_tweet.apply(lambda x: ' '.join(x))

    processed_tweets = preprocess(tweet)
    df['processed_tweets'] = processed_tweets

    # Vectorizing the tweets
    testing_data = count_vect.transform(df['processed_tweets'])

    # Predicting the tweets
    y_preds = model.predict(testing_data)
    dframe = panda.DataFrame()
    dframe['tweets'] = df['text']
    dframe['class'] = y_preds

    # Save predicted tweets to CSV
    dframe.to_csv('./livedata/classified_tweets.csv', index=False)

    # Count the number of offensive tweets (class=1)
    return (dframe['class'] == 1).sum()
