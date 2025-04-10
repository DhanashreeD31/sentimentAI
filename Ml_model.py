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

def test_complaint(complaint_text):
    try:
        print("Loading ML model and vectorizer...")
        model = joblib.load('cb_sgd_final.sav')
        count_vect = pickle.load(open('count_vect.sav', 'rb'))
        
        if not complaint_text or len(complaint_text.strip()) == 0:
            print("Error: Empty complaint text")
            return {'error': 'Empty complaint text'}
            
        # Check if text contains at least 3 valid English words
        words = re.findall(r'\b[a-zA-Z]+\b', complaint_text.lower())
        if len(words) < 3:
            print("Error: Text must contain at least 3 valid words")
            return {'error': 'Text must contain at least 3 valid words'}
            
        print("Creating DataFrame with complaint text:", complaint_text)
        df = panda.DataFrame([complaint_text], columns=['text'])
        df['text length'] = df['text'].apply(len)
        
        stop_words = set(nltk.corpus.stopwords.words("english"))
        other_exclusions = {"#ff", "ff", "rt"}
        stop_words.update(other_exclusions)
        stemmer = PorterStemmer()
        
        def preprocess(tweet):
            try:
                print("Preprocessing tweet...")
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
                
                # Check if any valid words remain after preprocessing
                if tweet_lower.iloc[0].strip() == '':
                    print("Error: No valid text remains after preprocessing")
                    return None
                
                # Tokenizing
                tokenized_tweet = tweet_lower.apply(lambda x: x.split())
                
                # Removal of stopwords
                tokenized_tweet = tokenized_tweet.apply(lambda x: [item for item in x if item not in stop_words])
                
                # Check if any tokens remain after stopword removal
                if len(tokenized_tweet.iloc[0]) == 0:
                    print("Error: No valid words remain after stopword removal")
                    return None
                
                # Stemming
                tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])
                
                # Convert list back to string
                processed = tokenized_tweet.apply(lambda x: ' '.join(x))
                print("Preprocessing complete. Result:", processed.iloc[0])
                return processed
            except Exception as e:
                print(f"Error in preprocessing: {str(e)}")
                return None
        
        print("Starting preprocessing...")
        processed_tweet = preprocess(df['text'])
        
        if processed_tweet is None:
            print("Error: Preprocessing failed")
            return {'error': 'Text contains no valid words after preprocessing'}
            
        df['processed_tweets'] = processed_tweet
        
        if df['processed_tweets'].iloc[0].strip() == '':
            print("Error: Empty processed text")
            return {'error': 'Text contains no valid words after processing'}
        
        print("Vectorizing tweet...")
        testing_data = count_vect.transform(df['processed_tweets'])
        
        print("Making prediction...")
        try:
            prediction = model.predict(testing_data)[0]
            
            # Get raw decision function score
            decision_score = model.decision_function(testing_data)[0]
            
            # Convert to probability-like score (0 to 1) using sigmoid function
            # Clip the values to avoid overflow
            clipped_score = np.clip(decision_score, -100, 100)
            confidence = float(1 / (1 + np.exp(-clipped_score)))
            
            result = {
                'is_bullying': bool(prediction),
                'confidence': confidence,
                'processed_text': df['processed_tweets'].iloc[0],
                'raw_score': float(decision_score)  # Include raw score for debugging
            }
            print("Prediction complete:", result)
            return result
        except AttributeError as e:
            if 'predict_proba' in str(e):
                # Handle the case where predict_proba is not available
                prediction = model.predict(testing_data)[0]
                # Use a simple confidence of 1.0 for positive predictions, 0.0 for negative
                confidence = 1.0 if prediction else 0.0
                result = {
                    'is_bullying': bool(prediction),
                    'confidence': confidence,
                    'processed_text': df['processed_tweets'].iloc[0],
                    'note': 'Probability estimation not available, using binary prediction'
                }
                return result
            else:
                raise e
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in test_complaint: {error_details}")
        return {'error': str(e)}
