
import tweepy
import time
import json
import os
import pandas as pd

BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAFGozgEAAAAAycqQqRLvMIXmUSTcGWss9%2FMOhh0%3DvDD16rOQvkFjGh9GJ6asS8AR9MjxdBEwPAiRpuCmdA47pF7q2J"



class TwitterListener(tweepy.StreamingClient):
    def __init__(self, bearer_token, fetched_tweets_filename, time_limit=180):
        super().__init__(bearer_token)
        self.start_time = time.time()
        self.limit = time_limit
        self.fetched_tweets_filename = fetched_tweets_filename

    def on_tweet(self, tweet):
        if (time.time() - self.start_time) < self.limit:
            print(tweet.text)
            with open(self.fetched_tweets_filename, "a") as tf:
                json.dump(tweet.data, tf)
                tf.write("\n")
        else:
            print("⏳ Time limit reached. Stopping stream.")
            self.disconnect()

if __name__ == '__main__':
    hash_tag_list = ["bullying", "hate speech", "violence"]
    
    fetched_tweets_filename = "./livedata/real_time_tweets.json"
    open(fetched_tweets_filename, 'w').close()

    listener = TwitterListener("AAAAAAAAAAAAAAAAAAAAAFGozgEAAAAAycqQqRLvMIXmUSTcGWss9%2FMOhh0%3DvDD16rOQvkFjGh9GJ6asS8AR9MjxdBEwPAiRpuCmdA47pF7q2J", fetched_tweets_filename)

    try:
        rules = listener.get_rules()
        if rules and rules.data:
            listener.delete_rules([rule.id for rule in rules.data])
        listener.add_rules(tweepy.StreamRule(" OR ".join(hash_tag_list)))
        print("✅ Rules added successfully!")
    except Exception as e:
        print(f"⚠️ Could not update rules: {e}")

    try:
        listener.filter(tweet_fields=["text"])
    except Exception as e:
        print(f"❌ Streaming failed: {e}")







