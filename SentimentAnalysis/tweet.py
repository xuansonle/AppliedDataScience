##############################################################
# Import libraries
##############################################################
import tweepy
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")


############################################################
# Authentication
############################################################
consumerKey = os.environ.get("CONSUMER_KEY")
consumerSecret = os.environ.get("CONSUMER_SECRET")
accessToken = os.environ.get("ACCESS_TOKEN")
accessTokenSecret = os.environ.get("ACCESS_TOKEN_SECRET")
# Create auth object
authenticate = tweepy.OAuthHandler(consumerKey, consumerSecret)
# Set the access token & access token secret
authenticate.set_access_token(accessToken, accessTokenSecret)
# Create the API object & passing the auth information
api = tweepy.API(authenticate, wait_on_rate_limit=True)


############################################################
# Extract tweets from a twitter user and save them to a dataframe
############################################################
posts = api.user_timeline(screen_name="BillGates",
                          count=100, lang="en", tweet_mode="extended")
# Print the last 5 tweets from the account
print("Show the 5 recent tweets: ")
i = 1
for tweet in posts[0:5]:
    print(f"{i}) {tweet.full_text} \n")
    i += 1
# Save to a dataframe
df = pd.DataFrame({'Tweets': [tweet.full_text for tweet in posts]})


############################################################
# Data Cleaning
############################################################
def cleanText(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)  # remove mentions @abc
    text = re.sub(r'#', '', text)  # remove #
    text = re.sub(r'RT[\s]+', '', text)  # remove RT(ReTweet)
    text = re.sub(r'https?:\/\/\S', '', text)  # remove hyperlinks

    return text


df['Tweets'] = df['Tweets'].apply(cleanText)


############################################################
# Get the subjectivity (how subjective/opinionated)
# Get the polarity (how positive - negative)
############################################################
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity


def getPolarity(text):
    return TextBlob(text).sentiment.polarity


df["Subjectivity"] = df['Tweets'].apply(getSubjectivity)
df["Polarity"] = df['Tweets'].apply(getPolarity)


############################################################
# Plot Word Cloud
############################################################
allWords = ' '.join([tweet for tweet in df['Tweets']])
wordCloud = WordCloud(width=500, height=300, random_state=21,
                      max_font_size=119).generate(allWords)

plt.imshow(wordCloud, interpolation="bilinear")
plt.axis("off")
plt.show()


############################################################
# Compute the positive, neutral and negative analysis
############################################################
def getAnalysis(score):
    if score < 0:
        return "Negative"
    elif score == 0:
        return "Neutral"
    else:
        return "Positive"


df["Analysis"] = df
