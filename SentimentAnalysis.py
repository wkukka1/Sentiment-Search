"""
Using the Roberta model This program should take a subject and the find the sentiment analyisis of it
"""
import snscrape.modules.twitter as sntwitter
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax


def SentimentAnalysis(query: str):
    """
    Takes what ever the query is and finds 1000 tweets relating to that string and uses the Roberta model to create a cs
    v that shows the positve, negative, and nuetral scores

    :param query:
    """
    tweets = []

    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        # Gets the 100 Tweets related to the users query
        if len(tweets) == 100:
            break
        elif tweet.lang == "en":
            tweets.append([tweet.user, tweet.date, tweet.content, tweet.likeCount, tweet.replyCount])
    df = pd.DataFrame(tweets, columns=['User', 'Date', 'Text', 'Likes', 'Replys'])

    roberta = "cardiffnlp/twitter-roberta-base-sentiment"
    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)

    # Creating three columns for the differnet sentiment scores of the tweet
    positive_score = []
    negative_score = []
    nuetral_score = []

    # sentiment analysis
    for tweet in tweets:
        encoded_tweet = tokenizer(tweet[2], return_tensors='pt')
        output = model(**encoded_tweet)

        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        positive_score.append(scores[2])
        negative_score.append(scores[0])
        nuetral_score.append(scores[1])

    # Adds the colums of postive negative and nutral sentiment scores
    df["Positive Score"] = positive_score
    df["Negative Score"] = negative_score
    df["Nuetral Score"] = nuetral_score
    return df


def getAvg(df):
    """
    Returns the average positive, negative and Nuetral scores from a CSV
    :param df:
    :return:
    """
    avg_pos, num, avg_neutral, avg_neg = 0, 0, 0, 0
    df = df.reset_index()
    for index, row in df.iterrows():
        avg_pos += row["Positive Score"]
        avg_neg += row["Negative Score"]
        avg_neutral += row["Nuetral Score"]
        num += 1
    avg_pos /= num
    avg_neg /= num
    avg_neutral /= num
    return [avg_pos, avg_neutral, avg_neg]
