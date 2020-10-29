# -*- coding: utf-8 -*-
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import re

app = Flask(__name__)
lr = pickle.load(open('lr.pkl', 'rb'))
mulNb = pickle.load(open('MultinomialNB_topic.pkl', 'rb'))
nb = pickle.load(open('nb.pkl', 'rb'))

vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
topicVect = pickle.load(open('tfidf_topic.pkl', 'rb'))
dialectVect = pickle.load(open('tfidf.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html',variable=None, variables=None)

def cleanPunc(sentence):
    cleaned = re.sub(r'[?|؟|،|:|!|\'|"]',r'',sentence)
    cleaned = re.sub(r'[.|-|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    cleaned = re.sub(r'^هه+', '', cleaned)
    
    return cleaned


def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                           "]+", flags=re.UNICODE)
    string = emoji_pattern.sub(r'', string)
    
    return string


def clean_NonArabs(Twits):
    Twits = re.sub(r'[A-Za-z0-9]+', '', Twits)
    return Twits


def stopWords(text):
    stemSentence = ""
    for word in str(text).split():
        if word not in arb_stopwords:
            stemSentence += word
            stemSentence += " "
    
    return stemSentence


# +
arb_stopwords = nltk.corpus.stopwords.words('arabic')

c = pd.read_excel('stop.xlsx')
c = c['Unnamed: 1'].tolist() 

arb_stopwords.extend(c)
# -

from nltk.stem import ISRIStemmer
stemmer = ISRIStemmer()
def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence


@app.route('/predict',methods=['POST'])
def predict():
    tweet = request.form['text']
    result = list()
    result.append(dict())
    result[0]['tweet'] = tweet
    
    sentiment_tweet = stopWords(tweet)
    sentiment_tweet = stemming(sentiment_tweet)
    sentiment_tweet = cleanPunc(sentiment_tweet)
    sentiment_vectTweet = vectorizer.transform([sentiment_tweet])
    prediction = lr.predict(sentiment_vectTweet).toarray()[0]
    proba = lr.predict_proba(sentiment_vectTweet).toarray()[0]
    result[0]['sentiment'] = {}
    if prediction[0] == 1:
        result[0]['sentiment']["Negative"] = round(proba[0]*100, 2)
    if prediction[1] == 1:
        result[0]['sentiment']["Neutre"] = round(proba[1]*100, 2)
    if prediction[2] == 1:
        result[0]['sentiment']["Positive"] = round(proba[2]*100, 2)
    
    topic_tweet = remove_emoji(tweet)
    #topic_tweet = stopWords(tweet)
    #topic_tweet = stemming(topic_tweet)
    topic_tweet = cleanPunc(topic_tweet)
    #topic_tweet = clean_NonArabs(topic_tweet)
    topic_vectTweet = topicVect.transform([topic_tweet])
    topic_prediction = mulNb.predict(topic_vectTweet)
    topic_proba = mulNb.predict_proba(topic_vectTweet)[0]
    result[0]['topic'] = {}
    if topic_prediction == "autre":
        result[0]['topic']["autre"] = round(topic_proba[0]*100, 2)
    if topic_prediction == "politique":
        result[0]['topic']["politique"] = round(topic_proba[1]*100, 2)
    if topic_prediction == "sante":
        result[0]['topic']["sante"] = round(topic_proba[2]*100, 2)
    if topic_prediction == "social":
        result[0]['topic']["social"] = round(topic_proba[3]*100, 2)
    if topic_prediction == "sport":
        result[0]['topic']["sport"] = round(topic_proba[4]*100, 2)
    if topic_prediction == "économique":
        result[0]['topic']["économique"] = round(topic_proba[5]*100, 2)
        
    #dialect_tweet = remove_emoji(tweet)
    dialect_tweet = stopWords(tweet)
    dialect_tweet = stemming(dialect_tweet)
    dialect_tweet = cleanPunc(dialect_tweet)
    #dialect_tweet = clean_NonArabs(dialect_tweet)
    dialect_vectTweet = dialectVect.transform([dialect_tweet])
    dialect_prediction = nb.predict(dialect_vectTweet)
    dialect_proba = nb.predict_proba(dialect_vectTweet)[0]
    result[0]['dialect'] = {}
    if dialect_prediction == "Algerian":
        result[0]['dialect']["Algerian"] = round(dialect_proba[0]*100, 2)
    if dialect_prediction == "Lebanon":
        result[0]['dialect']["Lebanon"] = round(dialect_proba[1]*100, 2)
    if dialect_prediction == "Morocco":
        result[0]['dialect']["Morocco"] = round(dialect_proba[2]*100, 2)
    if dialect_prediction == "Tunisian":
        result[0]['dialect']["Tunisian"] = round(dialect_proba[3]*100, 2)
    if dialect_prediction == "egypt":
        result[0]['dialect']["egypt"] = round(dialect_proba[4]*100, 2)
    
    return render_template('index.html', variable=result, variables=None)


@app.route('/predictFile',methods=['POST'])
def predictFile():
    tweets = request.files['csvFile']  
    data = pd.read_excel(tweets)
    result = list()
    
    sentiment_tweets = pd.DataFrame()
    sentiment_tweets['Twits'] = data['Twits'].apply(stopWords)
    sentiment_tweets['Twits'] = sentiment_tweets['Twits'].apply(stemming)
    sentiment_tweets['Twits'] = sentiment_tweets['Twits'].apply(cleanPunc)
    vectTweets = vectorizer.transform(sentiment_tweets['Twits'])
    sentiment_predictions = lr.predict(vectTweets).toarray()
    sentiment_probas = lr.predict_proba(vectTweets).toarray()
    
    topic_tweets = pd.DataFrame()
    topic_tweets['Twits'] = data['Twits'].apply(remove_emoji)
    topic_tweets['Twits'] = topic_tweets['Twits'].apply(cleanPunc)
    topic_vectTweets = topicVect.transform(topic_tweets['Twits'])
    topic_predictions = mulNb.predict(topic_vectTweets)
    topic_probas = mulNb.predict_proba(topic_vectTweets)
    
    dialect_tweets = pd.DataFrame()
    #dialect_tweets['Twits'] = data['Twits'].apply(stopWords)
    #dialect_tweets['Twits'] = dialect_tweets['Twits'].apply(stemming)
    dialect_tweets['Twits'] = data['Twits'].apply(cleanPunc)
    dialect_vectTweets = dialectVect.transform(dialect_tweets['Twits'])
    dialect_predictions = nb.predict(dialect_vectTweets)
    dialect_probas = nb.predict_proba(dialect_vectTweets)
    
    
    for i, prediction in enumerate(sentiment_predictions):
        tmp = {}
        tmp['tweet'] = data.iloc[i][0]
        tmp['sentiment'] = {}
        tmp['topic'] = {}
        tmp['dialect'] = {}
        if prediction[0] == 1:
            tmp['sentiment']["Negative"] = round(sentiment_probas[i][0]*100, 2)
        if prediction[1] == 1:
            tmp['sentiment']["Neutre"] = round(sentiment_probas[i][1]*100, 2)
        if prediction[2] == 1:
            tmp['sentiment']["Positive"] = round(sentiment_probas[i][2]*100, 2)
            
        if topic_predictions[i] == "autre":
            tmp['topic']["autre"] = round (topic_probas[i][0]*100, 2)
        if topic_predictions[i] == "politique":
            tmp['topic']["politique"] = round (topic_probas[i][1]*100, 2)
        if topic_predictions[i] == "sante":
            tmp['topic']["sante"] = round (topic_probas[i][2]*100, 2)
        if topic_predictions[i] == "social":
            tmp['topic']["social"] = round (topic_probas[i][3]*100, 2)
        if topic_predictions[i] == "sport":
            tmp['topic']["sport"] = round (topic_probas[i][4]*100, 2)
        if topic_predictions[i] == "économique":
            tmp['topic']["économique"] = round (topic_probas[i][5]*100, 2)
            
        if dialect_predictions[i] == "Algerian":
            tmp['dialect']["Algerian"] = round(dialect_probas[i][0]*100, 2)
        if dialect_predictions[i] == "Lebanon":
            tmp['dialect']["Lebanon"] = round(dialect_probas[i][1]*100, 2)
        if dialect_predictions[i] == "Morocco":
            tmp['dialect']["Morocco"] = round(dialect_probas[i][2]*100, 2)
        if dialect_predictions[i] == "Tunisian":
            tmp['dialect']["Tunisian"] = round(dialect_probas[i][3]*100, 2)
        if dialect_predictions[i] == "egypt":
            tmp['dialect']["egypt"] = round(dialect_probas[i][4]*100, 2)
            
        result.append(tmp)
        
    return render_template('index.html',variable=None, variables=result)

if __name__ == "__main__":
    app.run(debug=False)

vectTweet = vectorizer.transform(["tweet"])
prediction = model.predict(vectTweet).toarray()[0]
proba = model.predict_proba(vectTweet).toarray()[0]
result = list()
result.append(dict())
result[0]['tweet'] = "de"
result[0]['sentiment'] = {}
if 1== 1:
    result[0]['sentiment']["Negative"] = round(proba[0]*100, 2)
if 1 == 1:
    result[0]['sentiment']["Neutre"] = round(proba[1]*100, 2)
if prediction[2] == 1:
    result[0]['sentiment']["Positive"] = round(proba[2]*100, 2)

teest = [{'tweet': 'de', 'sentiment': {'Negative': 0.0, 'Neutre': 0.98}}]
