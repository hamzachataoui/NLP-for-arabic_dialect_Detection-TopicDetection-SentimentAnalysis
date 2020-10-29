# -*- coding: utf-8 -*-
# +
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import nltk
import re
import csv
import os, glob
from sklearn.metrics import accuracy_score , f1_score, auc

def cleanPunc(sentence):
    cleaned = re.sub(r'[?|؟|،|:|!|\'|"]',r'',sentence)
    cleaned = re.sub(r'[.|-|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    cleaned = re.sub(r'^هه+', '', cleaned)
    
    return cleaned

import nltk
nltk.download('stopwords')

arb_stopwords = nltk.corpus.stopwords.words('arabic')

c = pd.read_excel('stop.xlsx')
c = c['Unnamed: 1'].tolist() 

arb_stopwords.extend(c)

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
tagedData = pd.read_excel('Sentiments_Analysis_labledData.xlsx')
del tagedData['Unnamed: 0']

labeledData = pd.read_excel('Sentiments_Analysis.xlsx')
del labeledData['Unnamed: 0']

train = pd.concat([labeledData,tagedData])

train.dropna(inplace=True)
print(train.shape)

def stopWords(text):
    stemSentence = ""
    for word in str(text).split():
        if word not in arb_stopwords:
            stemSentence += word
            stemSentence += " "
    
    return stemSentence

train['Text'] = train['Text'].apply(stopWords)

train["Text"] = train["Text"].apply(stemming)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1,2), preprocessor = cleanPunc)

from sklearn.model_selection import train_test_split
trainData, testData = train_test_split(train, test_size=0.3)
print(trainData.shape)

training_features = vectorizer.fit_transform(trainData["Text"])
test_features = vectorizer.transform(testData["Text"])
print(training_features.shape)

# !pip install scikit-multilearn

from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.metrics import classification_report
from skmultilearn.problem_transform import LabelPowerset
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier

from sklearn import feature_selection
y = ['Negative','Neutral','Positive']
X_names = vectorizer.get_feature_names()
p_value_limit = 0.05
dtf_features = pd.DataFrame()
for cat in y:
    chi2, p = feature_selection.chi2(training_features, trainData[cat])
    dtf_features = dtf_features.append(pd.DataFrame({"feature":X_names, "score":p, "y":cat}))
    dtf_features = dtf_features.sort_values(["score"], ascending=True)
    dtf_features = dtf_features[dtf_features["score"]<p_value_limit]
X_names = dtf_features["feature"].unique().tolist()

new_vectorizer = TfidfVectorizer(vocabulary=X_names)
new_training_features = new_vectorizer.fit_transform(trainData["Text"])
new_test_features = new_vectorizer.transform(testData["Text"])
new_training_features.shape

# +
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import nltk
import re
import csv
import os, glob
from sklearn.metrics import accuracy_score , f1_score, auc
tagedData = pd.read_excel('Sentiments_Analysis_labledData.xlsx')
del tagedData['Unnamed: 0']

labeledData = pd.read_excel('Sentiments_Analysis.xlsx')
del labeledData['Unnamed: 0']

train = pd.concat([labeledData,tagedData])

train.dropna(inplace=True)
print(train.shape)

# +
import nltk
nltk.download('stopwords')

arb_stopwords = nltk.corpus.stopwords.words('arabic')

c = pd.read_excel('stop.xlsx')
c = c['Unnamed: 1'].tolist() 

arb_stopwords.extend(c)
def stopWords(text):
    stemSentence = ""
    for word in str(text).split():
        if word not in arb_stopwords:
            stemSentence += word
            stemSentence += " "
    
    return stemSentence


# -

def cleanPunc(sentence):
    cleaned = re.sub(r'[?|؟|،|:|!|\'|"]',r'',sentence)
    cleaned = re.sub(r'[.|-|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.replace("\n"," ")
    cleaned = re.sub(r'^هه+', '', cleaned)
    cleaned = cleaned.strip()
    
    return cleaned


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


def clean_NonArabs(Twits):
    Twits = re.sub(r'[A-Za-z0-9]+', '', Twits)
    return Twits


train['Text'].apply(clean_NonArabs)

import matplotlib.pyplot as plt
plt.bar(['Algerian', 'Lebanon', 'egypt', 'Tunisian', 'Morocco'] , [312, 40, 38, 15, 15])
plt.xlabel('Dialects')
plt.ylabel('Twits')
plt.title('Twits by Arab dialect')

lr = LabelPowerset(LogisticRegression(verbose= 1,n_jobs=-1, solver= 'lbfgs', penalty= 'l2', C= 10,multi_class='multinomial', max_iter=1000))
lr.fit(new_training_features, trainData[['Negative','Neutral','Positive']])

# Saving model to disk
pickle.dump(lr, open('lr.pkl','wb'))

cc = lr.predict(new_vectorizer.transform(['tweet']))
print('accuracy %s' % accuracy_score([[0,1,0]], cc))
print(classification_report([[0,1,0]], cc))

pd.DataFrame(X_names).to_csv('vocabulary.csv')

pickle.dump(new_vectorizer, open('vectorizer.pkl','wb'))


