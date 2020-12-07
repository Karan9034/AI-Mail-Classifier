import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import random
import numpy as np
import joblib

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

train=pd.read_csv("email.csv",encoding='cp1252')
test=pd.read_csv("test.csv",encoding='cp1252')

for i in range(len(train.Body.values)):
  if df.iloc[i,4]=='Transfers':
    df.iloc[i,4]=2
  elif df.iloc[i,4]=='Retirements':
    df.iloc[i,4]=0
  else:
    df.iloc[i,4]=1

sentences = list(train.Body.values)
labels = list(train.Label.values)
test_sent=list(test.Body.values)
for i in range(len(sentences)):
  para=df.iloc[i,3]
  para=nltk.sent_tokenize(para)
  if len(para)<=1:
    continue
  random.shuffle(para)
  para='. '.join(para)
  sentences.append(para)
  labels.append(df.iloc[i,4])

pc=PorterStemmer()
body_train=[]
for bb in sentences:
    bb=bb.lower()
    bb=bb.split()
    bb=[pc.stem(word) for word in bb if word not in set(stopwords.words('english'))]
    bb=' '.join(bb)
    body_train.append(bb)

body_test=[]
for bb in test_sent:
    bb=bb.lower()
    bb=bb.split()
    bb=[pc.stem(word) for word in bb if word not in set(stopwords.words('english'))]
    bb=' '.join(bb)
    body_test.append(bb)

from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer()
X_train=tf.fit_transform(body_train).toarray()
X_test=tf.transform(body_test).toarray()

import xgboost

xgb_model="xgboost.pkl"
model=joblib.load(xgb_model)

pred=model.predict(body_test)