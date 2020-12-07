import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import random
import numpy as np
import joblib


train=pd.read_csv('email.csv',encoding='cp1252')
# test=pd.read_csv("test.csv",encoding='cp1252')

for i in range(len(train.Body.values)):
  if train.iloc[i,4]=='Transfers':
    train.iloc[i,4]=2
  elif train.iloc[i,4]=='Retirements':
    train.iloc[i,4]=0
  else:
    train.iloc[i,4]=1

sentences = list(train.Body.values)
sentences = sentences[:140]
labels = list(train.Label.values)
labels = labels[:140]
test_sent=list(train.Body.values)
test_sent = test_sent[140:]
test_labels=list(train.Label.values)
test_labels = test_labels[140:]


for i in range(len(sentences)):
  para=train.iloc[i,3]
  para=nltk.sent_tokenize(para)
  if len(para)<=1:
    continue
  random.shuffle(para)
  para='. '.join(para)
  sentences.append(para)
  labels.append(train.iloc[i,4])

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

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
model = rfc.fit(X_train, labels)
# rfc_model="random_forest.pkl"
# model=joblib.load(rfc_model)

pred=model.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
matrix=confusion_matrix(test_labels,pred)
print(matrix)
score=accuracy_score(test_labels,pred)
print(score)
report=classification_report(test_labels,pred)
print(report)