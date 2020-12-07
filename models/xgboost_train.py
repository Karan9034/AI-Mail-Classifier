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

df=pd.read_csv("email.csv",encoding='cp1252')

for i in range(157):
  if df.iloc[i,4]=='Transfers':
    df.iloc[i,4]=2
  elif df.iloc[i,4]=='Retirements':
    df.iloc[i,4]=0
  else:
    df.iloc[i,4]=1

sentences = list(df.Body.values)
labels = list(df.Label.values)
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
bodye=[]
for bb in sentences:
    bb=bb.lower()
    bb=bb.split()
    bb=[pc.stem(word) for word in bb if word not in set(stopwords.words('english'))]
    bb=' '.join(bb)
    bodye.append(bb)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(bodye,labels,random_state=42,test_size=0.1)
from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer()
X_train=tf.fit_transform(X_train).toarray()
X_test=tf.transform(X_test).toarray()

import xgboost
xgb=xgboost.XGBClassifier()
model=xgb.fit(X_train,Y_train)

xgboost_model="xgboost.pkl"
joblib.dump(model,xgboost_model)

pred=model.predict(X_test)

from sklearn.metrics import accuracy_score
cf=accuracy_score(Y_test,pred)
print(cf)