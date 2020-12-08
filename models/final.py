import pandas as pd
import numpy as np
import nltk, random
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

def Xgboost(data,labels):
    xgb = xgboost.XGBClassifier()
    model = xgb.fit(data,labels)
    filename = '../test-uploads/model-output/xgboost.sav'
    joblib.dump(model, filename)
    return model

def Rfc(data,labels):
    rfc = RandomForestClassifier()
    model = rfc.fit(data,labels)
    filename = '../test-uploads/model-output/rfc.sav'
    joblib.dump(model, filename)
    return model

def Training(Data,model):
    data=pd.read_csv(Data,encoding='cp1252')
    data["Body"]=data["Subject"]+'. '+data["Body"]
    l = data.Body.values.tolist()
    mdu_index = [i for i in range(len(l)) if data.iloc[i,4] == 'MDU' ]
    tra_index = [i for i in range(len(l)) if data.iloc[i,4] == 'Transfers' ]
    ret_index = [i for i in range(len(l)) if data.iloc[i,4] == 'Retirements' ]
    newlist=[]
    l1=[]
    labels_new = []
    for i in range(int(mdu_index[0]),int(mdu_index[-1])):
        words=nltk.word_tokenize(data.iloc[i,3])
        if len(words)>=150:
            sent=nltk.sent_tokenize(data.iloc[i,3])
            newlist.append('. '.join(sent[:int(len(sent)/2)]))
            newlist.append(data.iloc[i,0]+'. '.join(sent[int(len(sent)/2):]))
            l1.append(i)
            labels_new.append('MDU')
            labels_new.append('MDU')
    for i in range(int(tra_index[0]),int(tra_index[-1])):
        words=nltk.word_tokenize(data.iloc[i,3])
        if len(words)>=150:
            sent=nltk.sent_tokenize(data.iloc[i,3])
            newlist.append('. '.join(sent[:int(len(sent)/2)]))
            newlist.append(data.iloc[i,0]+'. '.join(sent[int(len(sent)/2):]))
            l1.append(i)
            labels_new.append('Transfers')
            labels_new.append('Transfers')
    for i in range(int(ret_index[0]),int(ret_index[-1])):
        words=nltk.word_tokenize(data.iloc[i,3])
        if len(words)>=150:
            sent=nltk.sent_tokenize(data.iloc[i,3])
            newlist.append('. '.join(sent[:int(len(sent)/2)]))
            newlist.append(data.iloc[i,0]+'. '.join(sent[int(len(sent)/2):]))
            l1.append(i)
            labels_new.append('Retirements')
            labels_new.append('Retirements')
    data=data.drop(l1,axis=0)
    data.drop(["Subject","Date","Sender"],axis =1)
    sentences=list(data.Body.values)
    sentences.extend(newlist)
    labels=list(data.Label.values)
    labels.extend(labels_new)
    data = pd.DataFrame({'Body':sentences,'Label':labels})
    ret_test = data.iloc[20:30,:]
    tra_test = data.iloc[90:100,:]
    mdu_test = data.iloc[180:191,:]
    test_data = pd.concat((ret_test,tra_test,mdu_test),axis=0)
    test_data=test_data.reset_index()
    data = data.drop([20,21,22,23,24,25,26,27,28,29],axis=0)
    data = data.drop([90,91,92,93,94,95,96,97,98,99],axis=0)
    data = data.drop([180,181,182,183,184,185,186,187,188,189,190],axis=0)
    data = data.reset_index()
    test_data.reset_index()
    test_data = test_data.drop('index',axis=1)
    data = data.drop('index',axis=1)
    sentences = data.Body.values.tolist()
    labels = data.Label.values.tolist()
    sentences_test = test_data.Body.values.tolist()
    labels_test= test_data.Label.values.tolist()

    for i in range(len(sentences)):
      para=data.iloc[i,0]
      para=nltk.sent_tokenize(para)
      if len(para)<=1:
        continue
      random.shuffle(para)
      para='. '.join(para)
      sentences.append(para)
      labels.append(data.iloc[i,1])
    body_train=[]
    pc=PorterStemmer()
    for bb in sentences:
        bb=bb.lower()
        bb=bb.split()
        bb=[pc.stem(word) for word in bb if word not in set(stopwords.words('english'))]
        bb=' '.join(bb)
        body_train.append(bb)
    body_test = []
    for bb in sentences_test:
        bb=bb.lower()
        bb=bb.split()
        bb=[pc.stem(word) for word in bb if word not in set(stopwords.words('english'))]
        bb=' '.join(bb)
        body_test.append(bb)
    tf = TfidfVectorizer()
    X_train = tf.fit_transform(body_train).toarray()
    X_test = tf.transform(body_test).toarray()
    if model == 'XgBoost':
        Model = Xgboost(X_train,labels)
    elif model == 'RandomForestClassifier':
        Model = Rfc(X_train,labels)
    pred = Model.predict(X_test)

    matrix=confusion_matrix(labels_test,pred)
    score=accuracy_score(labels_test,pred)
    report=classification_report(labels_test,pred)
    return matrix,score,report

def Processing_Train(Data,model) :
    data=pd.read_csv(Data,encoding='cp1252')
    data["Body"]=data["Subject"]+'. '+data["Body"]
    l = data.Body.values.tolist()
    mdu_index = [i for i in range(len(l)) if data.iloc[i,4] == 'MDU' ]
    tra_index = [i for i in range(len(l)) if data.iloc[i,4] == 'Transfers' ]
    ret_index = [i for i in range(len(l)) if data.iloc[i,4] == 'Retirements' ]
    newlist=[]
    l1=[]
    labels_new = []
    for i in range(int(mdu_index[0]),int(mdu_index[-1])):
        words=nltk.word_tokenize(data.iloc[i,3])
        if len(words)>=150:
            sent=nltk.sent_tokenize(data.iloc[i,3])
            newlist.append('. '.join(sent[:int(len(sent)/2)]))
            newlist.append(data.iloc[i,0]+'. '.join(sent[int(len(sent)/2):]))
            l1.append(i)
            labels_new.append('MDU')
            labels_new.append('MDU')
    for i in range(int(tra_index[0]),int(tra_index[-1])):
        words=nltk.word_tokenize(data.iloc[i,3])
        if len(words)>=150:
            sent=nltk.sent_tokenize(data.iloc[i,3])
            newlist.append('. '.join(sent[:int(len(sent)/2)]))
            newlist.append(data.iloc[i,0]+'. '.join(sent[int(len(sent)/2):]))
            l1.append(i)
            labels_new.append('Transfers')
            labels_new.append('Transfers')
    for i in range(int(ret_index[0]),int(ret_index[-1])):
        words=nltk.word_tokenize(data.iloc[i,3])
        if len(words)>=150:
            sent=nltk.sent_tokenize(data.iloc[i,3])
            newlist.append('. '.join(sent[:int(len(sent)/2)]))
            newlist.append(data.iloc[i,0]+'. '.join(sent[int(len(sent)/2):]))
            l1.append(i)
            labels_new.append('Retirements')
            labels_new.append('Retirements')
    data=data.drop(l1,axis=0)
    data.drop(["Subject","Date","Sender"],axis =1)
    sentences=list(data.Body.values)
    sentences.extend(newlist)
    labels=list(data.Label.values)
    labels.extend(labels_new)
    data = pd.DataFrame({'Body':sentences,'Label':labels})
    sentences = data.Body.values.tolist()
    labels = data.Label.values.tolist()

    for i in range(len(sentences)):
      para=data.iloc[i,0]
      para=nltk.sent_tokenize(para)
      if len(para)<=1:
        continue
      random.shuffle(para)
      para='. '.join(para)
      sentences.append(para)
      labels.append(data.iloc[i,1])
    body_train=[]
    pc=PorterStemmer()
    for bb in sentences:
        bb=bb.lower()
        bb=bb.split()
        bb=[pc.stem(word) for word in bb if word not in set(stopwords.words('english'))]
        bb=' '.join(bb)
        body_train.append(bb)
    tf = TfidfVectorizer()
    X_train = tf.fit_transform(body_train).toarray()
    if model == 'XgBoost':
        Model = Xgboost(X_train,labels)
    elif model == 'RandomForestClassifier':
        Model = Rfc(X_train,labels)
    return tf

def Processing_Test (Training_Data,Testing_Data,model='XgBoost'):
    tf = Processing_Train(Training_Data, model)
    data = pd.read_csv(Testing_Data,encoding='cp1252')
    data["Body"]=data["Subject"]+'. '+data["Body"]
    sentences=list(data.Body.values)
    body_test=[]
    pc=PorterStemmer()
    for bb in sentences:
        bb=bb.lower()
        bb=bb.split()
        bb=[pc.stem(word) for word in bb if word not in set(stopwords.words('english'))]
        bb=' '.join(bb)
        body_test.append(bb)
    X_test = tf.transform(body_test)
    if model == 'XgBoost':
        Model = joblib.load('../test-uploads/model-output/xgboost.sav')
    elif model == 'RandomForestClassifier':
        Model = joblib.load('../test-uploads/model-output/rfc.sav')
    pred = Model.predict(X_test)
    df = pd.DataFrame({'Label': pred})
    df.to_csv("../test-uploads/model-output/Predictions.csv",index=False)