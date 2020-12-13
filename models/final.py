import pandas as pd
import os
import numpy as np
import nltk, random
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost
import lightgbm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def Xgboost(data,labels):
    xgb = xgboost.XGBClassifier(max_depth=5,min_child_weight=5,gamma=0.3)
    model = xgb.fit(data,labels)
    filename = os.path.join(os.getcwd(),'test-uploads','model-output','xgboost.sav')
    joblib.dump(model, filename)
    return model

def Rfc(data,labels):
    rfc = RandomForestClassifier(max_depth=80, min_samples_split=2, n_estimators=1200, min_samples_leaf=1)
    model = rfc.fit(data,labels)
    filename = os.path.join(os.getcwd(),'test-uploads','model-output','rfc.sav')
    joblib.dump(model, filename)
    return model

def Lgb(data,labels) :
    lgb = lightgbm.LGBMClassifier(max_depth=5, num_leaves=40, min_child_samples=100, min_child_weight=0.1)
    model = lgb.fit(data,labels)
    filename = os.path.join(os.getcwd(),'test-uploads','model-output','lgb.sav')
    joblib.dump(model, filename)
    return model


def Training(Data,model):
    data=pd.read_csv(Data,encoding='cp1252')
    sentences = data.Body.values.tolist()
    labels = data.Label.values.tolist()
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
    elif model == 'LightGBM':
        Model = Lgb(X_train,labels)
    pred = Model.predict(X_test)

    matrix=confusion_matrix(labels_test,pred)
    score=accuracy_score(labels_test,pred)
    report=classification_report(labels_test,pred)

    return matrix,score,report

def Processing_Train(Data,model) :

    data=pd.read_csv(Data,encoding='cp1252')
    sentences = data.Body.values.tolist()
    labels = data.Label.values.tolist()
    data = pd.DataFrame({'Body':sentences,'Label':labels})

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
    elif model == 'LightGBM':
        Model = Lgb(X_train,labels)
    elif model == 'Bagging':
        Model1 = Xgboost(X_train,labels)
        Model2 = Rfc(X_train,labels)
        Model3 = Lgb(X_train,labels)
    return tf

def Processing_Test (Training_Data,Testing_Data,model='Bagging'):
    tf = Processing_Train(Training_Data, model)
    data = pd.read_csv(Testing_Data,encoding='cp1252')
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
        Model = joblib.load(os.path.join(os.getcwd(),'test-uploads','model-output','xgboost.sav'))
    elif model == 'RandomForestClassifier':
        Model = joblib.load(os.path.join(os.getcwd(),'test-uploads','model-output','rfc.sav'))
    elif  model == 'LightGBM':
        Model = joblib.load(os.path.join(os.getcwd(),'test-uploads','model-output','lgb.sav'))
    elif model == 'Bagging':
        Model1 = joblib.load(os.path.join(os.getcwd(),'test-uploads','model-output','xgboost.sav'))
        Model2 = joblib.load(os.path.join(os.getcwd(),'test-uploads','model-output','rfc.sav'))
        Model3 = joblib.load(os.path.join(os.getcwd(),'test-uploads','model-output','lgb.sav'))
    if model == 'Bagging':
        pred1 = Model1.predict(X_test)
        pred2 = Model2.predict(X_test)
        pred3 = Model2.predict(X_test)
        pred=[]
        for i in range(len(pred)):
            if pred1[i]==pred2[i] and pred1[i]==pred3[i]:
                pred.append(pred1[i])
            elif pred1[i]==pred2[i]:
                pred.append(pred1[i])
            elif pred2[i]==pred3[i]:
                pred.append(pred2[i])
            elif pred1[i]==pred3[i]:
                pred.append(pred1[i])
            else:
                pred.append(pred2[i])
    else:
        pred = Model.predict(X_test)
        
    pred = pd.DataFrame({'Label': pred})
    pred.to_csv(os.path.join(os.getcwd(),'test-uploads','model-output',"pred.csv"),index=False)
    transfers,retirements,mdu = 0,0,0
    for i in range(len(pred['Label'])):
        if pred.iloc[i,-1]=='Transfers':
            transfers+=1
        if pred.iloc[i,-1]=='Retirements':
            retirements+=1
        if pred.iloc[i,-1]=='MDU':
            mdu+=1
    return transfers,retirements,mdu