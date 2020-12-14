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
import nlpaug.augmenter.word as naw

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


def Training(Data,model='XgBoost'):
    data=pd.read_csv(Data,encoding='cp1252')
    sentences = data.Body.values.tolist()
    labels = data.Label.values.tolist()
    aug=naw.RandomWordAug()
    sen_aug=aug.augment(sentences,n=len(sentences))
    aug2=naw.SynonymAug()
    sen_aug2=aug2.augment(sentences,n=len(sentences))
    sentences.extend(sen_aug4)
    sentences.extend(sen_aug5)
    labels1=labels.copy()
    labels.extend(labels1)
    labels.extend(labels1)
    sentences_test = sentences[-50:]
    labels_test = labels[-50:]
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
    joblib.dump(tf,"tfidf.sav")
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

    return score


def Processing_Test (Testing_Data,model='Bagging'):
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
    tf = joblib.load("tfidf.sav")
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
        for i in range(len(pred1)):
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
