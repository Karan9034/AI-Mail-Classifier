import pandas as pd
import os
import numpy as np
import nltk, random
from nltk.corpus import stopwords,wordnet
from nltk.stem import PorterStemmer
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost
import lightgbm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import nlpaug.augmenter.word as naw
from simpletransformers.classification import ClassificationModel


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
def get_synonyms(word):
    """
    Get synonyms of a word
    """
    synonyms = set()

    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)

    if word in synonyms:
        synonyms.remove(word)

    return list(synonyms)

def synonym_replacement(words, n):

    words = words.split()

    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in set(stopwords.words('english'))]))
    random.shuffle(random_word_list)
    num_replaced = 0

    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)

        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1

        if num_replaced >= n: #only replace up to n words
            break

    sentence = ' '.join(new_words)

    return sentence

def swap_word(new_words):

    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0

    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1

        if counter > 3:
            return new_words

    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words

def random_swap(words, n):

    words = words.split()
    new_words = words.copy()

    for _ in range(n):
        new_words = swap_word(new_words)
    if len(new_words) == 1:
        return new_words[0]
    else:
        sentence = ' '.join(new_words)


    return sentence

def add_word(new_words):

    synonyms = []
    counter = 0

    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words)-1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return

    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words)-1)
    new_words.insert(random_idx, random_synonym)

def random_insertion(words, n):

    words = words.split()
    new_words = words.copy()

    for _ in range(n):
        add_word(new_words)

    sentence = ' '.join(new_words)
    return sentence

def Xgboost(data,labels):
    xgb = xgboost.XGBClassifier(max_depth=5,min_child_weight=5,gamma=0.3)
    model = xgb.fit(data,labels)
    filename = os.path.join(os.getcwd(),'test-uploads','model-output','xgboost.sav')
    joblib.dump(model, filename)
    return model

def Rfc(data,labels):
    rfc = RandomForestClassifier(max_depth=20, min_samples_split=12, n_estimators=1200, min_samples_leaf=5)
    model = rfc.fit(data,labels)
    filename = os.path.join(os.getcwd(),'test-uploads','model-output','rfc.sav')
    joblib.dump(model, filename)
    return model

def Lgb(data,labels) :
    lgb = lightgbm.LGBMClassifier(reg_alpha=7, reg_lambda=5, subsample=0.7, colsample_bytree=0.4, num_leaves=50, min_child_samples=100, min_child_weight=0.1)
    model = lgb.fit(data,labels)
    filename = os.path.join(os.getcwd(),'test-uploads','model-output','lgb.sav')
    joblib.dump(model, filename)
    return model

def Distilbert(data,labels):
    df = pd.DataFrame({'Body':data,'Label':labels})
    for i in range(len(df.Body.values)):
        if df.iloc[i,1] == 'Retirements':
            df.iloc[i,1] = 0
        elif df.iloc[i,1] == 'MDU':
            df.iloc[i,1] = 1
        elif df.iloc[i,1] == 'Transfers':
            df.iloc[i,1] = 2
        elif df.iloc[i,1] == 'Death':
            df.iloc[i,1] = 3
    model=ClassificationModel('distilbert','distilbert-base-cased',num_labels=4,use_cuda=False,args={'learning_rate':1e-5, 'num_train_epochs': 6,
    'reprocess_input_data': True, 'overwrite_output_dir': True, "best_model_dir": "outputs/"})
    model.train_model(df)
    return model

def Training(Data,model='Bagging'):
    data=pd.read_csv(Data,encoding='cp1252')
    sentences = data.Body.values.tolist()
    labels = data.Label.values.tolist()
    sen = []
    lab = []
    for i in range(0,5):
        for j in range(len(sentences)):
            words = nltk.word_tokenize(sentences[j])
            n = int(0.05*len(words))
            sen.append(synonym_replacement(sentences[j],n))
            sen.append(random_swap(sentences[j],n))
            sen.append(random_insertion(sentences[j],n))
            lab.append(labels[j])
            lab.append(labels[j])
            lab.append(labels[j])
    sentences.extend(sen)
    labels.extend(lab)
    sentences_test = sentences[-50:]
    labels_test = labels[-50:]
    if model == 'DistilBERT':
        Model = Distilbert(sentences,labels)
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
    joblib.dump(tf,os.path.join(os.getcwd(),'test-uploads','model-output','tfidf.sav'))
    X_test = tf.transform(body_test).toarray()
    if model == 'XgBoost':
        Model = Xgboost(X_train,labels)
    elif model == 'RandomForestClassifier':
        Model = Rfc(X_train,labels)
    elif model == 'LightGBM':
        Model = Lgb(X_train,labels)
    elif model == 'Bagging':
        Model1 = Xgboost(X_train, labels)
        Model2 = Rfc(X_train, labels)
        Model3 = Lgb(X_train, labels)
    if model == 'Bagging':
        pred1 = Model1.predict(X_test)
        pred2 = Model2.predict(X_test)
        pred3 = Model3.predict(X_test)
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

    matrix=confusion_matrix(labels_test,pred)
    score=accuracy_score(labels_test,pred)
    report=classification_report(labels_test,pred)

    return score


def Processing_Test (Testing_Data,model='Bagging'):
    data = pd.read_csv(Testing_Data,encoding='cp1252')
    sentences=list(data.Body.values)
    body_test=[]
    x_test = sentences.copy()
    pc=PorterStemmer()
    for bb in sentences:
        bb=bb.lower()
        bb=bb.split()
        bb=[pc.stem(word) for word in bb if word not in set(stopwords.words('english'))]
        bb=' '.join(bb)
        body_test.append(bb)
    tf = joblib.load(os.path.join(os.getcwd(),'test-uploads','model-output','tfidf.sav'))
    X_test = tf.transform(body_test).toarray()
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
    elif model == 'DistilBERT':
        Model = ClassificationModel('distilbert',os.path.join(os.getcwd(),'models','outputs'))

    if model == 'Bagging':
        pred1 = Model1.predict(X_test)
        pred2 = Model2.predict(X_test)
        pred3 = Model3.predict(X_test)
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
                pred.append(pred3[i])
    elif model == 'DistilBERT':
        pred1,_ = Model.predict(x_test)
    else:
        pred = Model.predict(X_test)
    if model == 'DistilBERT':
        pred = []
        for i in pred1:
            if i==2:
                i = 'Transfers'
            elif i==0:
                i = 'Retirements'
            elif i==1:
                i = 'MDU'
            elif i==3:
                i = 'Death'
            pred.append(i)
    pred = pd.DataFrame({'Label': pred})
    pred.to_csv(os.path.join(os.getcwd(),'test-uploads','model-output',"pred.csv"),index=False)
    transfers,retirements,mdu,death = 0,0,0,0
    for i in range(len(pred['Label'])):
        if pred.iloc[i,-1]=='Transfers':
            transfers+=1
        if pred.iloc[i,-1]=='Retirements':
            retirements+=1
        if pred.iloc[i,-1]=='MDU':
            mdu+=1
        if pred.iloc[i,-1]=='Death':
            death+=1
    return transfers,retirements,mdu,death
