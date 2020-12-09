from sklearn.metrics import accuracy_score
def accuracy(pred,labels):
    cf = accuracy_score(pred,labels)
    return cf
