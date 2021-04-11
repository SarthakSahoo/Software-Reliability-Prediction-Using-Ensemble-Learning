# Importing Libraries
import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# Importing the Dataset
dataset = pd.read_csv('Dataset/srp-class-csv.csv')
dataset['CLASS'] = (dataset['CLASS'] == False).astype(int)
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
np.seterr(divide='ignore', invalid='ignore')

# Generate Dataset Function
def generateDataset(length, X, Y):
    DiX = []
    DiY = []
    for _ in range(length):
        p = np.random.randint(1, length)
        DiX.append(X[p])
        DiY.append(Y[p])
    return np.array(DiX),np.array(DiY)

# Defining Logistic Regression Model and fitting our data
def Logistic(X_train, y_train):
    lg = LogisticRegression(random_state = None,solver='liblinear')
    lg.fit(X_train, y_train)
    return lg

# Defining NaiveBayes model and fitting our data
def NaiveBayes(X_train, y_train):
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    return nb

# Defining KNN and fitting our data
def Knn(X_train, y_train):
    knn = KNeighborsClassifier(n_neighbors = 2, metric = 'minkowski', p = 2)
    knn.fit(X_train, y_train)
    return knn

# Implementing Boosting Algorithm
def TrainBoost(k,func):
    M = [0 for i in range(0,k)]
    E = [0 for i in range(0,k)]
    w = [1/len(dataset) for i in range(0,len(dataset))]
    for i in range(0,k):
        plp=0
        while(True):
            dix = []
            diy = []
            output = []
            
            dix, diy = generateDataset(len(X),X,y)
            M[i] = func(dix,diy)
            pred = M[i].predict(dix)
            output = diy
            E[i]=0
            for j in range(0, len(dix)):
                if(pred[j] != output[j]):
                    E[i] = E[i] + w[j]
            E[i]=E[i]/sum(w)
            plp=plp+1
            if(E[i] < 0.15):
                break
        #print('=> i=',i,' plp=',plp,' E[i]=',E[i])
        for j in range(0, len(dix)):
            if(pred[j] == output[j]):
                w[j] = w[j] * (E[i] / (1 - E[i]))
        w = np.array(w)
        w = (w-w.min())/(w.max()-w.min())
    return M,E
# To use ensemble to classify Tuple
def classify_model(E,k,M,x):
    w=[0,0]
    for i in range(0,k):
        wt = math.log((1-E[i])/E[i])
        c = M[i].predict(x.reshape(1,-1))
        c = c[0]
        w[c] = w[c] + wt
    return w.index(max(w))

# Using Adaboost for better classification
def predict(M,E,k,x_test):
    Y = []
    for i in x_test:
        Y.append(classify_model(E,k,M,i))
    cm = confusion_matrix(y, Y)
    accuracy = (cm[0,0]+cm[1,1])/len(X)*100
    return cm,accuracy

# Predicting the datapoints based on Logistic, NaiveBayes, KNN model
def pred(M,k):
    acc = []
    for i in range(0,k):
        Y = M[i].predict(X)
        cm = confusion_matrix(y, Y)
        acc.append((cm[0,0]+cm[1,1])/len(X)*100)
    return acc

if __name__ == "__main__":
    func = [Logistic,NaiveBayes,Knn]
    k = 9
    # For different value of k starting from 9 to 15 checking how the model performs by using Ensemble method and Adaboost. The increment of k is done by 2
    # K is the number of round one classifier generated per round
    while(k<=15):
        print('\nk = ',k)
        for i in range(len(func)):
            M,E = TrainBoost(k,func[i])
            print('Ind Model',max(pred(M,k)))
            print('AdaBoost',predict(M,E,k,X))
        k=k+2
