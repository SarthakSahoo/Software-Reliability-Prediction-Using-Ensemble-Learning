# Scaling different Dataset using Logarithmic Scaling Method

# Importing Libraries
import pandas as pd
import numpy as np
import pred_lib as pl
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# Importing Dataset
dataset = pd.read_csv('../Dataset/MusaDataSet_1.csv')
X = dataset.iloc[:,1].values

# Linear Interpolation Method
l = int(len(X)*0.8)
a = X[:l]
test = X[l:]
train = []
train.append(a[0])
for i in range(1,len(a)):
    train.append((a[i-1]+a[i])/2)
    train.append(a[i]) 

new_X = pl.Scaling(np.append(train, test))

functions = [pl.RandomForestRegression, pl.DecisionTreeRegression,
             pl.RidgeRegression, pl.BayesianRidgeRegression,
             pl.SVRegression,  pl.KNeighborsRegression]

function_name = ['RandomForest Regression', 'DecisionTree Regression', 'Ridge Regression',
                  'BayesianRidge Regression', 'Support Vector Regression',
              'KNeighbors Regression']


for lag_len in range(2,6):
    print('------------','Lag Length: ',lag_len,'------------')
    for index in range(len(functions)):
        z = pl.Lagging(new_X, lag_len)
        X2D = z[:,:-1]
        y2D = z[:,-1]
        
        t_size = (len(X)-(len(X)*0.8))/((len(X)*0.8)+len(X)-1)

        
        X_train, X_test, y_train, y_test = train_test_split(X2D, y2D, test_size=t_size,shuffle=False)
        result1, result2 = functions[index](X_train, X_test, y_train, y_test)
        
        y_test = X[-len(y_test):]
        result2 = pl.RevScale(result2)
        
        print('***********************************')
        print('Regression: ',function_name[index])
        print('NRMSE: ', round(pl.NRMSE(y_test, result2), 6))
        print('SSE: ', round(pl.SSE(y_test, result2), 6))
        print('MSE: ', round(mean_squared_error(y_test, result2), 6))
        print('Next Failure time: ',pl.RevScale(result1)[0])
        print('***********************************')
        