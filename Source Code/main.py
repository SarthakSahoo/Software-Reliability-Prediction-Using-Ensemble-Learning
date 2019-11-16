# Scaling different Dataset using Logarithmic Scaling Method

# Importing Libraries
import pandas as pd
import pred_lib as pl
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# Importing Dataset
dataset = pd.read_csv('../Dataset/MusaDataSet_1.csv')
X = dataset.iloc[:,1].values

new_X = pl.Scaling(X)

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
        
        X_train, X_test, y_train, y_test = train_test_split(X2D, y2D, test_size=0.2,shuffle=False)
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
        