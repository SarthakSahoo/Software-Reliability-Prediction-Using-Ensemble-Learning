# Radial Basis Function Neural Network

# Importing the Libraries
import pandas as pd
import nnpred as pn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Importing Dataset
dataset = pd.read_csv('../Dataset/IyerLeeDataSet.csv')
X = dataset.iloc[:,1].values

new_X = pn.Scaling(X)

functions = [pn.RBFNN]

function_name = ['RBFNN']

for lag_len in range(2,6):
    print('------------','Lag Length: ',lag_len,'------------')
    for index in range(len(functions)):
        z = pn.Lagging(new_X, lag_len)
        X2D = z[:,:-1]
        y2D = z[:,-1]
        
        X_train, X_test, y_train, y_test = train_test_split(X2D, y2D, test_size=0.2,shuffle=False)
    
        result1, result2 = functions[index](X_train, X_test, y_train, y_test)
        
        y_test = X[-len(y_test):]
        result2 = pn.RevScale(result2)
        
        print('***********************************')
        print('Regression: ',function_name[index])
        print('NRMSE: ',round(pn.NRMSE(y_test, result2)[0],6))
        print('SSE: ',round(pn.SSE(y_test, result2)[0],6))
        print('MSE: ',round(mean_squared_error(y_test, result2),6))
        print('Next Failure time: ',pn.RevScale(result1)[0])
        print('***********************************')