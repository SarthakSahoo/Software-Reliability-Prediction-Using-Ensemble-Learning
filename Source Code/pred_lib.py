# Importing the Libraries
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge, Lasso
from sklearn.neighbors import KNeighborsRegressor


b = 0

#X is list : Scaling function helps in logarithmic scaling for values in range 0 to 1
def Scaling(X):
    global b
    b = (np.e**0.9 - 1) / max(X)
    new_X = [np.log(1+(b*y)) for y in X]    
    return new_X

# Reverse Scaling Function 
def RevScale(value):
    global b
    x = [((np.e**v)-1)/b for v in value]
    return x

#x : list of items, K: lag length
def Lagging(x,k):
	LagLength = len(x)
	z=[list(x[i:i+k]) for i in range(0,LagLength-k+1)]
	return(np.array(z))

# Normalized Root Mean Squared Error function
def NRMSE(YActual,YPredicted):
    Sum1 = 0
    Sum2 = 0
    for index in range(len(YActual)):
        Sum1 = Sum1 + (YActual[index]-YPredicted[index])**2
        Sum2 = Sum2 + YActual[index]**2
    return np.sqrt(Sum1/Sum2)

# Sum of Squared due to Error function
def SSE(YActual,YPredicted):
    result = 0
    for index in range(len(YActual)):
        result = result + (YActual[index] - YPredicted[index])**2
    return result

# RE function to compute error
def RE(YActual,YPredicted):
    return np.abs((YPredicted-YActual)/YActual)*100

# Function to return one specific row for prediction
def getRow(X2D, y2D):
    l = len(X2D)
    row = X2D[l-1][1:]
    row = np.append(row, y2D[l-1])
    return row.reshape(1,-1)

# Function to fit the data and predict the data using the return function from getRow() and by the provided value
def Model(regressor, X_train, X_test, y_train, y_test):
    regressor.fit(X_train, y_train)
    row = getRow(X_test, y_test)
    result1 = regressor.predict(row)
    result2 = regressor.predict(X_test)
    return result1, result2

# RandomForest Regression Function
def RandomForestRegression(X_train, X_test, y_train, y_test):
    regressor = RandomForestRegressor(n_estimators = 17, random_state = 0)
    result1, result2 = Model(regressor, X_train, X_test, y_train, y_test)
    return result1, result2

# DecisionTree Regression Function
def DecisionTreeRegression(X_train, X_test, y_train, y_test):
    regressor = DecisionTreeRegressor(random_state = 0)
    result1, result2 = Model(regressor, X_train, X_test, y_train, y_test)
    return result1, result2

# Ridge Regression Function
def RidgeRegression(X_train, X_test, y_train, y_test):
    regressor = Ridge(alpha = 0.0001)
    result1, result2 = Model(regressor, X_train, X_test, y_train, y_test)
    return result1, result2

# Lasso Regression Function
def LassoRegression(X_train, X_test, y_train, y_test):
    regressor = Lasso(alpha = 0.1)
    result1, result2 = Model(regressor, X_train, X_test, y_train, y_test)
    return result1, result2

# BayesianRidge Regression Function
def BayesianRidgeRegression(X_train, X_test, y_train, y_test):
    regressor = BayesianRidge()
    result1, result2 = Model(regressor, X_train, X_test, y_train, y_test)
    return result1, result2

# Support Vector Regression Function
def SVRegression(X_train, X_test, y_train, y_test):
    regressor = SVR(kernel='rbf', epsilon = 0.01, gamma = 'scale')
    result1, result2 = Model(regressor, X_train, X_test, y_train, y_test)
    return result1, result2

# ElasticNet Regression Function
def ElasticNetRegression(X_train, X_test, y_train, y_test):
    regressor = ElasticNet(random_state = None, alpha = 1.0, l1_ratio = 0.5)
    result1, result2 = Model(regressor, X_train, X_test, y_train, y_test)
    return result1, result2

# Nearest Neighbors Regression Function
def KNeighborsRegression(X_train, X_test, y_train, y_test):
    regressor = KNeighborsRegressor(n_neighbors=2)
    result1, result2 = Model(regressor, X_train, X_test, y_train, y_test)
    return result1, result2