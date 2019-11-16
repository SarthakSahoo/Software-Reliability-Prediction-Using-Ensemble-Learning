# Importing the libraries
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop
from rbflayer import RBFLayer, InitCentersRandom
import numpy as np

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


def ANN(X_train, X_test, y_train, y_test):
    regressor = Sequential()
    regressor.add(Dense(input_dim=len(X_train[0]), output_dim=len(X_train[0])+1, kernel_initializer='uniform', activation='relu'))
    regressor.add(Dense(output_dim=len(X_train[0])+1, kernel_initializer='uniform', activation='relu'))
    regressor.add(Dense(output_dim=1, kernel_initializer='uniform',activation='sigmoid'))
    regressor.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    regressor.fit(X_train, y_train, batch_size = 10, epochs = 100, verbose=0)
    row = getRow(X_test, y_test)
    result1 = regressor.predict(row)
    y_pred = regressor.predict(X_test)
    return result1, y_pred

def RBFNN(X_train, X_test, y_train, y_test):
    model = Sequential()
    rbflayer = RBFLayer(10, initializer=InitCentersRandom(X_train), betas=2.0, input_shape=(len(X_train[0]),))
    model.add(rbflayer)
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=RMSprop())
    model.fit(X_train, y_train, batch_size=50, epochs=2000, verbose=0)
    row = getRow(X_test, y_test)
    result1 = model.predict(row)
    y_pred = model.predict(X_test)
    return result1, y_pred