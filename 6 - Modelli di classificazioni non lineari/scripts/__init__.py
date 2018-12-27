from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from numpy.random import rand
from numpy import sum
from numpy import dot
from numpy import e
import numpy as np

iris = load_breast_cancer()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2)

def accuracy(y_true, y_pred):
    y_true[y_true>0.5]=1
    y_true[y_true<=0.5]=0
    return np.sum(y_true==y_pred)/len(y_true)

def predict(X,W,b):
    return activation(dot(W,X.T)+b)

def activation(z):
    return 1./(1+np.exp(-z))

def mini_batch_GD(X,y,batch_size=32,epochs=100, alpha=.001):

    W=np.zeros(X.shape[1])
    b=rand(1,1)

    batches=int(X.shape[0]/batch_size)

    for _ in range(epochs):
        for batch in range(batches):
            if(batch == batches-1):
                X_batch=X[batch*batch_size:,:]
                y_batch=y[batch*batch_size:]
            else:
                X_batch=X[batch*batch_size:(batch+1)*batch_size,:]
                y_batch=y[batch*batch_size:(batch+1)*batch_size]

            z = dot(W,X_batch.T)+b
            a = activation(z)
            errors = a - y_batch
            dW = np.dot(errors,X_batch)
            db = errors.sum()
            W = W - alpha*dW
            b = b - alpha*db
    return W,b


W,b = mini_batch_GD(X,y)
accuracy(y,predict(X,W,b))
