## https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import time


'''def knn(Xtrain,ytrain,Xtest):
    start = time.time()
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(Xtrain, ytrain)
    train = time.time() - start
    start = time.time()
    result = neigh.predict(Xtest)
    predict = time.time() - start
    print('{}\t{}'.format(train, predict))
    return result
'''
if __name__ == '__main__':

    dataset = np.loadtxt('data_banknote_authentication.txt', delimiter=',')
    dataset = shuffle(dataset)

    X_train, X_test, y_train, y_test = train_test_split(dataset[:,:4], dataset[:,4:5], test_size=0.3)

    '''xtrain = dataset[0:800,0:4]
    ytrain = dataset[0:800,4:5]
    print(xtrain.shape)
    print(ytrain.shape)

    xtest = dataset[800:,0:4]
    ytest = dataset[800:,4:5]
    print(xtest.shape)
    print(ytest.shape)'''

    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(X_train,np.ravel(y_train))

    ypred = neigh.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, ypred))

