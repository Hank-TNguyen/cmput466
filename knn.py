## https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

from loaddata import loaddata, splitdata
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import time


def knn(Xtrain,ytrain,Xtest):
    start = time.time()
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(Xtrain, ytrain)
    train = time.time() - start
    start = time.time()
    result = neigh.predict(Xtest)
    predict = time.time() - start
    print('{}\t{}'.format(train, predict))
    return result

if __name__ == '__main__':
    X,y = loaddata()
    print("Finish loading {} data points ==================".format(len(y)))
    result = []
    for split in splitdata(X, y, 8):
        Xtrain, ytrain = split[0]
        Xtest, ytest = split[1]
        print("Running K nearest neighbors on {} training samples and {} test samples"
              .format(len(ytrain), len(ytest)))
        ypredict = knn(Xtrain, ytrain, Xtest)
        incorrect = np.count_nonzero(np.subtract(ypredict, ytest))
        result.append(1- incorrect / len(ypredict))
    print('Result')
    for r in result: print(r)