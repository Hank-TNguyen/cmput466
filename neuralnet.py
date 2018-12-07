## https://scikit-learn.org/stable/modules/neural_networks_supervised.html

from loaddata import loaddata, splitdata
from sklearn.neural_network import MLPClassifier
import numpy as np
import time

def neuralnet(Xtrain, ytrain, Xtest):
    Xtrain = np.divide(Xtrain, 255)
    Xtest = np.divide(Xtest, 255)
    start = time.time()
    clf = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=400, alpha=1e-4,
                        solver='sgd', tol=1e-4, random_state=1)
    clf.fit(Xtrain, ytrain)
    train = time.time() - start
    start = time.time()
    result = clf.predict(Xtest)
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
        print("Running neural network on {} training samples and {} test samples"
              .format(len(ytrain), len(ytest)))
        ypredict = neuralnet(Xtrain, ytrain, Xtest)
        incorrect = np.count_nonzero(np.subtract(ypredict, ytest))
        result.append(1 - incorrect / len(ypredict))
    print('Result')
    for r in result:print(r)