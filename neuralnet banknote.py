## https://scikit-learn.org/stable/modules/neural_networks_supervised.html

from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
import time


'''def neuralnet(Xtrain, ytrain, Xtest):
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
    return result'''

if __name__ == '__main__':
    dataset = np.loadtxt('data_banknote_authentication.txt', delimiter=',')
    dataset = shuffle(dataset)
    
    X_train, X_test, y_train, y_test = train_test_split(dataset[:,:4], dataset[:,4:5], test_size=0.3)
    
    
    clf = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=400, alpha=1e-4,
                        solver='sgd', tol=1e-4, random_state=1)
                        
    clf.fit(X_train, np.ravel(y_train))
    ypred = clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, ypred))

