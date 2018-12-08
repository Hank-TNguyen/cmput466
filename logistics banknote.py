## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
import time


'''def logistics(Xtrain, ytrain, Xtest):
    start = time.time()
    clf = LogisticRegression(random_state=0, solver='sag', multi_class='multinomial', max_iter=100)
    clf.fit(Xtrain, ytrain)
    train = time.time() - start
    start = time.time()
    result = clf.predict(Xtest)
    predict = time.time() - start
    print('{}\t{}'.format(train, predict))
    return result
'''

if __name__ == '__main__':

    dataset = np.loadtxt('data_banknote_authentication.txt', delimiter=',')
    dataset = shuffle(dataset)
    
    X_train, X_test, y_train, y_test = train_test_split(dataset[:,:4], dataset[:,4:5], test_size=0.3)

    clf = LogisticRegression(random_state=0, solver='sag', multi_class='multinomial', max_iter=1000)
    clf.fit(X_train, np.ravel(y_train))

    ypred = clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, ypred))
