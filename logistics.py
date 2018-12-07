## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

from loaddata import loaddata, splitdata
from sklearn.linear_model import LogisticRegression
import numpy as np
import time


def logistics(Xtrain, ytrain, Xtest):
    start = time.time()
    clf = LogisticRegression(random_state=0, solver='sag', multi_class='multinomial', max_iter=100)
    clf.fit(Xtrain, ytrain)
    train = time.time() - start
    start = time.time()
    result = clf.predict(Xtest)
    predict = time.time() - start
    print('{}\t{}'.format(train, predict))
    return result


if __name__ == '__main__':
    X, y = loaddata()
    print("Finish loading {} data points ==================".format(len(y)))
    result = []
    for split in splitdata(X, y, 8):
        Xtrain, ytrain = split[0]
        Xtest, ytest = split[1]
        print("Running logistics regression on {} training samples and {} test samples"
              .format(len(ytrain), len(ytest)))
        ypredict = logistics(Xtrain, ytrain, Xtest)
        incorrect = np.count_nonzero(np.subtract(ypredict, ytest))
        result.append(1 - incorrect / len(ypredict))
    print('Result')
    for r in result: print(r)
