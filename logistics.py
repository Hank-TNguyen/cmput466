## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

from loaddata import loaddata, splitdata
from sklearn.linear_model import LogisticRegression
import numpy as np


def logistics(Xtrain, ytrain, Xtest):
    clf = LogisticRegression(random_state=0, solver='sag', multi_class='multinomial', max_iter=1000)
    clf.fit(Xtrain, ytrain)
    return clf.predict(Xtest)


if __name__ == '__main__':
    X, y = loaddata(8000)
    print("Finish loading data ==================")
    result = []
    for split in splitdata(X, y, 8):
        Xtrain, ytrain = split[0]
        Xtest, ytest = split[1]
        print("Running neural network on {} training samples and {} test samples"
              .format(len(ytrain), len(ytest)))
        ypredict = logistics(Xtrain, ytrain, Xtest)
        correct = np.count_nonzero(np.subtract(ypredict, ytest))
        result.append(correct / len(ypredict))
        print(correct / len(ypredict))
    print(result)
