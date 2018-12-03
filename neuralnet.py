## https://scikit-learn.org/stable/modules/neural_networks_supervised.html

from loaddata import loaddata, splitdata
from sklearn.neural_network import MLPClassifier
import numpy as np

def neuralnet(Xtrain, ytrain, Xtest):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(Xtrain, ytrain)
    return clf.predict(Xtest)

if __name__ == '__main__':
    X,y = loaddata()
    print("Finish loading data ==================")
    result = []
    for split in splitdata(X, y, 8):
        Xtrain, ytrain = split[0]
        Xtest, ytest = split[1]
        print("Running logistics regression on {} training samples and {} test samples"
              .format(len(ytrain), len(ytest)))
        ypredict = neuralnet(Xtrain, ytrain, Xtest)
        correct = np.count_nonzero(np.subtract(ypredict, ytest))
        result.append(correct / len(ypredict))
        print(correct / len(ypredict))
    print(result)