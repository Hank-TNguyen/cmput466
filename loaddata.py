import numpy as np

def loaddata(numrows=-1):
    dataset = np.genfromtxt('train.csv', delimiter=',',skip_header=1)
    X = dataset[:numrows,1:]
    y = dataset[:numrows,0]
    return X,y

def splitdata(X,y,k):
    N = len(y)

    idx_count = {}
    for i in range(N):
        try: idx_count[y[i]].append(i)
        except: idx_count[y[i]] = [i]

    for i in range(k):
        train_index = []
        test_index = []
        for _class in idx_count:
            chunk = len(idx_count[_class]) // k
            train_index += idx_count[_class][0: i * chunk] + \
                          idx_count[_class][(i + 1) * chunk: -1]
            test_index += idx_count[_class][i * chunk: (i + 1) * chunk]
        np.random.shuffle(train_index)
        np.random.shuffle(test_index)

        trainset = (X[train_index], y[train_index])
        testset = (X[test_index], y[test_index])

        yield trainset, testset


if __name__ == '__main__':
    X,y = loaddata()

    for split in splitdata(X,y, 8):
        Xtrain, ytrain = split[0]
        Xtest, ytest = split[1]
        print(Xtest, ytest)
        break