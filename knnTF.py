## https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

from loaddata import loaddata, splitdata
import tensorflow as tf
import numpy as np


def knnTF(x_train, y_train, x_test):
    """
    x_train: 60000 x 784 matrix: each row is a flattened image of an MNIST digit
    y_train: 60000 vector: label for x_train
    x_test: 5000 x 784 testing images
    return: predicted y_test which is a 5000 vector
    """

    y_test = []
    init = tf.global_variables_initializer()
    k = 5

    train_images = tf.placeholder("float", [None, 784])
    test_image = tf.placeholder("float", [784])

    distances = -tf.reduce_sum(tf.sqrt(tf.square(tf.subtract(train_images, test_image))), 1)
    top_labels = tf.gather(y_train, tf.nn.top_k(distances, k).indices)

    with tf.Session() as sess:
        sess.run(init)
        for i in range(len(x_test)):
            t = sess.run(top_labels, feed_dict={train_images: x_train, test_image: x_test[i, :]})

            freq = 0
            c = 0
            labels = set(t)
            for i in labels:
                if freq < list(t).count(i):
                    freq = list(t).count(i)
                    c = i
            y_test.append(c)

    return y_test

if __name__ == '__main__':
    X,y = loaddata(8000)
    print("Finish loading {} data points ==================".format(len(y)))
    result = []
    for split in splitdata(X, y, 8):
        Xtrain, ytrain = split[0]
        Xtest, ytest = split[1]
        print("Running K nearest neighbors on {} training samples and {} test samples"
              .format(len(ytrain), len(ytest)))
        ypredict = knnTF(Xtrain, ytrain, Xtest)
        incorrect = np.count_nonzero(np.subtract(ypredict, ytest))
        result.append(1 - incorrect / len(ypredict))
        print(1 - incorrect / len(ypredict))
    print(result)