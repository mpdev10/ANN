from keras.datasets import mnist

from Util.Preprocessing import one_hot, flatten

TRAINING_SIZE = 50000


def get_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_val = flatten(X_train[TRAINING_SIZE:])
    y_val = one_hot(y_train[TRAINING_SIZE:])

    X_train = flatten(X_train[:TRAINING_SIZE])
    y_train = one_hot(y_train[:TRAINING_SIZE])

    X_test = flatten(X_test)
    y_test = one_hot(y_test)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
