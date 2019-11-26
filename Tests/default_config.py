from Util.Preprocessing.data_loader import get_data

default_parameters = {
    'epochs': 40,
    'batch_size': 25,
    'learning_rate': 0.01,
}

(X_train, y_train), (X_val, y_val), (X_test, y_test) = get_data()
