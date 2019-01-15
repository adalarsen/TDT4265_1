
import mnist
import numpy as np
#mnist.init()
X_train, Y_train, X_test, Y_test = mnist.load()

#print("X: {}. Y: {}".format(X_test[0], Y_test[0]))

X_train = np.concatenate((X_train, np.ones((X_train.shape[0], 1))), axis=1)
print("X shape with bias:", X_train.shape)


def remove_all_but_twos_and_threes(X_train, Y_train, X_test, Y_test):
    count_train = 0
    count_test = 0
    for i in Y_train:
        if (i == 2 or i == 3):
            count_train = count_train + 1
    for i in Y_test:
        if (i == 2 or i == 3):
            count_test = count_test + 1
    X_train_2 = np.zeros([count_train, 785])
    X_test_2 = np.zeros([count_train, 784])
    Y_train_2 = np.zeros(count_train)
    Y_test_2 = np.zeros(count_test)
    count = 0
    for i in range(0,len(Y_train)):
        if (Y_train[i] == 2 or Y_train[i] == 3):
            Y_train_2[count] = Y_train[i]
            X_train_2[count] = X_train[i]
            count = count + 1
    count = 0
    for i in range(0,len(Y_test)):
        if (Y_test[i] == 2 or Y_test[i] == 3):
            Y_test_2[count] = Y_test[i]
            X_test_2[count] = X_test[i]
            count = count + 1
    X_test = X_test_2
    Y_test = Y_test_2
    X_train = X_train_2
    Y_train = Y_train_2
    return X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = remove_all_but_twos_and_threes(X_train,Y_train, X_test, Y_test)

X_train = X_train[:10000]
Y_train = Y_train[:10000]
X_test = X_test[:1000]
Y_test = Y_test[:1000]

def train_val_split(X, Y, val_percentage):
    """
      Selects samples from the dataset randomly to be in the validation set. Also, shuffles the train set.
      --
      X: [N, num_features] numpy vector,
      Y: [N, 1] numpy vector
      val_percentage: amount of data to put in validation set
    """
    dataset_size = X.shape[0]
    idx = np.arange(0, dataset_size)
    np.random.shuffle(idx)

    train_size = int(dataset_size * (1 - val_percentage))
    idx_train = idx[:train_size]
    idx_val = idx[train_size:]
    X_train, Y_train = X[idx_train], Y[idx_train]
    X_val, Y_val = X[idx_val], Y[idx_val]
    return X_train, Y_train, X_val, Y_val

X_train, Y_train, X_val, Y_val = train_val_split(X_train, Y_train, 0.1)
print("Train shape: X: {}, Y: {}".format(X_train.shape, Y_train.shape))
print("Validation shape: X: {}, Y: {}".format(X_val.shape, Y_val.shape))

