
import mnist
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
#mnist.init()
X_train, Y_train, X_test, Y_test = mnist.load()

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
    X_test_2 = np.zeros([count_test, 784])
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

    X_test = (X_test_2)
    X_test /= 255
    Y_test = Y_test_2-2
    X_train = X_train_2
    X_train /= 255
    Y_train = Y_train_2-2
    return X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = remove_all_but_twos_and_threes(X_train,Y_train, X_test, Y_test)
print(X_train.shape)
print(Y_train.shape)

X_train = X_train[:1000]
Y_train = Y_train[:1000]
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

def logistic_loss(targets, outputs):
    targets = np.reshape(targets,outputs.shape)
    assert targets.shape == outputs.shape
    log_error = targets*np.log(outputs) + (1-targets)*np.log(1-outputs)
    mean_log_error = -log_error.mean()
    return mean_log_error

def forward_pass(X, w):
    return X.dot(w)

def gradient_descent(X, outputs, targets, weights, learning_rate):
    N = X.shape[0]

    targets = np.reshape(targets,outputs.shape)
    assert outputs.shape == targets.shape

    for i in range(weights.shape[0]):
        # Gradient for logistic regression

        dw_i = -(targets-1/(1+np.exp(-outputs)))*X[:, i:i+1]
        expected_shape = (N, 1)
        assert dw_i.shape == expected_shape, \
        "dw_j shape was: {}. Expected: {}".format(dw_i.shape, expected_shape)
        dw_i = dw_i.sum(axis=0)

        weights[i] = weights[i] - learning_rate * dw_i

    return weights

def prediction(X, w):
    outs = forward_pass(X,w)
    outputs = np.divide(1, (1+np.exp(-outs)))
    pred = (outputs > .5)[:, 0]
    print(pred.shape)
    return pred

## TRAINING

# Hyperparameters
epochs = 40
batch_size = 32

# Tracking variables
TRAIN_LOSS = []
VAL_LOSS = []
TRAINING_STEP = []
TRAIN_ACC = []
num_features = X_train.shape[1]

num_batches_per_epoch = X_train.shape[0] // batch_size
check_step = num_batches_per_epoch // 10



w = np.random.normal(size=(num_features, 1))*0.01

def train_loop(w):
    training_it = 0
    T = 0.01
    for epoch in range(epochs):
        print(epoch / epochs)
        # shuffle(X_train, Y_train)
        for i in range(num_batches_per_epoch):
            init_learning_rate = 0.001
            #learning_rate = init_learning_rate / (1 + training_it/T)
            learning_rate = 0.0001
            training_it += 1
            X_batch = X_train[i * batch_size:(i + 1) * batch_size]
            Y_batch = Y_train[i * batch_size:(i + 1) * batch_size]

            out = forward_pass(X_batch, w)
            w = gradient_descent(X_batch, out, Y_batch, w, learning_rate)

            if i % check_step == 0:
                # Training set
                train_out = forward_pass(X_train, w)
                train_out = np.divide(1,(1+np.exp(-train_out)))
                train_loss = logistic_loss(Y_train, train_out)
                TRAIN_LOSS.append(train_loss)
                TRAINING_STEP.append(training_it)

                val_out = 1/(1+np.exp(-forward_pass(X_val, w)))
                val_loss = logistic_loss(Y_val, val_out)
                VAL_LOSS.append(val_loss)

                TRAIN_ACC.append(100*np.sum(prediction(X_train, w)==Y_train)/len(Y_train))
                print(TRAIN_ACC[-1])

                if (epoch % 1 == 0):
                    print("Epoch: %d, Loss: %.8f, Error: %.8f"
                    % (epoch, train_loss, np.mean(TRAIN_LOSS)))

    return w

w = train_loop(w)
plt.figure(figsize=(12, 8 ))
#plt.ylim([0, 1])
plt.xlabel("Training steps")
plt.ylabel("MSE Loss")
plt.plot(TRAINING_STEP, TRAIN_LOSS, label="Training loss")
plt.plot(TRAINING_STEP, VAL_LOSS, label="Validation loss")
plt.legend() # Shows graph labels
plt.show()
