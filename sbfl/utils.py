import numpy as np
import math
from scipy.stats import rankdata
from sklearn.preprocessing import binarize

def filtering_mask(X, y):
    return np.sum(X[y==0, :], axis=0) == 0

def ranking(l, method='max'):
    return rankdata(-np.array(l), method=method)

def matrix_to_index(X, y):
    X, y = np.array(X), np.array(y)
    assert np.all(X >= 0)
    assert np.all(np.isin(y, [0, 1]))

    X = binarize(X)
    y = np.array(y)
    e_f = np.sum(X[y==0], axis=0)
    n_f = np.sum(y == 0) - e_f
    e_p = np.sum(X[y==1], axis=0)
    n_p = np.sum(y == 1) - e_p
    return e_p, n_p, e_f, n_f

def simple_nn(input_dim, num_nodes=300):
    import keras
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(activation='relu', input_dim=input_dim, units=num_nodes,
        kernel_initializer='uniform'))
    model.add(keras.layers.Dense(activation='softmax', units=2,
        kernel_initializer='uniform'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model

def shuffle(X, y):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]

    return X, y

def random_oversampler(X, y, ratio=0.5):
    count = np.bincount(y)
    if count[0] < count[1] * ratio:
        # oversample 0 class
        target_class = 0
        num_samples = int(count[1] * ratio - count[0])
    else:
        # oversample 1 class
        target_class = 1
        num_samples = int(count[0] / ratio - count[1])
    repeat = math.ceil(num_samples/X[y==0].shape[0])
    res_X = np.concatenate((X, np.repeat(X[y==0], repeat, axis=0)), axis=0)
    res_y = np.concatenate((y, np.array([target_class] * repeat * X[y==0].shape[0])), axis=0)
    
    shuffle(res_X, res_y)

    return res_X, res_y
