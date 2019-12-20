import numpy as np
from scipy.stats import rankdata
from sklearn.preprocessing import binarize

def ranking(l, method='max'):
    return rankdata(-np.array(l), method=method)

def matrix_to_index(X, y):
    X, y = np.array(X), np.array(y)
    assert np.all(X >= 0)
    assert np.all(np.isin(y, [0, 1]))

    X_bt = np.transpose(binarize(X))
    y = np.array(y)
    ep = np.sum(X_bt & y, axis=1)
    ef = np.sum(X_bt & (1 - y), axis=1)
    return ep, np.sum(y) - ep, ef, np.sum(y) - ef

def simple_nn(input_dim, num_nodes=300):
    import keras
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(activation='relu', input_dim=input_dim, units=num_nodes,
        kernel_initializer='uniform'))
    model.add(keras.layers.Dense(activation='softmax', units=2,
        kernel_initializer='uniform'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model
