import numpy as np
from . import formula
from .scaling import default_scale
from .utils import matrix_to_index, simple_nn, filtering_mask, random_oversampler

def ochiai(X, y):
    return formula.ochiai(*matrix_to_index(X, y))

def op2(X, y):
    return formula.op2(*matrix_to_index(X, y))

def dstar(X, y, star=2):
    return formula.dstar(*matrix_to_index(X, y), star=star)

def tarantula(X, y):
    return formula.tarantula(*matrix_to_index(X, y))

def neural_network(X, y, model=None, epochs=50, ratio=0.5, preprocessor=None, virtual_inputs=None, verbose=False, mask=None):
    import keras
    from keras.utils import np_utils

    X, y = np.array(X), np.array(y)
    if verbose:
        print("Original shape       -- X: {}, y: {}".format(X.shape, y.shape))

    if mask is None:
        # Default Preprocessing
        mask = filtering_mask(X, y)

    if np.all(mask):
        return np.ma.array(np.zeros(mask.shape[0]), mask=mask)

    if preprocessor:
        res_X, res_y = preprocessor(X[:, ~mask], y)
    else:
        X = default_scale(X[:, ~mask])
        res_X, res_y = random_oversampler(X, y, ratio=ratio)

    if verbose:
        print("After preprocessing  -- X: {}, y: {}".format(res_X.shape, res_y.shape))

    res_y = np_utils.to_categorical(res_y, 2)
    with keras.backend.tensorflow_backend.tf.device('/gpu:0'):
        if model is None:
            num_nodes = max(2, int(res_X.shape[1]/2))
            print("# neurons in the hidden layer: {}".format(num_nodes))
            model = simple_nn(np.sum(~mask), num_nodes=num_nodes)
        hist = model.fit(res_X, res_y, epochs=epochs, verbose=verbose, batch_size=16, shuffle=True)

        if virtual_inputs is not None:
            scores = model.predict(virtual_inputs)[:, 0]
        else:
            virtual_inputs = np.identity(np.sum(~mask))
            scores = np.ma.array(np.zeros(mask.shape[0]), mask=mask)
            scores[~scores.mask] = model.predict(virtual_inputs)[:, 0]

    return scores
