import numpy as np
from . import formula
from .scaling import default_scale
from .utils import matrix_to_index, simple_nn, filtering_mask

def ochiai(X, y):
    return formula.ochiai(*matrix_to_index(X, y))

def op2(X, y):
    return formula.op2(*matrix_to_index(X, y))

def dstar(X, y, star=2):
    return formula.dstar(*matrix_to_index(X, y), star=star)

def tarantula(X, y):
    return formula.tarantula(*matrix_to_index(X, y))

def neural_network(X, y, model=None, epochs=50, preprocessor=None, verbose=False):
    import keras
    from keras.utils import np_utils
    from imblearn.over_sampling import RandomOverSampler
    from sklearn.preprocessing import maxabs_scale
    X, y = np.array(X), np.array(y)
    mask = np.ones(X.shape[1], dtype=bool)

    if verbose:
        print("Original shape       -- X: {}, y: {}".format(X.shape, y.shape))
    
    if preprocessor:
        X, y = preprocessor(X, y)
    else:
        from imblearn.over_sampling import RandomOverSampler
        from sklearn.preprocessing import maxabs_scale
        # Default Preprocessing
        mask = filtering_mask(X, y)
        if np.all(mask):
            return np.ma.array(np.zeros(mask.shape[0]), mask=mask)
        X = default_scale(X[:, ~mask])
        X, y = RandomOverSampler(sampling_strategy=0.5).fit_resample(X, y)
    
    if verbose:
        print("After preprocessing  -- X: {}, y: {}".format(X.shape, y.shape))

    if np.all(mask):
        return scroes

    y = np_utils.to_categorical(y, 2)
    with keras.backend.tensorflow_backend.tf.device('/gpu:0'):
        if model is None:
            model = simple_nn(np.sum(~mask))
        hist = model.fit(X, y, epochs=epochs, verbose=verbose)
        virtual_inputs = np.identity(np.sum(~mask))
        scores = np.ma.array(np.zeros(mask.shape[0]), mask=mask)
        scores[~scores.mask] = model.predict(virtual_inputs)[:, 0]
    
    return scores
