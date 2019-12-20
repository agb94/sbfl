import math
import numpy as np
from sklearn.preprocessing import maxabs_scale, FunctionTransformer

def log_scale(base=10):
    def log_b(n):
        return math.log(n, base) + 1 if n > 0 else 0
    log_transformer = FunctionTransformer(np.vectorize(log_b), validate=True)
    return log_transformer.transform

def default_scale(X):
    return maxabs_scale(log_scale(base=10)(X))
