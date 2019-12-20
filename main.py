from sbfl import neural_network
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import binarize

print(neural_network(
	[ [1,0,0], [0, 0, 1], [0,1,0], [0,1,2],  [0,0,3], [0, 1, 1], [0,1,2]],
	[0, 1, 1, 1, 1, 1, 1],
        verbose=True
))
