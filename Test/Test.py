import numpy as np
from knn_c import get_dists_and_knn

X = np.array([[1, 2], [1, 1], [2, 2], [4, 4]])

dists, knn = get_dists_and_knn(X, 2, 2)

print(X)

print(dists)
print(knn)
