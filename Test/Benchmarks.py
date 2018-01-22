import time
import random as rd
import statistics as stat
import matplotlib.pyplot as plt
import multiprocessing as mt
import numpy as np
from knn_c import get_dists_and_knn
from sklearn.neighbors import NearestNeighbors
from scanpy.data_structs.data_graph import get_distance_matrix_and_neighbors, get_sparse_distance_matrix

def c_get_distance_matrix_and_neighbors(X, k, n_jobs=1):
    dists, knn = get_dists_and_knn(X, k-1, n_jobs)
    dists = dists**2
    Dsq = get_sparse_distance_matrix(knn, dists, X.shape[0], k)
    return Dsq, knn, dists

def sklearn_get_distance_matrix_and_neighbors(X, k, n_jobs=1):
    sklearn_neighbors = NearestNeighbors(n_neighbors=k-1, n_jobs=n_jobs)
    sklearn_neighbors.fit(X)
    dists, knn = sklearn_neighbors.kneighbors()
    dists = dists.astype('float32')**2
    Dsq = get_sparse_distance_matrix(knn, dists, X.shape[0], k)
    return Dsq, knn, dists

sequential_get_distance_matrix_and_neighbors = lambda X, k, n_jobs: get_distance_matrix_and_neighbors(X, k)
sequential_get_distance_matrix_and_neighbors.__name__ = 'sequential_get_distance_matrix_and_neighbors'

num_jobs = mt.cpu_count()

print('Test on a small matrix.')
X = np.array([[1, 2], [1, 1], [2, 2], [4, 4]])

_, knn_c, dists_c = c_get_distance_matrix_and_neighbors(X, 3, n_jobs=num_jobs)
_, knn, dists = get_distance_matrix_and_neighbors(X, 3, n_jobs=num_jobs)

assert np.equal(knn_c, knn).all()
assert np.isclose(dists_c, dists).all()

print('Test on a medium matrix.\n')

X = np.random.rand(5000, 500)

_, knn_c, dists_c = c_get_distance_matrix_and_neighbors(X, 10, n_jobs=num_jobs)
_, knn, dists = get_distance_matrix_and_neighbors(X, 10, n_jobs=num_jobs)

assert np.equal(knn_c, knn).all()
assert np.isclose(dists_c, dists).all()

functions = get_distance_matrix_and_neighbors, sequential_get_distance_matrix_and_neighbors, \
            c_get_distance_matrix_and_neighbors, sklearn_get_distance_matrix_and_neighbors

total_time = {f.__name__: [] for f in functions}

shapes_j = [(2000, 200, 4), (5000, 500, 4), (8000, 800, 4), (12000, 1000, 4), (22000, 1500, 4)]
ticks = []

for n, m, j in shapes_j:

    times = {f.__name__: [] for f in functions}
    ticks.append(str((n, m)))

    for i in range(j):
        X = np.random.rand(n, m)
        func = functions[i] if j == len(functions) else rd.choice(functions)
        print('step ', i+1, ' with function ', func.__name__)

        t0 = time.time()
        func(X, 10, n_jobs=num_jobs)
        t1 = time.time()
        times[func.__name__].append(t1 - t0)

    print('\nX shape: ', (n, m))
    for name, numbers in times.items():

        if j == len(functions):
            print('FUNCTION:', name, 'Used', len(numbers), 'times')
            print('\tTIME', numbers[0], 'seconds')
            total_time[name].append(numbers[0])
        else:
            print('FUNCTION:', name, 'Used', len(numbers), 'times')
            print('\tMEDIAN', stat.median(numbers), 'seconds')
            print('\tMEAN  ', stat.mean(numbers), 'seconds')
            total_time[name].append(stat.mean(numbers))
            print('\tSTDEV ', stat.stdev(numbers), 'seconds')
    print('')

x = range(len(shapes_j))
for name, numbers in total_time.items():
    lbl = name+' with n_jobs='+str(num_jobs) if name != 'sequential_get_distance_matrix_and_neighbors' else \
                                                            'get_distance_matrix_and_neighbors with n_jobs=1'
    plt.plot(x, numbers, marker='o', label=lbl)
plt.xticks(x, ticks)
plt.title('Running time (num of neighbours = 10)')
plt.xlabel('Matrix size')
plt.ylabel('Time (seconds)')
plt.legend()

plt.savefig('time.png')
