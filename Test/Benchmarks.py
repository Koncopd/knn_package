import time
import random as rd
import statistics as stat
import matplotlib.pyplot as plt
import multiprocessing as mt
import numpy as np

import utils as u

num_jobs = mt.cpu_count()

#print('Test on a small matrix.')
#X = np.array([[1, 2], [1, 1], [2, 2], [4, 4]])

#_, knn_c, dists_c = c_get_distance_matrix_and_neighbors(X, 3, n_jobs=num_jobs)
#_, knn, dists = u.get_distance_matrix_and_neighbors(X, 3, n_jobs=num_jobs)
#_, knn_n, dists_n = u.knn_nmslib(X, 3, n_jobs=num_jobs)

#assert np.equal(knn_n, knn).all()
#assert np.isclose(dists_n, dists).all()

#print('Test on a medium matrix.\n')
#X = np.random.rand(5000, 500)

#_, knn_c, dists_c = c_get_distance_matrix_and_neighbors(X, 10, n_jobs=num_jobs)
#_, knn, dists = u.get_distance_matrix_and_neighbors(X, 10, n_jobs=num_jobs)
#_, knn_n, dists_n = u.knn_nmslib(X, 10, n_jobs=num_jobs)

#assert np.equal(knn_n, knn).all()
#assert np.isclose(dists_n, dists).all()

functions = u.get_distance_matrix_and_neighbors, u.sequential_get_distance_matrix_and_neighbors, \
            u.c_get_distance_matrix_and_neighbors, u.knn_nmslib, u.approx_annoy_knn, u.approx_knn_nmslib, \
            u.sklearn_get_distance_matrix_and_neighbors

total_time = {f.__name__: [] for f in functions}

shapes_j = [(2000, 200, 7), (5000, 500, 7), (8000, 800, 7), (12000, 1000, 7), (22000, 1500, 7)]
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

plt.figure(figsize=(8, 7))

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
