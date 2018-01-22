#get_distance_matrix_and_neighbors - 11.4, 30.4, 34.7
#sequential_get_distance_matrix_and_neighbors - 10.5, 35.6, 51.5
#c_get_distance_matrix_and_neighbors - 2.1, 4.3, 7.0
#sklearn_get_distance_matrix_and_neighbors - 1.6, 2.8, 4.3

import numpy as np
import matplotlib.pyplot as plt

ind = np.arange(3)

gdp = [11.4, 30.4, 34.7]
gds = [10.5, 35.6, 51.5]
cgd = [2.1, 4.3, 7.0]
skgd = [1.6, 2.8, 4.3]

width = 0.2

fig, ax = plt.subplots()
rects1 = ax.bar(ind, gdp, width)
rects2 = ax.bar(ind + width, gds, width)
rects3 = ax.bar(ind + 2*width, cgd, width)
rects4 = ax.bar(ind + 3*width, skgd, width)

ax.set_xticks(ind + 3*width / 2)
ax.set_xticklabels(('(12000, 1000)', '(22000, 1500)', '(30000, 2000)'))

lbls = ('get_distance_matrix_and_neighbors n_jobs=4', 'get_distance_matrix_and_neighbors n_jobs=1', \
        'c_get_distance_matrix_and_neighbors n_jobs=4', 'sklearn_get_distance_matrix_and_neighbors n_jobs=4')

ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]), lbls)

ax.set_ylabel('Percentage of total memory')
ax.set_xlabel('Matrix size')
ax.set_title('Memory usage (num of neighbours = 10)')

plt.savefig('memory.png')
