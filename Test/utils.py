import nmslib, scipy, numpy as np

from knn_c import get_dists_and_knn
from approx_knn_c import get_dists_and_knn as approx_get_dists_and_knn
from sklearn.neighbors import NearestNeighbors

from scanpy.data_structs.data_graph import get_sparse_distance_matrix, get_distance_matrix_and_neighbors

sequential_get_distance_matrix_and_neighbors = lambda X, k, n_jobs: get_distance_matrix_and_neighbors(X, k)
sequential_get_distance_matrix_and_neighbors.__name__ = 'sequential_get_distance_matrix_and_neighbors'

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

def approx_annoy_knn(X, k, n_jobs=1):
    dists, knn = approx_get_dists_and_knn(X, k-1, 120, 2500, n_jobs)
    dists = dists**2
    Dsq = get_sparse_distance_matrix(knn, dists, X.shape[0], k)
    return Dsq, knn, dists

def knn_nmslib(data, k=30, method='vptree', verbose=False, n_jobs=1, **kwargs):
    if isinstance(data, scipy.sparse.csr_matrix):
        space = 'l2_sparse'
        data_type=nmslib.DataType.SPARSE_VECTOR
    else:
        space = 'l2'
        data_type=nmslib.DataType.DENSE_VECTOR

    index = nmslib.init(method=method, space=space, data_type=data_type)
    index.addDataPointBatch(data)
    index.createIndex(print_progress=verbose, **kwargs)

    if verbose:
        print('knn: Indexing done.')

    neig = index.knnQueryBatch(data, k=k, num_threads=n_jobs)
    if verbose:
        print('knn: Query done.')

    indices = np.vstack(x[0][1:] for x in neig)  # exclude self
    distances = np.vstack(x[1][1:] for x in neig)

    distances = distances.astype('float32')**2
    Dsq = get_sparse_distance_matrix(indices, distances, data.shape[0], k)

    # two N x k-1 matrices
    return Dsq, indices, distances

def approx_knn_nmslib(data, k=30, verbose=False, n_jobs=1):
    return knn_nmslib(data=data, k=k, method='hnsw', verbose=verbose, n_jobs=n_jobs)
