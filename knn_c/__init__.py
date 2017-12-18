import threading

from glob import glob
import os
import sys

import numpy as np
import cffi


class FuncThread(threading.Thread):
    def __init__(self, target, *args):
        threading.Thread.__init__(self)
        self._target = target
        self._args = args

    def run(self):
        self._target(*self._args)


def get_dists_and_knn(X, K, num_threads):

    ffi = cffi.FFI()
    ffi.cdef(
            """void get_distances_and_neighbors(double* X, int N, int D,
                                                int* knn, double* dists, int K,
                                                int num_threads);""")

    path = os.path.dirname(os.path.realpath(__file__))
    try:
        sofile = (glob(os.path.join(path, '*knn*.so')) +
                    glob(os.path.join(path, '*knn*.dll')))[0]
        C = ffi.dlopen(os.path.join(path, sofile))
    except (IndexError, OSError):
        raise RuntimeError('Cannot find/open knn shared library')

    assert X.ndim == 2, 'X should be 2D array.'

    X = np.array(X, dtype=float, order='C', copy=True)
    N, D = X.shape

    knn = np.zeros((N, K), dtype='intc')
    dists = np.zeros((N, K))

    cffi_X = ffi.cast('double*', X.ctypes.data)
    cffi_knn = ffi.cast('int*', knn.ctypes.data)
    cffi_dists = ffi.cast('double*', dists.ctypes.data)

    t = FuncThread(C.get_distances_and_neighbors, cffi_X, N, D,
                                                    cffi_knn, cffi_dists, K, num_threads)
    t.daemon = True
    t.start()

    while t.is_alive():
        t.join(timeout=1.0)
        sys.stdout.flush()

    #C.get_distances_and_neighbors(cffi_X, N, D, cffi_knn, cffi_dists, K, num_threads)

    return (dists, knn)
