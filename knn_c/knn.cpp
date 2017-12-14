#ifdef _OPENMP
#include <omp.h>
#endif

#include "vptree.h"


#ifdef _OPENMP
    #define NUM_THREADS(N) ((N) >= 0 ? (N) : omp_get_num_procs() + (N) + 1)
#else
    #define NUM_THREADS(N) (1)
#endif

extern "C"
{

  #ifdef _WIN32
  __declspec(dllexport)
  #endif

  void get_distances_and_neighbors(double* X, int N, int D,
                                    int* knn, double* dists, int K,
                                    int num_threads)
  {
    #ifdef _OPENMP
      omp_set_num_threads(NUM_THREADS(num_threads));
    #if _OPENMP >= 200805
      omp_set_schedule(omp_sched_guided, 0);
    #endif
    #endif

    VpTree<DataPoint, euclidean_distance>* tree = new VpTree<DataPoint, euclidean_distance>();
    std::vector<DataPoint> obj_X(N, DataPoint(D, -1, X));

    for (int n = 0; n < N; n++) {
        obj_X[n] = DataPoint(D, n, X + n * D);
    }
    tree->create(obj_X);

    #ifdef _OPENMP
      #pragma omp parallel for
    #endif
    for (int n = 0; n < N; n++)
    {
        std::vector<DataPoint> indices;
        std::vector<double> distances;

        // Find nearest neighbors
        tree->search(obj_X[n], K + 1, &indices, &distances);

        for (int m = 0; m < K; m++) {
            knn[n*K + m] = indices[m + 1].index();
            dists[n*K + m] = distances[m + 1];
        }

    }

    obj_X.clear();
    delete tree;
  }

}
