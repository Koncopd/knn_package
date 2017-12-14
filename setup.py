from distutils.core import setup, Extension

knn_c = Extension('knn_c.knn',
        language = 'c++',
        extra_compile_args = ['-Wall', '-O3', '-fPIC', '-ffast-math', '-funroll-loops', '-fopenmp'],
        extra_link_args = ['-fopenmp'],
        include_dirs = ['.'],
        sources = ['knn_c/knn.cpp'],
        depends = ['knn_c/vptree.h'])

setup (name = 'knn_c',
       version = '0.1',
       url = '',
       author = '',
       author_email = '',
       license = '',
       description = 'K-NN search',
       packages=['knn_c'],
       ext_modules = [knn_c]
)
