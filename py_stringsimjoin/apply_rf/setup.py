from distutils.core import setup, Extension
from Cython.Build import cythonize

setup(ext_modules = cythonize([
    Extension("tokenizers", sources=["tokenizers.pyx"], language="c++",
              extra_compile_args = ["-O3", "-ffast-math", "-march=native"]),
    Extension("jaccard_join", sources=["jaccard_join.pyx", "PositionIndex.cpp"], language="c++",             
              extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp"], 
              extra_link_args=['-fopenmp']),
    Extension("executor", sources=["executor.pyx"], language="c++",         
              extra_compile_args = ["-O3", "-ffast-math", "-march=native"]) 
 ]))
