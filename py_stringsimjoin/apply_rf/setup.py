from distutils.core import setup, Extension
from Cython.Build import cythonize

setup(ext_modules = cythonize([
    Extension("sim_join", sources=["sim_join.pyx", "sim_cpp.cpp"],
              language="c++", libraries=["m"],
              extra_compile_args = ["-O3", "-ffast-math", "-march=native"])
 ]))
