from distutils.core import setup, Extension
from Cython.Build import cythonize

setup(ext_modules = cythonize([
    Extension("tokenizers", sources=["tokenizers.pyx"], language="c++",
              extra_compile_args = ["-O3", "-ffast-math", "-march=native"]),
    Extension("executor", sources=["executor.pyx"], language="c++",         
              extra_compile_args = ["-O3", "-ffast-math", "-march=native"]) 
 ]))
