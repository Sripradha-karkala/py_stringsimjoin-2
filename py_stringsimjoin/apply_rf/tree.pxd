
from libcpp.vector cimport vector                                               

from py_stringsimjoin.apply_rf.rule cimport Rule                                


cdef extern from "tree.h" nogil:                                                
    cdef cppclass Tree nogil:                                                   
        Tree()                                                                  
        Tree(vector[Rule]&)                                                     
        vector[Rule] rules  
