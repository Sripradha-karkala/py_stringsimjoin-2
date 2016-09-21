
from libcpp cimport bool                                                        
from libcpp.vector cimport vector                                               
from libcpp.string cimport string  

cdef class WhitespaceTokenizer:
    cdef bool return_set
    cdef vector[string] tokenize(self, const string&)

cdef class QgramTokenizer:                                                 
    cdef int qval                                                               
    cdef char prefix_pad, suffix_pad                                            
    cdef bool padding, return_set  
    cdef vector[string] tokenize(self, const string&)    
