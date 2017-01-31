
from libcpp.vector cimport vector                                               
from libcpp.map cimport map as omap                                             
from libcpp.pair cimport pair                                                   

cdef extern from "cache.h" nogil:                                      
    cdef cppclass Cache nogil:                                          
        Cache()
        Cache(int)                                                         
        void add_entry(int, pair[int, int]&, double&)
        double lookup(int, pair[int, int]&)                  
        vector[omap[pair, double]] cache_map
