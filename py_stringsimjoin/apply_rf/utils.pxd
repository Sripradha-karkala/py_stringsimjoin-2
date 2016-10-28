
from libcpp cimport bool
from libcpp.vector cimport vector                                               
from libcpp.string cimport string 


ctypedef double (*simfnptr)(const vector[int]&, const vector[int]&) nogil       
ctypedef bool (*compfnptr)(double, double) nogil                                

cdef int get_sim_type(const string&)
cdef simfnptr get_sim_function(const int) nogil

cdef int get_comp_type(const string&)
cdef compfnptr get_comparison_function(const int) nogil
