
from libcpp.vector cimport vector                                               
from libcpp.pair cimport pair                                                   
from py_stringsimjoin.apply_rf.cache cimport Cache

cpdef pair[vector[pair[int, int]], vector[double]] set_sim_join_no_cache(
                                          vector[vector[int]]& ltokens,
                                          vector[vector[int]]& rtokens,         
                                          int sim_type,                         
                                          double threshold, int n_jobs)

cdef pair[vector[pair[int, int]], vector[double]] set_sim_join(vector[vector[int]]& ltokens, 
                                          vector[vector[int]]& rtokens,
                                          int sim_type,
                                          double threshold, int n_jobs, 
                                          Cache& cache, int tok_type)

