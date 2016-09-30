
from libcpp.vector cimport vector                                               
from libcpp.pair cimport pair                                                   

cpdef vector[pair[int, int]] set_sim_join(vector[vector[int]]& ltokens, 
                                          vector[vector[int]]& rtokens,
                                          int sim_type,
                                          double threshold)
