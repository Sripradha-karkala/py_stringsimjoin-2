
from libcpp.vector cimport vector                                               
from libcpp.pair cimport pair                                                   

cpdef vector[pair[int, int]] jaccard_join(vector[vector[int]]& ltokens, vector[vector[int]]& rtokens,
                   double threshold)
