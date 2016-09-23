
from libc.math cimport ceil
from libcpp.vector cimport vector                                               
from libcpp.set cimport set as oset                                             
from libcpp.string cimport string                                               
from libcpp cimport bool                                                        
from libcpp.map cimport map as omap                                             
from libcpp.pair cimport pair     


cpdef jaccard_join(const vector[vector[int]]& ltokens, const vector[vector[int]]& rtokens, 
                   double threshold):
    cdef omap[int, vector[pair[int, int]]] index
    cdef omap[int, int] candidate_overlap
    cdef vector[pair[int, int]] candidates
    cdef pair[int, int] cand
    cdef int i, j, m, n=rtokens.size(), prefix_length
    cdef vector[int] tokens
    index = build_index(ltokens, threshold)
    
    for i in range(n):
        tokens = rtokens[i]
        m = tokens.size()
        prefix_length = m - ceil(threshold * m) + 1                             
        
        for j in range(prefix_length):
            candidates = index[tokens[i]]
            for cand in candidates:
                candidate_overlap[cand] += 1 
        
         
 

cdef omap[int, vector[pair[int, int]]] build_index(const vector[vector[int]]& token_vectors, double threshold):
    cdef vector[int] tokens
    cdef int prefix_length, token, i, j, m, n=token_vectors.size()
    cdef umap[int, vector[int]] index
    for i in range(n):
        tokens = token_vectors[i]
        m = tokens.size()
        prefix_length = m - ceil(threshold * m) + 1
        for j in range(prefix_length):
            index[tokens[j]].push_back(pair[int, int](i, j))
    return index
        
              
        

