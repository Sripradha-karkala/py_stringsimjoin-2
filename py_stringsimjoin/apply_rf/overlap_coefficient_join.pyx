
from cython.parallel import prange                                              

from libcpp.vector cimport vector                                               
from libcpp.set cimport set as oset                                             
from libcpp.string cimport string                                               
from libcpp cimport bool                                                        
from libcpp.map cimport map as omap                                             
from libcpp.pair cimport pair     


cpdef vector[pair[int, int]] ov_coeff_join(vector[vector[int]]& ltokens, 
                                           vector[vector[int]]& rtokens,
                                           double threshold):                                           
    print 'l size. : ', ltokens.size(), ' , r size : ', rtokens.size()          
    cdef vector[vector[pair[int, int]]] output_pairs
    cdef vector[pair[int, int]] partitions, final_output_pairs, part_pairs
    cdef int i, n=rtokens.size(), ncpus=4, partition_size, start=0, end                                   
    cdef InvertedIndex index
    build_inverted_index(ltokens, index)                                 
    
    partition_size = <int>(<float> n / <float> ncpus)
    print 'part size : ', partition_size
    for i in range(ncpus):
        end = start + partition_size
        if end > n or i == ncpus - 1:
            end = n
        partitions.push_back(pair[int, int](start, end))
        print start, end
        start = end            
        output_pairs.push_back(vector[pair[int, int]]())

    for i in prange(ncpus, nogil=True):    
        ov_coeff_join_part(partitions[i], ltokens, rtokens, threshold, index, output_pairs[i])

    for part_pairs in output_pairs:
        final_output_pairs.insert(final_output_pairs.end(), part_pairs.begin(), part_pairs.end())

    return final_output_pairs

cdef inline int int_min(int a, int b) nogil: return a if a <= b else b

cdef void ov_coeff_join_part(pair[int, int] partition, 
                             vector[vector[int]]& ltokens, 
                             vector[vector[int]]& rtokens, 
                             double threshold, InvertedIndex& index, 
                             vector[pair[int, int]]& output_pairs) nogil:              
    cdef omap[int, int] candidate_overlap              
    cdef vector[int] candidates                                      
    cdef vector[int] tokens
    cdef pair[int, int] entry                                             
    cdef int j=0, m, i, cand                    
    cdef double sim_score               
 
    for i in range(partition.first, partition.second):
        tokens = rtokens[i]                        
        m = tokens.size()                                                      
                                                                                
        for j in range(m):                                          
            candidates = index.index[tokens[j]]                                 
            for cand in candidates:                                             
                candidate_overlap[cand] += 1                

#        print i, candidate_overlap.size()                                      
        for entry in candidate_overlap:                                         
            sim_score = <double>entry.second / <double>int_min(m, index.size_vector[entry.first])           
            #print ltokens[entry.first], rtokens[i], entry.second, sim_score
            if sim_score > threshold:                                       
                output_pairs.push_back(pair[int, int](entry.first, i))     


cdef void build_inverted_index(vector[vector[int]]& token_vectors, InvertedIndex &inv_index):
    cdef vector[int] tokens, size_vector                                                 
    cdef int i, j, m, n=token_vectors.size()
    cdef omap[int, vector[int]] index                                 
    for i in range(n):                                                      
        tokens = token_vectors[i]                                           
        m = tokens.size()                                                   
        size_vector.push_back(m)                                       
        for j in range(m):                                      
            index[tokens[j]].push_back(i)           
    inv_index.set_fields(index, size_vector)

cdef extern from "inverted_index.h" nogil:
    cdef cppclass InvertedIndex nogil:
        InvertedIndex()
        InvertedIndex(omap[int, vector[int]]&, vector[int]&)
        void set_fields(omap[int, vector[int]]&, vector[int]&)
        omap[int, vector[int]] index
        vector[int] size_vector
