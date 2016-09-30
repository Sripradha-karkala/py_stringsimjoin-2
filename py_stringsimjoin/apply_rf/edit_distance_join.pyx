
from cython.parallel import prange                                              

from libcpp.vector cimport vector                                               
from libcpp.set cimport set as oset                                             
from libcpp.string cimport string                                               
from libcpp cimport bool                                                        
from libcpp.map cimport map as omap                                             
from libcpp.pair cimport pair     

from py_stringsimjoin.apply_rf.sim_functions cimport edit_distance


cpdef vector[pair[int, int]] ed_join(vector[vector[int]]& ltokens, 
                                     vector[vector[int]]& rtokens,
                                     int qval,
                                     double threshold,
                                     vector[string]& lstrings,
                                     vector[string]& rstrings):                                           
    print 'l size. : ', ltokens.size(), ' , r size : ', rtokens.size()          
    cdef vector[vector[pair[int, int]]] output_pairs
    cdef vector[pair[int, int]] partitions, final_output_pairs, part_pairs
    cdef int i, n=rtokens.size(), ncpus=4, partition_size, start=0, end                                   
    cdef InvertedIndex index
    build_prefix_index(ltokens, qval, threshold, index)                                 
    
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
        ed_join_part(partitions[i], ltokens, rtokens, qval, threshold, index, lstrings, rstrings, output_pairs[i])

    for part_pairs in output_pairs:
        final_output_pairs.insert(final_output_pairs.end(), part_pairs.begin(), part_pairs.end())

    return final_output_pairs

cdef inline int int_min(int a, int b) nogil: return a if a <= b else b

cdef void ed_join_part(pair[int, int] partition, 
                       vector[vector[int]]& ltokens, 
                       vector[vector[int]]& rtokens, 
                       int qval, double threshold, InvertedIndex& index,
                       vector[string]& lstrings, vector[string]& rstrings, 
                       vector[pair[int, int]]& output_pairs) nogil:    
    cdef oset[int] candidates                                      
    cdef vector[int] tokens
    cdef int j=0, m, i, prefix_length, cand                    
    cdef double edit_dist               
 
    for i in range(partition.first, partition.second):
        tokens = rtokens[i]                        
        m = tokens.size()                                                      
        prefix_length = int_min(<int>(qval * threshold + 1), m)                 
                                                                                
        for j in range(prefix_length):                                          
            for cand in index.index[tokens[j]]:                                             
                candidates.insert(cand)               

#        print i, candidate_overlap.size()                                      
        for cand in candidates:
            if m - threshold <= index.size_vector[cand] <= m + threshold:
                edit_dist = edit_distance(lstrings[cand], rstrings[i])                                         
                if edit_dist <= threshold:                                       
                    output_pairs.push_back(pair[int, int](cand, i))     

        candidates.clear()


cdef void build_prefix_index(vector[vector[int]]& token_vectors, int qval, double threshold, InvertedIndex &inv_index):
    cdef vector[int] tokens, size_vector                                                 
    cdef int i, j, m, n=token_vectors.size(), prefix_length
    cdef omap[int, vector[int]] index                                 
    for i in range(n):                                                      
        tokens = token_vectors[i]                                           
        m = tokens.size()                                                   
        size_vector.push_back(m)                                       
        prefix_length = int_min(<int>(qval * threshold + 1), m)
            
        for j in range(prefix_length):                                      
            index[tokens[j]].push_back(i)           
    inv_index.set_fields(index, size_vector)


cdef extern from "inverted_index.h" nogil:
    cdef cppclass InvertedIndex nogil:
        InvertedIndex()
        InvertedIndex(omap[int, vector[int]]&, vector[int]&)
        void set_fields(omap[int, vector[int]]&, vector[int]&)
        omap[int, vector[int]] index
        vector[int] size_vector
