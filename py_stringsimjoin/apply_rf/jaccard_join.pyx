
from cython.parallel import prange                                              

from libc.math cimport ceil, floor, round
from libcpp.vector cimport vector                                               
from libcpp.set cimport set as oset                                             
from libcpp.string cimport string                                               
from libcpp cimport bool                                                        
from libcpp.map cimport map as omap                                             
from libcpp.pair cimport pair     


cpdef vector[pair[int, int]] jjaccard_join1(vector[vector[int]]& ltokens, const vector[vector[int]]& rtokens, 
                   double threshold):
    print 'l size. : ', ltokens.size(), ' , r size : ', rtokens.size()
    cdef vector[pair[int, int]] output_pairs
    cdef omap[int, int] candidate_overlap, overlap_threshold_cache
    cdef vector[pair[int, int]] candidates
    cdef pair[int, int] cand, entry
    cdef int k=0, i, j, m, n=rtokens.size(), prefix_length, cand_num_tokens, current_overlap, overlap_upper_bound
    cdef int size, size_lower_bound, size_upper_bound
    cdef vector[int] tokens
    cdef PositionIndex index
    cdef double sim_score
    index = PositionIndex(ltokens, threshold)
     
    for i in prange(n, nogil=True):
        tokens = rtokens[i]
        m = tokens.size()
        prefix_length = <int>(m - ceil(threshold * m) + 1.0)
        size_lower_bound = <int>ceil(threshold * m)
        size_upper_bound = <int>floor(m / threshold)

        for size in range(size_lower_bound, size_upper_bound + 1):
            overlap_threshold_cache[size] = <int>ceil((threshold / (1 + threshold)) * (size + m)) 
         
        for j in range(prefix_length):
            candidates = index.index[tokens[j]]
            for cand in candidates:
                current_overlap = candidate_overlap[cand.first]
                if current_overlap != -1:
                    cand_num_tokens = index.size_vector[cand.first]

                    # only consider candidates satisfying the size filter       
                    # condition.                                                
                    if size_lower_bound <= cand_num_tokens <= size_upper_bound: 
                                                                                
                        if m - j <= cand_num_tokens - cand.second:                    
                            overlap_upper_bound = m - j  
                        else:                                                   
                            overlap_upper_bound = cand_num_tokens - cand.second    
                                                                                
                        # only consider candidates for which the overlap upper  
                        # bound is at least the required overlap.               
                        if (current_overlap + overlap_upper_bound >=            
                                overlap_threshold_cache[cand_num_tokens]):      
                            candidate_overlap[cand.first] = current_overlap + 1       
                        else:                                                   
                            candidate_overlap[cand.first] = -1
#        print i, candidate_overlap.size()             
        for entry in candidate_overlap:
            if entry.second > 0:
                k += 1
                sim_score = jaccard(ltokens[entry.first], rtokens[i])
                #print ltokens[entry.first], rtokens[i], entry.second, sim_score
                if sim_score > threshold:
                    k = k - 1 + 1
#                    output_pairs.push_back(pair[int, int](entry.first, i))
        
        candidate_overlap.clear()
        overlap_threshold_cache.clear()
    print 'k : ', k
    return output_pairs

cpdef vector[pair[int, int]] jaccard_join(vector[vector[int]]& ltokens, vector[vector[int]]& rtokens,
                   double threshold):                                           
    print 'l size. : ', ltokens.size(), ' , r size : ', rtokens.size()          
    cdef vector[pair[int, int]] output_pairs
    cdef int i, n=rtokens.size()                                    
    cdef PositionIndex index
    index = PositionIndex(ltokens, threshold)                                 
                                                                                
    for i in prange(n, nogil=True):    
        fun(ltokens, rtokens[i], threshold, index)
    return output_pairs

cdef void fun(vector[vector[int]]& ltokens, vector[int]& tokens, double threshold, PositionIndex& index) nogil:              
    cdef omap[int, int] candidate_overlap, overlap_threshold_cache              
    cdef vector[pair[int, int]] candidates                                      
    cdef pair[int, int] cand, entry                                             
    cdef int k=0, j=0, m, prefix_length, cand_num_tokens, current_overlap, overlap_upper_bound
    cdef int size, size_lower_bound, size_upper_bound                       
    cdef double sim_score                                       
    m = tokens.size()                                                      
    prefix_length = <int>(m - ceil(threshold * m) + 1.0)                    
    size_lower_bound = <int>ceil(threshold * m)                             
    size_upper_bound = <int>floor(m / threshold)                            
                                                                                
    for size in range(size_lower_bound, size_upper_bound + 1):              
        overlap_threshold_cache[size] = <int>ceil((threshold / (1 + threshold)) * (size + m))
                                                                                
    for j in range(prefix_length):                                          
        candidates = index.index[tokens[j]]                                 
        for cand in candidates:                                             
            current_overlap = candidate_overlap[cand.first]                 
            if current_overlap != -1:                                       
                cand_num_tokens = index.size_vector[cand.first]             
                                                                                
                # only consider candidates satisfying the size filter       
                # condition.                                                
                if size_lower_bound <= cand_num_tokens <= size_upper_bound: 
                                                                                
                    if m - j <= cand_num_tokens - cand.second:              
                        overlap_upper_bound = m - j                         
                    else:                                                   
                        overlap_upper_bound = cand_num_tokens - cand.second 
                                                                                
                    # only consider candidates for which the overlap upper  
                    # bound is at least the required overlap.               
                    if (current_overlap + overlap_upper_bound >=            
                                overlap_threshold_cache[cand_num_tokens]):      
                        candidate_overlap[cand.first] = current_overlap + 1 
                    else:                                                   
                        candidate_overlap[cand.first] = -1                  
#        print i, candidate_overlap.size()                                      
    for entry in candidate_overlap:                                         
        if entry.second > 0:                                                
            k += 1                                                          
            sim_score = jaccard(ltokens[entry.first], tokens)           
            #print ltokens[entry.first], rtokens[i], entry.second, sim_score
            if sim_score > threshold:                                       
                k = k - 1 + 1                                               
#                    output_pairs.push_back(pair[int, int](entry.first, i))     
 

cdef extern from "PositionIndex.h" nogil:
    cdef cppclass PositionIndex nogil:
        PositionIndex()
        PositionIndex(vector[vector[int]]&, double&)
        omap[int, vector[pair[int, int]]] index
        int min_len, max_len
        vector[int] size_vector
        double threshold

cdef class PpositionIndex1:
    cdef readonly omap[int, vector[pair[int, int]]] index
    cdef readonly int min_len, max_len
    cdef readonly vector[int] size_vector
    cdef readonly double threshold

    def __init__(self, const vector[vector[int]]& token_vectors, double threshold):
        cdef vector[int] tokens                                        
        cdef int prefix_length, token, i, j, m, n=token_vectors.size(), min_len=100000, max_len=0             
        for i in range(n):                                                          
            tokens = token_vectors[i]                                               
            m = tokens.size()                                                       
            self.size_vector.push_back(m)                                                
            prefix_length = int(m - ceil(threshold * m) + 1.0)                             
            for j in range(prefix_length):                                          
                self.index[tokens[j]].push_back(pair[int, int](i, j))
            if m > max_len:
                max_len = m
            if m < min_len:
                min_len = m          
        self.threshold = threshold
        self.min_len = min_len
        self.max_len = max_len

cdef double jaccard(const vector[int]& tokens1, const vector[int]& tokens2) nogil:   
    cdef int i=0, j=0, size1 = tokens1.size(), size2 = tokens2.size()                                             
    cdef int sum_of_size = size1 + size2                                        
    if sum_of_size == 0:                                                        
        return 1.0                                                              
    if size1 == 0 or size2 == 0:                                                
        return 0.0                                                              
    cdef int overlap = 0                
    while i < size1 and j < size2:
        if tokens1[i] == tokens2[j]:
            overlap += 1
            i += 1
            j += 1
        elif tokens1[i] < tokens2[j]:
            i += 1
        else:
            j += 1                                      
    return (overlap * 1.0) / (sum_of_size - overlap)    
