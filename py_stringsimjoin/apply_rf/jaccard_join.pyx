
from libc.math cimport ceil, floor, round
from libcpp.vector cimport vector                                               
from libcpp.set cimport set as oset                                             
from libcpp.string cimport string                                               
from libcpp cimport bool                                                        
from libcpp.map cimport map as omap                                             
from libcpp.pair cimport pair     


cpdef vector[pair[int, int]] jaccard_join(const vector[vector[int]]& ltokens, const vector[vector[int]]& rtokens, 
                   double threshold):
    cdef vector[pair[int, int]] output_pairs
    cdef omap[int, int] candidate_overlap, overlap_threshold_cache
    cdef vector[pair[int, int]] candidates
    cdef pair[int, int] cand
    cdef int i, j, m, n=rtokens.size(), prefix_length, cand_num_tokens, current_overlap, overlap_upper_bound
    cdef int size, size_lower_bound, size_upper_bound
    cdef vector[int] tokens
    cdef PositionIndex index
    index = PositionIndex(ltokens, threshold)
    
    for i in range(n):
        tokens = rtokens[i]
        m = tokens.size()
        prefix_length = <int>(m - ceil(threshold * m) + 1.0)
        size_lower_bound = <int>ceil(threshold * m)
        size_upper_bound = <int>floor(m / threshold)

        for size in range(size_lower_bound, size_upper_bound + 1):
            overlap_threshold_cache[size] = <int>ceil((threshold / (1 + threshold)) * (size + m)) 
                     
        for j in range(prefix_length):
            candidates = index.index[tokens[i]]
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
                             
        for entry in candidate_overlap:
            if entry.second > 0:
                sim_score = jaccard(ltokens[entry.first], rtokens[i])
                if sim_score > threshold:
                    output_pairs.push_back(pair[int, int](entry.first, i))
        
        candidate_overlap.clear()
 

cdef class PositionIndex:
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

cdef double jaccard(const vector[int]& tokens1, const vector[int]& tokens2):   
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
