
from cython.parallel import prange                                              

from libcpp.vector cimport vector                                               
from libcpp.string cimport string                                               
from libcpp.pair cimport pair                                                   
from libcpp cimport bool                                                        
from libcpp.map cimport map as omap                                             
                                                                                
from py_stringsimjoin.apply_rf.predicatecpp cimport Predicatecpp                
from py_stringsimjoin.apply_rf.tokenizers cimport load_tok   
from py_stringsimjoin.apply_rf.utils cimport compfnptr, str_simfnptr, \
  token_simfnptr, get_comp_type, get_comparison_function, get_sim_type, \
  get_overlap_sim_function, get_str_sim_function, get_token_sim_function, \
  simfnptr_str, get_sim_function_str, overlap_simfnptr, get_tok_type 
from py_stringsimjoin.apply_rf.sim_functions cimport overlap

cdef vector[int] execute_filter_node(vector[pair[int, int]]& candset,          
                                     vector[int]& pair_ids,                     
                                     bool top_level_node,                       
                                     vector[string]& lstrings,                  
                                     vector[string]& rstrings,                  
                                     Predicatecpp predicate,                    
                                     int n_jobs, const string& working_dir,
                                     bool use_cache,
                                     omap[string, vector[vector[int]]]& ltokens_cache,                       
                                     omap[string, vector[vector[int]]]& rtokens_cache):     
    cdef vector[vector[int]] ltokens, rtokens                                   
    if predicate.is_tok_sim_measure:                                            
        if use_cache:
            ltokens = ltokens_cache[predicate.tokenizer_type]
            rtokens = rtokens_cache[predicate.tokenizer_type]
        else:
            load_tok(predicate.tokenizer_type, working_dir, ltokens, rtokens)           
            print 'loaded tok: ', ltokens.size(), rtokens.size()
    cdef vector[pair[int, int]] partitions                                      
    cdef vector[int] final_output_pairs, part_pairs                             
    cdef vector[vector[int]] output_pairs                                       
    cdef int n, start=0, end, i                                                 
                                                                                
    if top_level_node:                                                          
        n = candset.size()                                                      
    else:                                                                       
        n = pair_ids.size()                                                     
                                                                                
    partition_size = <int>(<float> n / <float> n_jobs)                          
                                                                                
    for i in range(n_jobs):                                                     
        end = start + partition_size                                            
        if end > n or i == n_jobs - 1:                                          
            end = n                                                             
        partitions.push_back(pair[int, int](start, end))                        
                                                                                
        start = end                                                             
        output_pairs.push_back(vector[int]())                                   
                                                                                
    cdef int sim_type, comp_type                                                
                                                                                
    sim_type = get_sim_type(predicate.sim_measure_type)                         
    comp_type = get_comp_type(predicate.comp_op)                                
    print 'parallen begin'                                                      
    for i in prange(n_jobs, nogil=True):                                        
        execute_filter_node_part(partitions[i], candset, pair_ids, top_level_node,
                                  ltokens, rtokens, lstrings, rstrings,   
                                  predicate, sim_type, comp_type,               
                                  output_pairs[i])                       
    print 'parallen end'                                                        
    for part_pairs in output_pairs:                                             
        final_output_pairs.insert(final_output_pairs.end(), part_pairs.begin(), part_pairs.end())
                                                                                
    return final_output_pairs                                                   
                                                                                
cdef void execute_filter_node_part(pair[int, int] partition,                   
                                   vector[pair[int, int]]& candset,             
                                   vector[int]& pair_ids,                       
                                   bool top_level_node,                         
                                   vector[vector[int]]& ltokens,                
                                   vector[vector[int]]& rtokens,                
                                   vector[string]& lstrings,                    
                                   vector[string]& rstrings,                    
                                   Predicatecpp& predicate,                     
                                   int sim_type, int comp_type,                 
                                   vector[int]& output_pairs) nogil:                         
                                                                                
    cdef pair[int, int] cand                                                    
    cdef int i                                                                  
                                                                                
    cdef str_simfnptr str_sim_fn                                                
    cdef token_simfnptr token_sim_fn                                            
    cdef compfnptr comp_fn = get_comparison_function(comp_type)                 
                                                                                
    if predicate.is_tok_sim_measure:                                            
        token_sim_fn = get_token_sim_function(sim_type)                         
        if top_level_node:                                                      
            for i in range(partition.first, partition.second):                  
                cand  = candset[i]                                              
                if comp_fn(token_sim_fn(ltokens[cand.first], rtokens[cand.second]),
                           predicate.threshold):                                
                    output_pairs.push_back(i)                                   
        else:                                                                   
            for i in range(partition.first, partition.second):                  
                cand  = candset[pair_ids[i]]                                    
                if comp_fn(token_sim_fn(ltokens[cand.first], rtokens[cand.second]),
                           predicate.threshold):                                
                    output_pairs.push_back(pair_ids[i])                         
    else:                                                                       
        str_sim_fn = get_str_sim_function(sim_type)                             
        if top_level_node:                                          
            for i in range(partition.first, partition.second):                  
                cand  = candset[i]                                              
                if comp_fn(str_sim_fn(lstrings[cand.first], rstrings[cand.second]),
                           predicate.threshold):                                
                    output_pairs.push_back(i)                                   
        else:                                                                   
            for i in range(partition.first, partition.second):                  
                cand  = candset[pair_ids[i]]                                    
                if comp_fn(str_sim_fn(lstrings[cand.first], rstrings[cand.second]),
                           predicate.threshold):                                
                    output_pairs.push_back(pair_ids[i])   


cdef vector[int] execute_filters(vector[pair[int, int]]& candset,               
                                 vector[int]& pair_ids,                         
                                 bool top_level_node,                           
                                 vector[string]& lstrings,                      
                                 vector[string]& rstrings,                      
                                 vector[Predicatecpp] predicates,               
                                 int n_jobs, const string& working_dir,
                                 bool use_cache,
                                 omap[string, vector[vector[int]]]& ltokens_cache,                       
                                 omap[string, vector[vector[int]]]& rtokens_cache):         
    cdef vector[vector[int]] ltokens, rtokens                                   
    if predicates[0].is_tok_sim_measure:                                        
        if use_cache:                                                           
            ltokens = ltokens_cache[predicates[0].tokenizer_type]                   
            rtokens = rtokens_cache[predicates[0].tokenizer_type]                   
        else:   
            load_tok(predicates[0].tokenizer_type, working_dir, ltokens, rtokens)       
    cdef vector[pair[int, int]] partitions                                      
    cdef vector[int] final_output_pairs, part_pairs                             
    cdef vector[vector[int]] output_pairs                                       
    cdef int n, start=0, end, i
                                                                                
    if top_level_node:                                                          
        n = candset.size()                                                      
    else:                                                                       
        n = pair_ids.size()                                                     
                                                                                
    partition_size = <int>(<float> n / <float> n_jobs)                          
                                                                                
    for i in range(n_jobs):                                                     
        end = start + partition_size                                            
        if end > n or i == n_jobs - 1:                                          
            end = n                                                             
        partitions.push_back(pair[int, int](start, end))                        
                                                                                
        start = end                                                             
        output_pairs.push_back(vector[int]())                                   
                                                                                
    cdef vector[int] sim_types, comp_types                                      
                                                                                
    for i in range(predicates.size()):                                          
        sim_types.push_back(get_sim_type(predicates[i].sim_measure_type))       
        comp_types.push_back(get_comp_type(predicates[i].comp_op))              
                                                                                
    print 'parallen begin'                                                      
    for i in prange(n_jobs, nogil=True):                                        
        execute_filters_part(partitions[i], candset, pair_ids, top_level_node,  
                             ltokens, rtokens, lstrings, rstrings,              
                             predicates, sim_types, comp_types, output_pairs[i])
    print 'parallen end'                                                        
                                                                                
    for i in range(n_jobs):                                                     
        final_output_pairs.insert(final_output_pairs.end(),                     
                                  output_pairs[i].begin(),                      
                                  output_pairs[i].end())                        
                                                                                
    return final_output_pairs                                                   
                                                                                
cdef void execute_filters_part(pair[int, int] partition,                        
                                   vector[pair[int, int]]& candset,             
                                   vector[int]& pair_ids,                       
                                   bool top_level_node,                         
                                   vector[vector[int]]& ltokens,                
                                   vector[vector[int]]& rtokens,                
                                   vector[string]& lstrings,                    
                                   vector[string]& rstrings,                    
                                   vector[Predicatecpp]& predicates,            
                                   vector[int]& sim_types, vector[int]& comp_types,
                               vector[int]& output_pairs) nogil:                
                                                                                
    cdef pair[int, int] cand                                                    
    cdef int i, j, size1, size2                                                 
    cdef double overlap_score                                                   
    cdef bool flag                                                              
    cdef vector[overlap_simfnptr] token_sim_fns                                 
    cdef vector[compfnptr] comp_fns                                             
                                                                                
    for i in range(sim_types.size()):                                           
        token_sim_fns.push_back(get_overlap_sim_function(sim_types[i]))         
        comp_fns.push_back(get_comparison_function(comp_types[i]))              
                                                                                
    if top_level_node:                                                          
        for i in range(partition.first, partition.second):                    
            cand  = candset[i]                                                  
            overlap_score = overlap(ltokens[cand.first], rtokens[cand.second])  
            size1 = ltokens[cand.first].size()                                  
            size2 = rtokens[cand.second].size()                                 
            flag = True                                                         
            for j in xrange(sim_types.size()):                                  
                if not comp_fns[j](token_sim_fns[j](size1, size2, overlap_score),
                                   predicates[j].threshold):                    
                    flag = False                                                
                    break                                                       
            if flag:                                                            
                output_pairs.push_back(i)                                       
    else:                                                                       
        for i in range(partition.first, partition.second):                      
            cand  = candset[pair_ids[i]]                                        
            overlap_score = overlap(ltokens[cand.first], rtokens[cand.second])  
            size1 = ltokens[cand.first].size()                                  
            size2 = rtokens[cand.second].size()                                 
            flag = True                                                         
            for j in xrange(sim_types.size()):                                  
                if not comp_fns[j](token_sim_fns[j](size1, size2, overlap_score),
                                   predicates[j].threshold):                    
                    flag = False                                                
                    break                                                       
            if flag:                                                            
                output_pairs.push_back(i)  

