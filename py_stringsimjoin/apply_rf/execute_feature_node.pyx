
from cython.parallel import prange                                              

from libcpp.vector cimport vector                                               
from libcpp.string cimport string                                               
from libcpp.pair cimport pair                                                   
from libcpp cimport bool
from libcpp.map cimport map as omap                                                                                                     
                                                                                
from py_stringsimjoin.apply_rf.predicatecpp cimport Predicatecpp                
from py_stringsimjoin.apply_rf.tokenizers cimport load_tok                      
from py_stringsimjoin.apply_rf.utils cimport str_simfnptr, token_simfnptr, \
  get_sim_type, get_str_sim_function, get_token_sim_function, \
  simfnptr_str, get_sim_function_str 

cdef vector[double] execute_feature_node(vector[pair[int, int]]& candset,       
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
                                                                                
    cdef int n, sim_type, i                                                     
                                                                                
    if top_level_node:                                                          
        n = candset.size()                                                      
    else:                                                                       
        n = pair_ids.size()                                                     
                                                                                
    cdef vector[double] feature_values = xrange(0, n)                           
                                                                                
    sim_type = get_sim_type(predicate.sim_measure_type)                         
    cdef token_simfnptr token_sim_fn                                            
    cdef str_simfnptr str_sim_fn                                                
    cdef pair[int, int] cand                                                    
                                                                                
    if predicate.is_tok_sim_measure:                                            
        token_sim_fn = get_token_sim_function(sim_type)                         
        if top_level_node:                                                      
            for i in prange(n, nogil=True, num_threads=n_jobs):                 
                cand = candset[i]                                               
                feature_values[i] = token_sim_fn(ltokens[cand.first],           
                                                 rtokens[cand.second])          
        else:                                                                   
            for i in prange(n, nogil=True, num_threads=n_jobs):                 
                cand = candset[pair_ids[i]]                                     
                feature_values[i] = token_sim_fn(ltokens[cand.first],           
                                                 rtokens[cand.second])          
    else:                                                                       
        str_sim_fn = get_str_sim_function(sim_type)
        if top_level_node:                                                      
            for i in prange(n, nogil=True, num_threads=n_jobs):                 
                cand = candset[i]                                               
                feature_values[i] = str_sim_fn(lstrings[cand.first],            
                                               rstrings[cand.second])           
        else:                                                                   
            for i in prange(n, nogil=True, num_threads=n_jobs):                 
                cand = candset[pair_ids[i]]                                     
                feature_values[i] = str_sim_fn(lstrings[cand.first],            
                                               rstrings[cand.second])           
    return feature_values   

                    
