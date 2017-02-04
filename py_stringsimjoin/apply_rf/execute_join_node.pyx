
from libcpp.vector cimport vector                                               
from libcpp.string cimport string                                               
from libcpp.pair cimport pair                                                   
from libcpp.map cimport map as omap                                             
from libcpp cimport bool                                                        

from py_stringsimjoin.apply_rf.set_sim_join cimport set_sim_join, set_sim_join_no_cache
from py_stringsimjoin.apply_rf.overlap_coefficient_join cimport ov_coeff_join, ov_coeff_join_no_cache
from py_stringsimjoin.apply_rf.edit_distance_join cimport ed_join 
from py_stringsimjoin.apply_rf.predicatecpp cimport Predicatecpp                
from py_stringsimjoin.apply_rf.tokenizers cimport load_tok


cdef pair[vector[pair[int, int]], vector[double]] execute_join_node(            
                            vector[string]& lstrings, vector[string]& rstrings, 
                            Predicatecpp predicate, int n_jobs,                 
                            const string& working_dir, bool use_cache,
                            omap[string, vector[vector[int]]]& ltokens_cache,                       
                            omap[string, vector[vector[int]]]& rtokens_cache):           
    cdef vector[vector[int]] ltokens, rtokens                                   
                                                                                
    cdef pair[vector[pair[int, int]], vector[double]] output                    
                                                                                
    if predicate.sim_measure_type.compare('COSINE') == 0:                       
        if use_cache:
            output = set_sim_join_no_cache(ltokens_cache[predicate.tokenizer_type], 
                                           rtokens_cache[predicate.tokenizer_type], 
                                           0, predicate.threshold, n_jobs)
        else:
            load_tok(predicate.tokenizer_type, working_dir, ltokens, rtokens)       
            output = set_sim_join_no_cache(ltokens, rtokens, 0, predicate.threshold, n_jobs)
    elif predicate.sim_measure_type.compare('DICE') == 0:
        if use_cache:                                                           
            output = set_sim_join_no_cache(ltokens_cache[predicate.tokenizer_type], 
                                           rtokens_cache[predicate.tokenizer_type], 
                                           1, predicate.threshold, n_jobs)      
        else:                        
            load_tok(predicate.tokenizer_type, working_dir, ltokens, rtokens)       
            output = set_sim_join_no_cache(ltokens, rtokens, 1, predicate.threshold, n_jobs)
    elif predicate.sim_measure_type.compare('JACCARD') == 0:                    
        if use_cache:                                                           
            output = set_sim_join_no_cache(ltokens_cache[predicate.tokenizer_type], 
                                           rtokens_cache[predicate.tokenizer_type], 
                                           2, predicate.threshold, n_jobs)      
        else: 
            load_tok(predicate.tokenizer_type, working_dir, ltokens, rtokens)       
            output = set_sim_join_no_cache(ltokens, rtokens, 2, predicate.threshold, n_jobs)
    elif predicate.sim_measure_type.compare('OVERLAP_COEFFICIENT') == 0:        
        if use_cache:
            output = ov_coeff_join_no_cache(ltokens_cache[predicate.tokenizer_type],
                                            rtokens_cache[predicate.tokenizer_type], 
                                            predicate.threshold, n_jobs)
        else:
            load_tok(predicate.tokenizer_type, working_dir, ltokens, rtokens)       
            output = ov_coeff_join_no_cache(ltokens, rtokens, predicate.threshold, n_jobs)
    elif predicate.sim_measure_type.compare('EDIT_DISTANCE') == 0:
        if use_cache:
            output = ed_join(ltokens_cache['qg2_bag'], rtokens_cache['qg2_bag'], 
                             2, predicate.threshold,          
                             lstrings, rstrings, n_jobs)          
        else:                
            load_tok('qg2_bag', working_dir, ltokens, rtokens)                      
            output = ed_join(ltokens, rtokens, 2, predicate.threshold,              
                             lstrings, rstrings, n_jobs)                            
    return output   
