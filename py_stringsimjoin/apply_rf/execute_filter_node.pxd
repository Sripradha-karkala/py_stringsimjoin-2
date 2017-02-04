
from libcpp.vector cimport vector                                               
from libcpp.string cimport string                                               
from libcpp.pair cimport pair                                                   
from libcpp cimport bool                                                        
from libcpp.map cimport map as omap                                             
                                                                                
from py_stringsimjoin.apply_rf.predicatecpp cimport Predicatecpp  


cdef vector[int] execute_filter_node(vector[pair[int, int]]& candset,           
                                     vector[int]& pair_ids,                     
                                     bool top_level_node,                       
                                     vector[string]& lstrings,                  
                                     vector[string]& rstrings,                  
                                     Predicatecpp predicate,                    
                                     int n_jobs, const string& working_dir,     
                                     bool use_cache,                            
                                     omap[string, vector[vector[int]]]& ltokens_cache,
                                     omap[string, vector[vector[int]]]& rtokens_cache)

cdef vector[int] execute_filters(vector[pair[int, int]]& candset,               
                                 vector[int]& pair_ids,                         
                                 bool top_level_node,                           
                                 vector[string]& lstrings,                      
                                 vector[string]& rstrings,                      
                                 vector[Predicatecpp] predicates,               
                                 int n_jobs, const string& working_dir,         
                                 bool use_cache,                                
                                 omap[string, vector[vector[int]]]& ltokens_cache,
                                 omap[string, vector[vector[int]]]& rtokens_cache)
