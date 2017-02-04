
from libcpp.vector cimport vector                                               
from libcpp.string cimport string                                               
from libcpp.pair cimport pair                                                   
from libcpp.map cimport map as omap                                            
from libcpp cimport bool                                                        
                                                                                
from py_stringsimjoin.apply_rf.predicatecpp cimport Predicatecpp   

cdef pair[vector[pair[int, int]], vector[double]] execute_join_node(            
                            vector[string]& lstrings, vector[string]& rstrings, 
                            Predicatecpp predicate, int n_jobs,                 
                            const string& working_dir, bool use_cache,          
                            omap[string, vector[vector[int]]]& ltokens_cache,   
                            omap[string, vector[vector[int]]]& rtokens_cache)
