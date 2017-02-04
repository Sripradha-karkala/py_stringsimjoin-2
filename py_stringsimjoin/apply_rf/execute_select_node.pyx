
from libcpp.vector cimport vector                                               
                                                                                
from py_stringsimjoin.apply_rf.predicatecpp cimport Predicatecpp                
from py_stringsimjoin.apply_rf.utils cimport compfnptr, get_comp_type, \
  get_comparison_function    

cdef vector[int] execute_select_node(vector[int]& pair_ids,                     
                                     vector[double]& feature_values,            
                                     Predicatecpp& predicate):                  
    cdef vector[int] output_pair_ids                                            
    cdef int n = pair_ids.size(), pair_id, comp_type                            
                                                                                
    comp_type = get_comp_type(predicate.comp_op)                                
    cdef compfnptr comp_fn = get_comparison_function(comp_type)                 
                                                                                
    for i in xrange(n):                                                         
        if comp_fn(feature_values[i], predicate.threshold):                     
            output_pair_ids.push_back(pair_ids[i])                              
                                                                                
    return output_pair_ids                                                      
                                                                                
cdef vector[int] execute_select_node_candset(int n,                             
                                             vector[double]& feature_values,    
                                             Predicatecpp& predicate):          
    cdef vector[int] output_pair_ids                                            
    cdef pair_id, comp_type                                                     
                                                                                
    comp_type = get_comp_type(predicate.comp_op)                                
    cdef compfnptr comp_fn = get_comparison_function(comp_type)                 
                                                                                
    for pair_id in xrange(n):                                                   
        if comp_fn(feature_values[pair_id], predicate.threshold):               
            output_pair_ids.push_back(pair_id)                                  
                                                                                
    return output_pair_ids  
