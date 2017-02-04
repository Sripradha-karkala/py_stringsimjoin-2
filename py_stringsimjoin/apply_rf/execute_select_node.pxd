
from libcpp.vector cimport vector                                               
                                                                                
from py_stringsimjoin.apply_rf.predicatecpp cimport Predicatecpp   

cdef vector[int] execute_select_node(vector[int]& pair_ids,                     
                                     vector[double]& feature_values,            
                                     Predicatecpp& predicate)

cdef vector[int] execute_select_node_candset(int n,                             
                                             vector[double]& feature_values,    
                                             Predicatecpp& predicate)
