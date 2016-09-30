
from libcpp.vector cimport vector    
from libcpp.string cimport string                                           
from libcpp.pair cimport pair                                                   

cpdef vector[pair[int, int]] ed_join(vector[vector[int]]& ltokens, 
                                     vector[vector[int]]& rtokens,
                                     int qval, 
                                     double threshold,
                                     vector[string]& lstrings, 
                                     vector[string]& rstrings)
