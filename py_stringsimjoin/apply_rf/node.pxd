
from libcpp.string cimport string                                               
from libcpp.vector cimport vector                                               
                                                                                
from py_stringsimjoin.apply_rf.predicatecpp cimport Predicatecpp    


cdef extern from "node.h" nogil:                                                
    cdef cppclass Node nogil:                                                   
        Node()                                                                  
        Node(vector[Predicatecpp]&, string&, vector[Node]&)                     
        Node(vector[Predicatecpp]&, string&)                                    
        Node(string&)                                                           
        void add_child(Node)                                                    
        void remove_child(Node&)                                                
        void set_node_type(string&)                                             
        void set_tree_id(int)
        vector[Predicatecpp] predicates                                         
        string node_type                                                        
        vector[Node] children 
