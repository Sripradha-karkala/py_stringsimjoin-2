
from libcpp.vector cimport vector                                               
from libcpp.string cimport string
from libcpp.map cimport map as omap                                             

from py_stringsimjoin.apply_rf.tree cimport Tree                                
from py_stringsimjoin.apply_rf.rule cimport Rule
from py_stringsimjoin.apply_rf.node cimport Node                                
from py_stringsimjoin.apply_rf.coverage cimport Coverage    


cdef void compute_predicate_cost_and_coverage(vector[string]& lstrings, 
                                              vector[string]& rstrings, 
                                              vector[Tree]& trees, 
                                              omap[string, Coverage]& coverage)

cdef vector[Tree] extract_pos_rules_from_rf(rf, feature_table)
cdef void generate_local_optimal_plans(vector[Tree]& trees, omap[string, Coverage]& coverage, int sample_size, vector[Node]& plans)
cdef Node generate_overall_plan(vector[Node] plans)
