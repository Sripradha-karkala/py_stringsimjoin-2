
from libcpp.vector cimport vector
from libcpp.set cimport set as oset
from libcpp.string cimport string
from libcpp.pair cimport pair
from libcpp.map cimport map as omap
from libc.stdio cimport printf, fprintf, fopen, fclose, FILE, sprintf

from py_stringsimjoin.apply_rf.execution_plan import get_predicate_dict
from py_stringsimjoin.apply_rf.tokenizers import tokenize
#from py_stringsimjoin.apply_rf import tokenizers

#cdef extern from "<algorithm>" namespace "std":
#    void std_sort "std::sort" [iter](iter first, iter last)

cdef void execute_plan(plan, rule_sets, vector[string]& lstrings, 
        vector[string]& rstrings, feature_table, const string& working_dir, n_jobs):
    tokenize_strings(plan, rule_sets, lstrings, rstrings, working_dir)
    queue = []
    queue.extend(plan.root.children)
    while len(queue) > 0:
        curr_node = queue.pop(0)
        execute_node(curr_node, lstrings, rstrings, feature_table, n_jobs)
        if curr_node.node_type == 'OUTPUT':
            continue
        queue.extend(curr_node.children)


cdef void execute_node(node, vector[string]& lstrings, vector[string]& rstrings,
                 feature_table, int n_jobs):
    if node.node_type == 'JOIN':
        return
    elif node.node_type == 'FILTER':
        return
    elif node.node_type == 'SELECT':
        return
    elif node.node_type == 'OUTPUT':
        return

cdef void tokenize_strings(plan, rule_sets, vector[string]& lstrings, 
                      vector[string]& rstrings, const string& working_dir):
    tokenizers = infer_tokenizers(plan, rule_sets)
    cdef string tok_type
    for tok_type in tokenizers:
        tokenize(lstrings, rstrings, tok_type, working_dir)

def test_tok1(df1, attr1, df2, attr2):                                                         
    cdef vector[string] lstrings, rstrings                                                 
    convert_to_vector1(df1[attr1], lstrings)
    convert_to_vector1(df2[attr2], rstrings)
    tokenize(lstrings, rstrings, 'ws', 'gh')                       

cdef void convert_to_vector1(string_col, vector[string]& string_vector):         
    for val in string_col:                                                      
        string_vector.push_back(val)   

cdef vector[string] infer_tokenizers(plan, rule_sets):
    cdef vector[string] tokenizers
    predicate_dict = get_predicate_dict(rule_sets)

    queue = []
    queue.extend(plan.root.children)
    cdef string s
    while len(queue) > 0:
        curr_node = queue.pop(0)

        if curr_node.node_type in ['JOIN', 'FEATURE', 'FILTER']:
            pred = predicate_dict.get(curr_node.predicate)
            s = pred.tokenizer_type
            tokenizers.push_back(s)
    
        if curr_node.node_type == 'OUTPUT':
            continue
        queue.extend(curr_node.children)
    return tokenizers

