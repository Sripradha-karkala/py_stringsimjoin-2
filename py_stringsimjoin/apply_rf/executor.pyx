
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libc.stdio cimport printf, fprintf, fopen, fclose, FILE, sprintf

from py_stringsimjoin.apply_rf.execution_plan import get_predicate_dict
cimport tokenizers
import tokenizers

cdef void execute_plan(plan, rule_sets, const vector[string]& lstrings, 
        const vector[string]& rstrings, feature_table, const string& working_dir, n_jobs):
    tokenize_strings(plan, rule_sets, lstrings, rstrings, working_dir)
    queue = []
    queue.extend(plan.root.children)
    while len(queue) > 0:
        curr_node = queue.pop(0)
        execute_node(curr_node, lstrings, rstrings, feature_table, n_jobs)
        if curr_node.node_type == 'OUTPUT':
            continue
        queue.extend(curr_node.children)


cdef void execute_node(node, const vector[string]& lstrings, const vector[string]& rstrings,
                 feature_table, int n_jobs):
    if node.node_type == 'JOIN':
        return
    elif node.node_type == 'FILTER':
        return
    elif node.node_type == 'SELECT':
        return
    elif node.node_type == 'OUTPUT':
        return

cdef void tokenize_strings(plan, rule_sets, const vector[string]& lstrings, 
                      const vector[string]& rstrings, const string& working_dir):
    tokenizers = infer_tokenizers(plan, rule_sets)

    tok = None                                                                  
    if tok_type.compare('ws') == 0:                                             
        tok = tokenizers.WhitespaceTokenizer(True)                              
    elif tok_type.compare('alph') == 0:                                         
        tok = tokenizers.AlphabeticTokenizer(True)                              
    elif tok_type.compare('alph_num') == 0:                                     
        tok = tokenizers.AlphanumericTokenizer(True)                            
    elif tok_type.compare('num') == 0:                                          
        tok = tokenizers.NumericTokenizer(True)                                 
    elif tok_type.compare('qg2') == 0:                                          
        tok = tokenizers.QgramTokenizer(2, True, ord('#'), ord('$'), True)      
    elif tok_type.compare('qg3') == 0:                                          
        tok = tokenizers.QgramTokenizer(3, True, ord('#'), ord('$'), True)      
    cdef string token
    cdef vector[string] tokens                                                  
    cdef umap[string, int] token_freq, token_ordering                                           
    cdef vector[vector[string]] ltokens, rtokens                                         
    for s in lstrings:
        tokens = tok.tokenize(s)
        ltokens.push_back(tokens)
        for token in tokens:
            token_freq[token]++
    for s in rstrings:                                                          
        tokens = tok.tokenize(s)                                                
        rtokens.push_back(tokens)                                               
        for token in tokens:                                                    
            token_freq[token]++                                                             
    token_ordering 
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

