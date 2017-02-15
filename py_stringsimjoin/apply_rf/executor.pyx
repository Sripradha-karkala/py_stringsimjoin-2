
import time
import random
import pandas as pd
from cython.parallel import prange                                              

from libcpp.vector cimport vector
from libcpp.set cimport set as oset
from libcpp.string cimport string
from libcpp.pair cimport pair
from libcpp.map cimport map as omap
from libcpp cimport bool                                                        
from libc.stdio cimport printf, fprintf, fopen, fclose, FILE, sprintf
from libc.stdlib cimport atoi                                                   

from py_stringsimjoin.apply_rf.predicate import Predicate                       
from py_stringsimjoin.apply_rf.execution_plan import get_predicate_dict
from py_stringsimjoin.apply_rf.tokenizers cimport tokenize, load_tok, tokenize_str
from py_stringsimjoin.apply_rf.set_sim_join cimport set_sim_join, set_sim_join_no_cache
from py_stringsimjoin.apply_rf.overlap_coefficient_join cimport ov_coeff_join, ov_coeff_join_no_cache                
from py_stringsimjoin.apply_rf.edit_distance_join cimport ed_join   
from py_stringsimjoin.apply_rf.sim_functions cimport cosine, dice, jaccard, overlap      
from py_stringsimjoin.apply_rf.utils cimport compfnptr, str_simfnptr, \
  token_simfnptr, get_comp_type, get_comparison_function, get_sim_type, \
  get_overlap_sim_function, get_str_sim_function, get_token_sim_function, \
  simfnptr_str, get_sim_function_str, overlap_simfnptr, get_tok_type
from py_stringsimjoin.apply_rf.predicatecpp cimport Predicatecpp                
from py_stringsimjoin.apply_rf.node cimport Node                                
from py_stringsimjoin.apply_rf.coverage cimport Coverage         
from py_stringsimjoin.apply_rf.rule cimport Rule                                
from py_stringsimjoin.apply_rf.tree cimport Tree  
from py_stringsimjoin.apply_rf.ex_plan cimport get_plans_for_rules, get_default_execution_plan, generate_ex_plan_for_stage2, compute_predicate_cost_and_coverage, extract_pos_rules_from_rf, generate_local_optimal_plans, generate_overall_plan  
from py_stringsimjoin.apply_rf.execute_join_node cimport execute_join_node
from py_stringsimjoin.apply_rf.execute_select_node cimport execute_select_node, execute_select_node_candset      
from py_stringsimjoin.apply_rf.execute_feature_node cimport execute_feature_node      
from py_stringsimjoin.apply_rf.execute_filter_node cimport execute_filter_node, execute_filters
      

cdef extern from "string.h" nogil:                                              
    char *strtok (char *inp_str, const char *delimiters)     

cdef void load_strings(data_path, attr, vector[string]& strings):
    df = pd.read_csv(data_path)
    convert_to_vector1(df[attr], strings)

def test_execute_rf(rf, feature_table, l1, l2, path1, attr1, path2, attr2, 
                    working_dir, n_jobs, py_reuse_flag, py_push_flag, 
                    tree_list, use_cache):
    start_time = time.time()                                                    
    cdef vector[Tree] trees, trees1, trees2                                     
    trees = extract_pos_rules_from_rf(rf, feature_table)                        
                                                                                
    cdef int i=0, num_total_trees = trees.size()      
    cdef bool reuse_flag, push_flag
    reuse_flag = py_reuse_flag
    push_flag = py_push_flag                                                                            
    cdef vector[string] lstrings, rstrings                                      
    load_strings(path1, attr1, lstrings)                                        
    load_strings(path2, attr2, rstrings)                                        
                                                                                
    cdef omap[string, Coverage] coverage                                        
    cdef omap[int, Coverage] tree_cov
    cdef vector[string] l, r                                                    
    for s in l1:                                                                
        l.push_back(lstrings[int(s)])                                       
    for s in l2:                                                                
        r.push_back(rstrings[int(s)])                                       
    print ' computing coverage'                                                  
    compute_predicate_cost_and_coverage(l, r, trees, coverage, tree_cov, n_jobs)                 
    cdef Node global_plan, join_node
    global_plan = get_default_execution_plan(trees, coverage, tree_cov, 
                                             l.size(), trees1, trees2,
                                             reuse_flag, push_flag, tree_list) 

    print 'num join nodes : ', global_plan.children.size()
    for join_node in global_plan.children:                                             
         print 'JOIN', join_node.predicates[0].pred_name
 
    tok_st = time.time()
    print 'tokenizing strings'                                                  
    tokenize_strings(trees, lstrings, rstrings, working_dir, n_jobs)                    
    print 'finished tokenizing. executing plan'                                 
    tok_time = time.time() - tok_st

    cdef omap[string, vector[vector[int]]] ltokens_cache, rtokens_cache         
    cdef oset[string] tokenizers
    cdef Tree tree
    cdef Rule rule
    cdef Predicatecpp predicate
    cdef string tok_type

    if use_cache:                                                               
        for tree in trees:
            for rule in tree.rules:
                for predicate in rule.predicates:
                    if predicate.sim_measure_type.compare('EDIT_DISTANCE') == 0:
                        tokenizers.insert('qg2_bag')
                        continue
                    tokenizers.insert(predicate.tokenizer_type)

        for tok_type in tokenizers:
            ltokens_cache[tok_type] = vector[vector[int]]()                            
            rtokens_cache[tok_type] = vector[vector[int]]()                            
            load_tok(tok_type, working_dir, ltokens_cache[tok_type], rtokens_cache[tok_type])

    cdef double join_time = 0.0
    stage1_st = time.time()
    execute_plan(global_plan, trees1, lstrings, rstrings, working_dir, n_jobs, 
                 use_cache, ltokens_cache, rtokens_cache)  
    stage1_time = time.time() - stage1_st
    

    cdef pair[vector[pair[int, int]], vector[int]] candset_votes                
    candset_votes = merge_candsets(num_total_trees, trees1,        
                                   working_dir)                                 
    cdef int sample_size = 5000                                                 
    print 'generating plan'
    cdef vector[Node] plans                                                     
    plans = generate_ex_plan_for_stage2(candset_votes,                    
                                        lstrings, rstrings,   
                                        trees2, sample_size, n_jobs, 
                                        push_flag)

    print 'executing remaining trees'                                           
    cdef int label = 1, num_trees_processed=trees1.size()                                                          
    i = 0
    stage2_st = time.time()                                                                       
    while candset_votes.first.size() > 0 and i < plans.size():                  
        candset_votes = execute_tree_plan(candset_votes, lstrings, rstrings, plans[i],
                                  num_total_trees, num_trees_processed, label,  
                                  n_jobs, working_dir, use_cache, 
                                  ltokens_cache, rtokens_cache)                          
        num_trees_processed += 1                                                
        label += 1                                                              
    stage2_time = time.time() - stage2_st
    
    print 'total time : ', time.time() - start_time   
    print 'tok time : ', tok_time
    print 'stage1 time : ', stage1_time
    print 'stage2 time : ', stage2_time

def naive_execute_rf(rf, feature_table, l1, l2, path1, attr1, path2, attr2,      
                     working_dir, n_jobs, py_reuse_flag, py_push_flag,           
                     tree_list, use_cache):                                      
    start_time = time.time()                                                    
    cdef vector[Tree] trees, trees1, trees2                                     
    trees = extract_pos_rules_from_rf(rf, feature_table)                        
                                                                                
    cdef int i=0,j, num_total_trees = trees.size()                                
    cdef bool reuse_flag, push_flag                                             
    reuse_flag = py_reuse_flag                                                  
    push_flag = py_push_flag                                                    
    cdef vector[string] lstrings, rstrings                                      
    load_strings(path1, attr1, lstrings)                                        
    load_strings(path2, attr2, rstrings)                                        
                                                                                
    cdef omap[string, Coverage] coverage                                        
    cdef omap[int, Coverage] tree_cov                                           
    cdef vector[string] l, r                                                    
    for s in l1:                                                                
        l.push_back(lstrings[int(s)])                                           
    for s in l2:                                                                
        r.push_back(rstrings[int(s)])                                           
    print ' computing coverage'                                                 
    compute_predicate_cost_and_coverage(l, r, trees, coverage, tree_cov, n_jobs)
    cdef omap[int, vector[Node]] plans
    global_plan = get_plans_for_rules(trees, coverage, tree_cov,         
                                             l.size(), trees1, trees2,          
                                             reuse_flag, push_flag, tree_list) 

    print 'tokenizing strings'                                                  
    tokenize_strings(trees, lstrings, rstrings, working_dir, n_jobs)            
    print 'finished tokenizing. executing plan' 
    cdef omap[string, vector[vector[int]]] ltokens_cache, rtokens_cache         
 
    cdef vector[int] trees2_index
    naive_rf_time = time.time() - start_time
    subset_rf_time = naive_rf_time
    sel_trees = {}
    for i in tree_list:
        sel_trees[i] = True
    for i in xrange(num_total_trees):
        start_time = time.time()
        for j in xrange(plans[i].size()):
            execute_plan(plans[i][j], trees, lstrings, rstrings, working_dir, n_jobs,  
                 use_cache, ltokens_cache, rtokens_cache)
        naive_rf_time += time.time() - start_time
        if sel_trees.get(i) is not None:
            subset_rf_time += time.time() - start_time
            trees1.push_back(trees[i])
        else:
            trees2_index.push_back(i)

    cdef pair[vector[pair[int, int]], vector[int]] candset_votes                
    start_time = time.time()
    candset_votes = merge_candsets(num_total_trees, trees1,                     
                                   working_dir)    
    print 'executing remaining trees'                                           
    cdef int label = 1, num_trees_processed=trees1.size()                       
    i = 0                                                                                                      
    while candset_votes.first.size() > 0 and i < trees2_index.size():
        for j in xrange(plans[trees2_index[i]].size()):                  
            candset_votes = execute_tree_plan(candset_votes, lstrings, rstrings, plans[trees2_index[i]][j],
                                  num_total_trees, num_trees_processed, label,  
                                  n_jobs, working_dir, use_cache,               
                                  ltokens_cache, rtokens_cache)                 
        num_trees_processed += 1                                                
        label += 1    

    print 'rem trees time : ', time.time() - start_time
    subset_rf_time += time.time() - start_time

    print 'naive rf time : ', naive_rf_time
    print 'subset rf time : ', subset_rf_time


cdef void execute_plan(Node& root, vector[Tree]& trees, vector[string]& lstrings, 
        vector[string]& rstrings, const string& working_dir, int n_jobs, 
        bool use_cache, 
        omap[string, vector[vector[int]]]& ltokens_cache,
        omap[string, vector[vector[int]]]& rtokens_cache):

    cdef pair[vector[pair[int, int]], vector[double]] candset
    print root.children.size(), root.children[0].children.size()

    cdef Node join_node, child_node, curr_node
    print root.node_type, root.predicates.size(), root.children.size()

    join_time = 0
    for join_node in root.children:
         print 'JOIN', join_node.predicates[0].sim_measure_type, \
               join_node.predicates[0].tokenizer_type, \
               join_node.predicates[0].comp_op, \
               join_node.predicates[0].threshold
         js = time.time()
         candset = execute_join_node(lstrings, rstrings, join_node.predicates[0], 
                                     n_jobs, working_dir, 
                                     use_cache, ltokens_cache, rtokens_cache)
         print 'join completed. starting subtree execution.'
         join_time += time.time() - js                                                 
         execute_join_subtree(candset.first, candset.second, lstrings, rstrings, 
                              join_node, n_jobs, working_dir, 
                              use_cache, ltokens_cache, rtokens_cache)
         print 'join subtree execution completed'
    print 'join time : ', join_time


cdef pair[vector[pair[int, int]], vector[int]] execute_join_subtree(               
                    vector[pair[int, int]]& candset,
                    vector[double]& feature_values,   
                    vector[string]& lstrings, vector[string]& rstrings,            
                    Node& join_subtree, int n_jobs, const string& working_dir,
                    bool use_cache,
                    omap[string, vector[vector[int]]]& ltokens_cache,                       
                    omap[string, vector[vector[int]]]& rtokens_cache):          
    cdef Node child_node, grand_child_node, curr_node                           
                                                                                
    cdef vector[pair[Node, int]] queue                                          
    cdef pair[Node, int] curr_entry                                             
    cdef vector[int] pair_ids, curr_pair_ids                   
                                                                                
    for child_node in join_subtree.children:                                            
        queue.push_back(pair[Node, int](child_node, -1))                        
                                                                                
    cdef omap[int, vector[int]] cached_pair_ids                                 
    cdef omap[int , int] cache_usage                                            
    cdef int curr_index = 0                                                     
    cdef bool top_level_node = False                                            
    cdef vector[double] curr_feature_values
                                                                                
    while queue.size() > 0:                                                     
        curr_entry = queue.back()                                               
        queue.pop_back();                                                       
        curr_node = curr_entry.first                                            

        top_level_node = False
                                                            
        if curr_entry.second == -1:                                             
            top_level_node = True                                               
        else:                                                                   
            pair_ids = cached_pair_ids[curr_entry.second]                       
            cache_usage[curr_entry.second] -= 1                                 
                                                                                
            if cache_usage[curr_entry.second]  == 0:                            
                cache_usage.erase(curr_entry.second)                            
                cached_pair_ids.erase(curr_entry.second)                        
       
        if top_level_node and curr_node.node_type.compare("SELECT") == 0:
            print 'SELECT', curr_node.predicates[0].sim_measure_type, \
                  curr_node.predicates[0].tokenizer_type, \
                  curr_node.predicates[0].comp_op, \
                  curr_node.predicates[0].threshold
            curr_pair_ids = execute_select_node_candset(candset.size(), 
                                    feature_values, curr_node.predicates[0])    
                                                                                
            for child_node in curr_node.children:                     
               queue.push_back(pair[Node, int](child_node, curr_index))
                                                                                
            cache_usage[curr_index] = curr_node.children.size()             
            cached_pair_ids[curr_index] = curr_pair_ids                        
            curr_index += 1
            continue                       

        while (curr_node.node_type.compare("OUTPUT") != 0 and                   
               curr_node.node_type.compare("FILTER") == 0 and                   
               curr_node.children.size() < 2):
            print 'FILTER', curr_node.predicates[0].sim_measure_type, \
                  curr_node.predicates[0].tokenizer_type, \
                  curr_node.predicates[0].comp_op, \
                  curr_node.predicates[0].threshold
            
            if curr_node.predicates.size() > 1:
                pair_ids = execute_filters(candset, pair_ids, top_level_node,
                                           lstrings, rstrings, 
                                           curr_node.predicates, n_jobs, 
                                           working_dir, use_cache,
                                           ltokens_cache, rtokens_cache)
            else:
                pair_ids = execute_filter_node(candset, pair_ids, 
                                               top_level_node,
                                               lstrings, rstrings,                 
                                               curr_node.predicates[0], n_jobs, 
                                               working_dir, use_cache,
                                               ltokens_cache, rtokens_cache)
            
            curr_node = curr_node.children[0]                                   
            top_level_node = False                                              
                                                                                
        if curr_node.node_type.compare("OUTPUT") == 0:
            if top_level_node:
                write_candset(candset, curr_node.tree_id, curr_node.rule_id,
                              working_dir)    
            else:
                write_candset_using_pair_ids(candset, pair_ids, 
                                             curr_node.tree_id, 
                                             curr_node.rule_id, working_dir)                                   
            continue                                                            
                                                                                
        if curr_node.node_type.compare("FEATURE") == 0:                         
           print 'FEATURE', curr_node.predicates[0].sim_measure_type            
           curr_feature_values = execute_feature_node(candset, pair_ids, 
                                                      top_level_node,
                                                      lstrings, rstrings,            
                                                      curr_node.predicates[0], 
                                                      n_jobs, working_dir, 
                                                      use_cache, ltokens_cache, 
                                                      rtokens_cache)                    
           for child_node in curr_node.children:                                
               print 'SELECT', child_node.predicates[0].sim_measure_type, \
                     child_node.predicates[0].tokenizer_type, \
                     child_node.predicates[0].comp_op, \
                     child_node.predicates[0].threshold
               if top_level_node:
                   curr_pair_ids = execute_select_node_candset(candset.size(), 
                                                       curr_feature_values,
                                                       child_node.predicates[0])   
               else:
                   curr_pair_ids = execute_select_node(pair_ids, curr_feature_values,    
                                                   child_node.predicates[0])    

               for grand_child_node in child_node.children:                     
                   queue.push_back(pair[Node, int](grand_child_node, curr_index))
                                                                                
               cache_usage[curr_index] = child_node.children.size()             
               cached_pair_ids[curr_index] = curr_pair_ids                      
               curr_index += 1                                                  
        elif curr_node.node_type.compare("FILTER") == 0:                        
            print 'FILTER', curr_node.predicates[0].sim_measure_type, \
                  curr_node.predicates[0].tokenizer_type, \
                  curr_node.predicates[0].comp_op, \
                  curr_node.predicates[0].threshold
            if curr_node.predicates.size() > 1:
                pair_ids = execute_filters(candset, pair_ids, top_level_node,
                                                lstrings, rstrings,
                                                curr_node.predicates, n_jobs, 
                                                working_dir, use_cache, 
                                                ltokens_cache, rtokens_cache)
            else:
                pair_ids = execute_filter_node(candset, pair_ids, top_level_node,
                                                lstrings, rstrings,
                                                curr_node.predicates[0], n_jobs, 
                                                working_dir, use_cache,
                                                ltokens_cache, rtokens_cache)

            for child_node in curr_node.children:                               
                queue.push_back(pair[Node, int](child_node, curr_index))        
                                                                                
            cache_usage[curr_index] = curr_node.children.size()                 
            cached_pair_ids[curr_index] = pair_ids                      
            curr_index += 1                                                     
            

cdef pair[vector[pair[int, int]], vector[int]] merge_candsets(
                                           int num_total_trees, 
                                           vector[Tree]& processed_trees,
                                           const string& working_dir):

    cdef int i=0
    cdef string string_pair
    cdef oset[string] curr_pairs
    cdef omap[string, int] merged_candset
    cdef pair[string, int] entry
    cdef Tree tree
    for tree in processed_trees:
        file_name = working_dir + "/tree_" + str(tree.tree_id)
        print file_name
        f = open(file_name, 'r')
        for line in f:
            curr_pairs.insert(line)
        f.close()
        for string_pair in curr_pairs:
            merged_candset[string_pair] += 1
        curr_pairs.clear()
    cnt = 0
    cdef vector[pair[int, int]] candset_to_be_processed, output_pairs
    cdef vector[int] votes, pair_id

    for entry in merged_candset:
        pair_id = split(entry.first)
        if <double>entry.second >= (<double>num_total_trees/2.0):
            output_pairs.push_back(pair[int, int](pair_id[0], pair_id[1]))
        else:
            candset_to_be_processed.push_back(pair[int, int](pair_id[0], 
                                                             pair_id[1]))
            votes.push_back(entry.second)      
        
    write_output_pairs(output_pairs, working_dir, 0)    

    return pair[vector[pair[int, int]], vector[int]](candset_to_be_processed, 
                                                     votes)


cdef void write_candset(vector[pair[int,int]]& candset, int tree_id, int rule_id, const string& working_dir):
    file_path = working_dir + "/tree_" + str(tree_id)
    f = open(file_path, 'a+')
    cdef pair[int, int] tuple_pair
    for tuple_pair in candset:
        s = str(tuple_pair.first) + ',' + str(tuple_pair.second)
        f.write(s + '\n') 
    f.close()

cdef void write_candset_using_pair_ids(vector[pair[int,int]]& candset, vector[int]& pair_ids, 
                                       int tree_id, int rule_id, const string& working_dir):
    file_path = working_dir + "/tree_" + str(tree_id) 
    f = open(file_path, 'a+')                                                   
    cdef pair[int, int] tuple_pair                                              
    cdef int pair_id
    for pair_id in pair_ids:
        tuple_pair = candset[pair_id]                                                  
        s = str(tuple_pair.first) + ',' + str(tuple_pair.second)                
        f.write(s + '\n')                                                       
    f.close()   

cdef void write_output_pairs(vector[pair[int,int]]& output_pairs, const string& working_dir, int label):
    file_path = working_dir + "/output_" + str(label)
    f = open(file_path, 'w')                                                    
    cdef pair[int, int] tuple_pair                                              
    for tuple_pair in output_pairs:                                                  
        s = str(tuple_pair.first) + ',' + str(tuple_pair.second)                
        f.write(s + '\n')                                                       
    f.close()       


cdef pair[vector[pair[int, int]], vector[int]] execute_tree_plan(
                    pair[vector[pair[int, int]], vector[int]]& candset_votes, 
                    vector[string]& lstrings, vector[string]& rstrings,
                    Node& plan, int num_total_trees, int num_trees_processed,
                    int label, int n_jobs, const string& working_dir,
                    bool use_cache,
                    omap[string, vector[vector[int]]]& ltokens_cache,                       
                    omap[string, vector[vector[int]]]& rtokens_cache):
    cdef Node child_node, grand_child_node, curr_node

    cdef vector[pair[Node, int]] queue
    cdef pair[Node, int] curr_entry
    cdef vector[int] pair_ids, curr_pair_ids, output_pair_ids

    for child_node in plan.children:
        queue.push_back(pair[Node, int](child_node, -1))   
   
    cdef omap[int, vector[int]] cached_pair_ids
    cdef omap[int , int] cache_usage
    cdef int curr_index = 0
    cdef bool top_level_node = False
    cdef vector[double] feature_values

    while queue.size() > 0:
        curr_entry = queue.back()
        queue.pop_back();
        curr_node = curr_entry.first                                        
        
        top_level_node = False

        if curr_entry.second == -1:
            top_level_node = True
        else:
            pair_ids = cached_pair_ids[curr_entry.second]
            cache_usage[curr_entry.second] -= 1
                
            if cache_usage[curr_entry.second]  == 0:
                cache_usage.erase(curr_entry.second)
                cached_pair_ids.erase(curr_entry.second)

        while (curr_node.node_type.compare("OUTPUT") != 0 and
               (curr_node.node_type.compare("FILTER") == 0 or 
                curr_node.node_type.compare("JOIN") == 0) and
               curr_node.children.size() < 2):
            print 'FILTER', curr_node.predicates[0].sim_measure_type, \
                  curr_node.predicates[0].tokenizer_type, \
                  curr_node.predicates[0].comp_op, \
                  curr_node.predicates[0].threshold
            if curr_node.predicates.size() > 1:
                pair_ids = execute_filters(candset_votes.first, pair_ids, 
                                           top_level_node,
                                           lstrings, rstrings,
                                           curr_node.predicates, n_jobs, 
                                           working_dir, use_cache,
                                           ltokens_cache, rtokens_cache)
            else:
                pair_ids = execute_filter_node(candset_votes.first, pair_ids, 
                                               top_level_node,
                                               lstrings, rstrings,
                                               curr_node.predicates[0], n_jobs, 
                                               working_dir, use_cache,
                                               ltokens_cache, rtokens_cache)

            curr_node = curr_node.children[0]
            top_level_node = False
        
        if curr_node.node_type.compare("OUTPUT") == 0:
            output_pair_ids.insert(output_pair_ids.end(), pair_ids.begin(), 
                                                          pair_ids.end())
            continue

        if curr_node.node_type.compare("FEATURE") == 0:
           print 'FEATURE', curr_node.predicates[0].sim_measure_type
           feature_values = execute_feature_node(candset_votes.first, pair_ids, 
                                                 top_level_node,
                                                 lstrings, rstrings,
                                                curr_node.predicates[0], n_jobs,
                                                working_dir, use_cache,
                                                ltokens_cache, rtokens_cache)
           for child_node in curr_node.children:
               print 'SELECT', child_node.predicates[0].sim_measure_type, \
                     child_node.predicates[0].tokenizer_type, \
                     child_node.predicates[0].comp_op, \
                     child_node.predicates[0].threshold

               if top_level_node:
                   curr_pair_ids = execute_select_node_candset(
                                    candset_votes.first.size(), feature_values,    
                                    child_node.predicates[0])  
               else:
                   curr_pair_ids = execute_select_node(pair_ids, feature_values, 
                                                   child_node.predicates[0])

               for grand_child_node in child_node.children:
                   queue.push_back(pair[Node, int](grand_child_node, curr_index))
              
               cache_usage[curr_index] = child_node.children.size()
               cached_pair_ids[curr_index] = curr_pair_ids                                 
               curr_index += 1         
        elif curr_node.node_type.compare("FILTER") == 0:
            print 'FILTER', curr_node.predicates[0].sim_measure_type, \
                  curr_node.predicates[0].tokenizer_type, \
                  curr_node.predicates[0].comp_op, \
                  curr_node.predicates[0].threshold
            if curr_node.predicates.size() > 1:
                pair_ids = execute_filters(candset_votes.first, pair_ids, 
                                           top_level_node, lstrings, rstrings,
                                           curr_node.predicates, n_jobs, 
                                           working_dir, use_cache,
                                           ltokens_cache, rtokens_cache)
            else:
                pair_ids = execute_filter_node(candset_votes.first, pair_ids, 
                                               top_level_node, lstrings, 
                                               rstrings, 
                                               curr_node.predicates[0], n_jobs, 
                                               working_dir, use_cache,
                                               ltokens_cache, rtokens_cache)

            for child_node in curr_node.children:
                queue.push_back(pair[Node, int](child_node, curr_index))

            cache_usage[curr_index] = curr_node.children.size()
            cached_pair_ids[curr_index] = pair_ids
            curr_index += 1

    cdef int pair_id
    for pair_id in output_pair_ids:
        candset_votes.second[pair_id] += 1
    
    cdef vector[pair[int, int]] next_candset, output_pairs
    cdef vector[int] next_votes
    cdef int curr_votes
    cdef double reqd_votes = (<double>num_total_trees)/2.0
    for i in xrange(candset_votes.second.size()):
        curr_votes = candset_votes.second[i]
        if curr_votes + num_total_trees - num_trees_processed - 1 < reqd_votes:
            continue
        if curr_votes >= reqd_votes:
            output_pairs.push_back(candset_votes.first[i])
        else:
            next_candset.push_back(candset_votes.first[i])
            next_votes.push_back(curr_votes)
 
    write_output_pairs(output_pairs, working_dir, label)                            
                                                                                
    return pair[vector[pair[int, int]], vector[int]](next_candset,   
                                                     next_votes)           
            

cdef vector[int] split(string inp_string) nogil:                                      
    cdef char* pch                                                              
    pch = strtok (<char*> inp_string.c_str(), ",")                              
    cdef vector[int] out_tokens                                                 
    while pch != NULL:                                                          
        out_tokens.push_back(atoi(pch))                                         
        pch = strtok (NULL, ",")                                                
    return out_tokens    

cdef void tokenize_strings(vector[Tree]& trees, vector[string]& lstrings, 
                      vector[string]& rstrings, const string& working_dir, int n_jobs):
    cdef oset[string] tokenizers
    cdef Tree tree
    cdef Rule rule
    cdef Predicatecpp predicate
    for tree in trees:
        for rule in tree.rules:
            for predicate in rule.predicates:
                if predicate.sim_measure_type.compare('EDIT_DISTANCE') == 0:
                    tokenizers.insert('qg2_bag')
                    continue 
                tokenizers.insert(predicate.tokenizer_type)
 
    cdef string tok_type
    for tok_type in tokenizers:
        tokenize(lstrings, rstrings, tok_type, working_dir, n_jobs)

def test_tok1(df1, attr1, df2, attr2):                                                         
    cdef vector[string] lstrings, rstrings                                                 
    convert_to_vector1(df1[attr1], lstrings)
    convert_to_vector1(df2[attr2], rstrings)
    tokenize(lstrings, rstrings, 'ws', 'gh', 4)                       
  
cdef void convert_to_vector1(string_col, vector[string]& string_vector):         
    for val in string_col:                                                      
        string_vector.push_back(str(val))   

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

def generate_tokens(ft, path1, attr1, path2, attr2, const string& working_dir):
    cdef oset[string] tokenizers                                                
    for idx, row in ft.iterrows():
        if row['sim_measure_type'] == 'EDIT_DISTANCE':
            tokenizers.insert('qg2_bag')
            continue
        tokenizers.insert(str(row['tokenizer_type']))

    cdef vector[string] lstrings, rstrings                                      
    load_strings(path1, attr1, lstrings)                                        
    load_strings(path2, attr2, rstrings)   

    cdef string tok_type                                                        
    for tok_type in tokenizers:                                                 
        tokenize(lstrings, rstrings, tok_type, working_dir, 4)    

def perform_join(path1, attr1, path2, attr2, tok_type, sim_type, threshold, const string& working_dir):
    cdef vector[vector[int]] ltokens, rtokens                                   
    cdef pair[vector[pair[int, int]], vector[double]] output   
    cdef pair[int, int] entry
    cdef vector[string] lstrings, rstrings                                      

    if sim_type == 'COSINE':
        load_tok(tok_type, working_dir, ltokens, rtokens)                           
        threshold = threshold - 0.0001                                              
        output = set_sim_join_no_cache(ltokens, rtokens, 0, threshold, 4)           
    elif sim_type == 'DICE':
        load_tok(tok_type, working_dir, ltokens, rtokens)                       
        threshold = threshold - 0.0001                                          
        output = set_sim_join_no_cache(ltokens, rtokens, 1, threshold, 4)   
    elif sim_type == 'JACCARD':
        load_tok(tok_type, working_dir, ltokens, rtokens)                       
        threshold = threshold - 0.0001                                          
        output = set_sim_join_no_cache(ltokens, rtokens, 2, threshold, 4)           
    elif sim_type == 'OVERLAP_COEFFICIENT':
        load_tok(tok_type, working_dir, ltokens, rtokens)                       
        threshold = threshold - 0.0001
        output = ov_coeff_join_no_cache(ltokens, rtokens, threshold, 4)
    elif sim_type == 'EDIT_DISTANCE':
        load_tok('qg2_bag', working_dir, ltokens, rtokens)
        load_strings(path1, attr1, lstrings)                                        
        load_strings(path2, attr2, rstrings)                         
        output = ed_join(ltokens, rtokens, 2, threshold, lstrings, rstrings, 4)
   
    output_pairs = []
    for i in xrange(output.first.size()):
        output_pairs.append([str(output.first[i].first) + ',' + str(output.first[i].second), output.second[i]])
    output_df = pd.DataFrame(output_pairs, columns=['pair_id', 'score'])
    return output_df
 
def test_jac(sim_type, threshold):
    st = time.time()
    print 'tokenizing'
    #test_tok1(df1, attr1, df2, attr2)
    print 'tokenizing done.'
    cdef vector[vector[int]] ltokens, rtokens
    cdef vector[pair[int, int]] output
    cdef pair[vector[pair[int, int]], vector[double]] output1
    load_tok('ws', 't5', ltokens, rtokens)
    print 'loaded tok'
    cdef int i
#    for i in xrange(50):
#        print 'i= ', i
#    if sim_type == 3:
    for i in xrange(50):
        output1 = ov_coeff_join_no_cache(ltokens, rtokens, threshold, 4)             
        print 'output size : ', output.size()                                       
#    else:
#        output1 = set_sim_join(ltokens, rtokens, sim_type, threshold)
#    print 'scores size : ', output1.second.size()

#    cdef pair[int, int] entry
#    for i in xrange(output1.first.size()):
#        if output1.first[i].first == 1954 and output1.first[i].second == 63847:
#            print 'sim score : ', output1.second[i]
    print 'time : ', time.time() - st

def test_ed(df1, attr1, df2, attr2, threshold):                      
    st = time.time()                                                            
    print 'tokenizing'                                                          
    cdef vector[string] lstrings, rstrings                                      
    convert_to_vector1(df1[attr1], lstrings)                                    
    convert_to_vector1(df2[attr2], rstrings)  
#    tokenize(lstrings, rstrings, 'qg2', 'gh1')                                    
    print 'tokenizing done.'                                                    
    cdef vector[vector[int]] ltokens, rtokens                                   
    load_tok('qg2', 't5', ltokens, rtokens)                                      
    print 'loaded tok'
    cdef pair[vector[pair[int, int]], vector[double]] output
    cdef int i
    for i in xrange(50):                                                          
        output = ed_join(ltokens, rtokens, 2, threshold, lstrings, rstrings, 4)            
        print 'output size : ', output.size()                                       
    print 'time : ', time.time() - st                                           

