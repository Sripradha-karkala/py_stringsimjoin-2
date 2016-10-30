
import time

from libcpp cimport bool
from libcpp.map cimport map as omap
from libcpp.pair cimport pair
from libcpp.vector cimport vector                                               
from libcpp.set cimport set as oset                                             
from libcpp.string cimport string   

from py_stringsimjoin.apply_rf.predicate import Predicate                       
from py_stringsimjoin.apply_rf.tokenizers cimport tokenize_without_materializing
from py_stringsimjoin.apply_rf.utils cimport compfnptr, simfnptr, get_comp_type, get_comparison_function, get_sim_type, get_sim_function
from py_stringsimjoin.apply_rf.predicatecpp cimport Predicatecpp
from py_stringsimjoin.apply_rf.rule cimport Rule
from py_stringsimjoin.apply_rf.tree cimport Tree                                
from py_stringsimjoin.apply_rf.node cimport Node
from py_stringsimjoin.apply_rf.coverage cimport Coverage


cdef void compute_predicate_cost_and_coverage(vector[string]& lstrings, vector[string]& rstrings, vector[Tree]& trees, omap[string, Coverage]& coverage):
    cdef omap[string, vector[double]] features
    cdef omap[string, double] cost    
    cdef int sample_size = lstrings.size()
    cdef omap[string, pair[string, string]] feature_info
    cdef omap[string, vector[vector[int]]] ltokens, rtokens
    cdef Tree tree
    cdef Rule rule
    cdef Predicatecpp predicate
    cdef oset[string] tok_types
    cdef double sim_score, start_time, end_time
    cdef simfnptr sim_fn 
    cdef compfnptr comp_fn
    cdef omap[string, vector[double]] feature_values 

    for tree in trees:
        for rule in tree.rules:
            for predicate in rule.predicates:
                feature_info[predicate.feat_name] = pair[string, string](predicate.sim_measure_type, predicate.tokenizer_type)
                tok_types.insert(predicate.tokenizer_type)
    print 't1'
    for tok_type in tok_types:
        ltokens[tok_type] = vector[vector[int]]()
        rtokens[tok_type] = vector[vector[int]]()                               
        tokenize_without_materializing(lstrings, rstrings, tok_type, 
                                       ltokens[tok_type], rtokens[tok_type]) 
        
    print 't2'
    for feature in feature_info: 
        sim_fn = get_sim_function(get_sim_type(feature.second.first))
        cost[feature.first] = 0.0
        for i in xrange(sample_size):
            start_time = time.time()
            sim_score = sim_fn(ltokens[feature.second.second][i], 
                               rtokens[feature.second.second][i])
            end_time = time.time()
            cost[feature.first] += (end_time - start_time)
            feature_values[feature.first].push_back(sim_score)
        cost[feature.first] /= sample_size
     #   print feature.first, cost[feature.first]
    print 't3'
    cdef int max_size = 0
    cdef omap[string, vector[bool]] cov
    cdef int x, y, z
    for x in xrange(trees.size()):                                                          
        for y in xrange(trees[x].rules.size()):                                                 
            for z in xrange(trees[x].rules[y].predicates.size()):
                predicate = trees[x].rules[y].predicates[z]
                trees[x].rules[y].predicates[z].set_cost(cost[predicate.feat_name])
    #            print predicate.feat_name, cost[predicate.feat_name], predicate.cost, trees[x].rules[y].predicates[z].cost                                
                if cov[predicate.pred_name].size() == 0:
                    comp_fn = get_comparison_function(get_comp_type(predicate.comp_op))         
                    for i in xrange(sample_size):
                        cov[predicate.pred_name].push_back(comp_fn(feature_values[predicate.feat_name][i], predicate.threshold))
                    if cov[predicate.pred_name].size() > max_size:
                        max_size = cov[predicate.pred_name].size()
                    coverage[predicate.pred_name] = Coverage(cov[predicate.pred_name])
    print 'max size ', max_size
#    cdef Optimizer op = Optimizer(trees, coverage) 

cdef void generate_local_optimal_plans(vector[Tree]& trees, omap[string, Coverage]& coverage, int sample_size, vector[Node]& plans):
    cdef Tree tree
    cdef Rule rule
    cdef vector[int] optimal_seq
    cdef vector[Node] nodes
    cdef Node root, new_node, curr_node
    cdef string node_type
    cdef int i, rule_id, tree_id = 0
    cdef bool join_pred

    for tree in trees:
        rule_id = 0
        for rule in tree.rules:
            nodes = vector[Node]()
            optimal_seq = get_optimal_predicate_seq(rule.predicates, coverage, sample_size)
            node_type = "ROOT"
            nodes.push_back(Node(node_type))
            join_pred = True

            for i in optimal_seq:
                node_type = "FILTER"
                if join_pred:
                    node_type = "JOIN"
                    join_pred = False

                new_node = Node(node_type)
                new_node.predicates.push_back(rule.predicates[i])
                nodes.push_back(new_node)
    
            node_type = "OUTPUT"
            new_node = Node(node_type)
            new_node.set_tree_id(tree_id)
            new_node.set_rule_id(rule_id)
            nodes.push_back(new_node)
            print 'n ', nodes.size()
            for i in xrange(nodes.size() - 2, -1, -1):
                nodes[i].add_child(nodes[i+1])
            plans.push_back(nodes[0])
            rule_id += 1
        tree_id += 1
            
cdef vector[int] get_optimal_predicate_seq(vector[Predicatecpp]& predicates,
                                           omap[string, Coverage]& coverage,
                                           const int sample_size):                                      
    cdef vector[int] valid_predicates, invalid_predicates, optimal_seq
    cdef vector[bool] selected_predicates
    cdef Predicatecpp predicate
    cdef int i, max_pred, j, n=0
    cdef double max_score, pred_score
                                                  
    for i in xrange(predicates.size()):                                                
        if predicates[i].is_join_predicate():                                 
            valid_predicates.push_back(i)                                  
            n += 1
        else:                                                                   
            invalid_predicates.push_back(i)
        selected_predicates.push_back(False)
    
                                
    if n == 0:                                              
        print 'invalid rf'                                                      
                                                                                                                        
    max_score = 0.0                                                               
    max_pred = -1                                                         
    cdef Coverage prev_coverage
                                                        
    for i in valid_predicates:                                      
        pred_score = (1.0 - (coverage[predicates[i].pred_name].count / sample_size)) / predicates[i].cost     
                                                                                
        if pred_score > max_score:                                              
            max_score = pred_score                                              
            max_pred = i                                                  
                                                                                
    optimal_seq.push_back(max_pred)              
    selected_predicates[max_pred] = True                                  

    prev_coverage.or_coverage(coverage[predicates[max_pred].pred_name])                   
    
    j = 1                                                                            
    while j < n:                  
        max_score = -1                                                          
        max_pred = -1                                                     
                                                                                
        for i in valid_predicates:                                  
            if selected_predicates[i]:                              
                continue                                                        
                                                                               
            pred_score = (1.0 - (prev_coverage.and_sum(coverage[predicates[i].pred_name]) / sample_size)) / predicates[i].cost             
            print pred_score, max_score

            if pred_score > max_score:                                          
                max_score = pred_score                                          
                max_pred = i                                              
                                                                                
        optimal_seq.push_back(max_pred)          
        selected_predicates[max_pred] = True                              
        prev_coverage.and_coverage(coverage[predicates[max_pred].pred_name])
        j += 1 
                                                                                  
    optimal_seq.insert(optimal_seq.end(), invalid_predicates.begin(),
                       invalid_predicates.end())                            
    return optimal_seq         

cdef vector[Rule] extract_pos_rules_from_tree(d_tree, feature_table):
    feature_names = list(feature_table.index)                                   
    # Get the left, right trees and the threshold from the tree                 
    left = d_tree.tree_.children_left                                             
    right = d_tree.tree_.children_right                                           
    threshold = d_tree.tree_.threshold                                            
                                                                                
    # Get the features from the tree                                            
    features = [feature_names[i] for i in d_tree.tree_.feature]                   
    value = d_tree.tree_.value                                                    
                                                                                
    cdef vector[Rule] rules                                                        
    traverse(0, left, right, features, threshold, value, feature_table, 0, [], rules)                        

    return rules
                                                                                
cdef void traverse(node, left, right, features, threshold, value, feature_table, depth, cache, vector[Rule]& rules):
    if node == -1:                                                          
        return                                                              
    cdef vector[Predicatecpp] preds                                     
    cdef Predicatecpp pred   
    cdef Rule rule                                                      
    if threshold[node] != -2:                                               
            # node is not a leaf node                                           
        feat_row = feature_table.ix[features[node]]                         
        p = Predicate(features[node],                                       
                          feat_row['sim_measure_type'],                         
                          feat_row['tokenizer_type'],                           
                          feat_row['sim_function'],                             
                          feat_row['tokenizer'], '<=', threshold[node], 0)
#            p.set_name(features[node]+' <= '+str(threshold[node]))                                                         
        cache.insert(depth, p)                                              
        traverse(left[node], left, right, features, threshold, value, feature_table, depth+1, cache, rules)
        prev_pred = cache.pop(depth)                                        
        feat_row = feature_table.ix[features[node]]                         
        p = Predicate(features[node],                                       
                      feat_row['sim_measure_type'],                         
                      feat_row['tokenizer_type'],                           
                      feat_row['sim_function'],                             
                      feat_row['tokenizer'], '>', threshold[node], 0)
#            p.set_name(features[node]+' > '+str(threshold[node]))                                               
        cache.insert(depth, p)                                              
        traverse(right[node], left, right, features, threshold, value, feature_table, depth+1, cache, rules)
        prev_pred = cache.pop(depth)                                        
    else:                                                                   
            # node is a leaf node                                               
        if value[node][0][0] <= value[node][0][1]:
            pred_dict = {}
            for i in xrange(depth):
                if pred_dict.get(cache[i].feat_name+cache[i].comp_op) is None:
                    pred_dict[cache[i].feat_name+cache[i].comp_op] = i
                    continue
                if cache[i].comp_op == '<=':
                    if cache[i].threshold > cache[pred_dict[cache[i].feat_name+cache[i].comp_op]].threshold:
                        pred_dict[cache[i].feat_name+cache[i].comp_op] = i
                else:
                    if cache[i].threshold < cache[pred_dict[cache[i].feat_name+cache[i].comp_op]].threshold:
                        pred_dict[cache[i].feat_name+cache[i].comp_op] = i    

            for k in pred_dict.keys():
                i = pred_dict[k]
                pred_name = cache[i].feat_name+cache[i].comp_op+str(cache[i].threshold) 
                pred = Predicatecpp(pred_name, cache[i].feat_name, cache[i].sim_measure_type, cache[i].tokenizer_type, cache[i].comp_op, cache[i].threshold)                  
                preds.push_back(pred)
            rule = Rule(preds)                                        
#                r.set_name('r'+str(start_rule_id + len(rule_set.rules)+1))      
            rules.push_back(rule)                                            
            print 'pos rule: ', cache[0:depth]                              
                                                                                
cdef vector[Tree] extract_pos_rules_from_rf(rf, feature_table):                               
    cdef vector[Tree] trees
    cdef vector[Rule] rules
    cdef Tree tree                                                             
    rule_id = 1                                                                 
    predicate_id = 1                                                            
    tree_id = 1                                                                 
    for dt in rf.estimators_:                                                   
        rules = extract_pos_rules_from_tree(dt, feature_table)                                                              
        tree = Tree(rules)                                                          
#        rs.set_name('t'+str(tree_id))                                           
#        tree_id += 1                                                            
#        rule_id += tree.rules.size()                                             
        trees.push_back(tree)                                                    
    return trees


#def extract_rules(rf, feature_table, l1, l2):
#    cdef vector[Tree] trees
#    trees = extract_pos_rules_from_rf(rf, feature_table)
#    print 'num trees : ', trees.size()
#    num_rules = 0
#    num_preds = 0
#    cdef Tree tree
#    cdef Rule rule
#    for tree in trees:
#        num_rules += tree.rules.size()
#        for rule in tree.rules:
#            num_preds += rule.predicates.size()
#    print 'num rules : ', num_rules
#    print 'num preds : ', num_preds

#    cdef omap[string, Coverage] coverage                                    
#    cdef vector[string] l, r                                                    
#    for s in l1:                                                                
#        l.push_back(s)                                                          
#    for s in l2:                                                                
#        r.push_back(s)                                                          
#    compute_predicate_cost_and_coverage(l, r, trees, coverage)
#    cdef Predicatecpp pred
#    for pred in trees[0].rules[0].predicates:
#        print pred.pred_name, pred.cost
#    cdef vector[Node] plans
#    generate_local_optimal_plans(trees, coverage, l.size(), plans)
#    print 'num pl : ', plans.size()
#    cdef Node node
#    cdef int i = 0
#    while i < 2:
#        node = plans[i]
#        print 'test', i, node.node_type, node.children.size()
#        while True:
#            print 'hello'
#            if node.children.size() == 0:
#                break
#            if node.predicates.size() > 0:
#                print node.predicates[0].pred_name, node.node_type
#            else:
#                print node.node_type                 
#            node = node.children[0]
#        i += 1
#    generate_overall_plan(plans)
##    cdef pair[string, Coverage] entry
#    cdef bool x
#    for entry in coverage:
#        print entry.first, entry.second.size()

cdef Node merge_plans(Node plan1, Node plan2):                                 
    cdef Node plan2_node = plan2.children[0]                                         
    cdef Predicatecpp pred1, pred2
    cdef string node_type = "SELECT"
    pred2 = plan2_node.predicates[0]
    cdef int i

    while i < plan1.children.size():
        print 'sib : ', plan1.children[i].node_type                              
        if nodes_can_be_merged(plan1.children[i], plan2_node, 
                               plan1.children[i].predicates[0], pred2):
            break
        i += 1     

    if i == plan1.children.size():
        plan1.add_child(plan2.children[0])
        return plan1
 
    print 't1', plan2_node.node_type                                
    if plan2_node.node_type.compare("JOIN") == 0:                              
        pred1 = plan1.children[i].predicates[0]                                                                        
        if ((pred1.threshold < pred2.threshold) or                  
            (pred1.threshold == pred2.threshold and                 
             pred1.comp_op.compare(">=") == 0 and 
             pred2.comp_op.compare(">") == 0)):      
            print 't2'
            plan2_node.set_node_type(node_type)                         
            plan1.children[i].add_child(plan2_node)                      
                                                                                
        elif ((pred1.threshold > pred2.threshold) or                
              (pred1.threshold == pred2.threshold and               
               pred1.comp_op.compare(">") == 0 and 
               pred2.comp_op.compare(">=") == 0)):    
            print 't3'                                              
            plan1.children[i].set_node_type(node_type)                      
            plan2_node.add_child(plan1.children[i])        
            plan1.remove_child(plan1.children[i])              
            plan1.add_child(plan2_node)               
                                                                                
        elif pred1.threshold == pred2.threshold:                    
            print 't4'                                                      
            plan1.children[i].add_child(plan2_node.children[0])
    else:
        print 'invalid rf'                                                             

    return plan1

cdef bool nodes_can_be_merged(Node& node1, Node& node2, Predicatecpp& pred1, 
                              Predicatecpp& pred2):                            
    if node1.node_type.compare(node2.node_type) != 0:                                      
        return False                                                            
    if pred1.feat_name.compare(pred2.feat_name) != 0:                                      
        return False                                                            
    return are_comp_ops_compatible(pred1.comp_op, pred2.comp_op, node1.node_type)
                                                                                
cdef bool are_comp_ops_compatible(comp_op1, comp_op2, node_type):                     
    if node_type == "FILTER":                                                   
        return True                                                             
    if node_type == "SELECT":
        return comp_op1 == comp_op2                                
    if comp_op1 in ['<', '<='] and comp_op2 in ['>' '>=']:                      
        return False                                                            
    if comp_op1 in ['>', '>='] and comp_op2 in ['<', '<=']:                     
        return False                                                            
    return True 

cdef Node generate_overall_plan(vector[Node] plans):
    cdef Node combined_plan = plans[0]
    cdef int i=1
    print 'before merge size : ', combined_plan.children.size()
    while i < plans.size():
        combined_plan = merge_plans(combined_plan, plans[i])
        i += 1
        print 'i = ', i, ' , num child nodes : ', combined_plan.children.size()
    return combined_plan
