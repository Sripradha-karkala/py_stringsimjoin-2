
from py_stringsimjoin.apply_rf.predicate import Predicate
from py_stringsimjoin.apply_rf.rule import Rule
from py_stringsimjoin.apply_rf.rule_set import RuleSet
from py_stringsimjoin.utils.generic_helper import COMP_OP_MAP

 
def extract_pos_rules_from_tree(tree, feature_table):                                         
    feature_names = list(feature_table.index)
    # Get the left, right trees and the threshold from the tree                 
    left = tree.tree_.children_left                                             
    right = tree.tree_.children_right                                           
    threshold = tree.tree_.threshold                                            
                                                                                
    # Get the features from the tree                                            
    features = [feature_names[i] for i in tree.tree_.feature]                   
    value = tree.tree_.value                                                    
                                                    
    rule_set = RuleSet()
                            
    def traverse(node, left, right, features, threshold, depth, cache):         
        if node == -1:                                                          
            return                                                              
        if threshold[node] != -2:                                               
            # node is not a leaf node
            feat_row = feature_table.ix[features[node]]
            p = Predicate(features[node],
                          feat_row['sim_measure_type'], 
                          feat_row['tokenizer_type'],
                          feat_row['sim_function'], 
                          feat_row['tokenizer'], '<=', threshold[node])                                           
            cache.insert(depth, p)   
            traverse(left[node], left, right, features, threshold, depth+1, cache)
            prev_pred = cache.pop(depth)
            feat_row = feature_table.ix[features[node]]                         
            p = Predicate(features[node],
                          feat_row['sim_measure_type'],                         
                          feat_row['tokenizer_type'],                           
                          feat_row['sim_function'],                             
                          feat_row['tokenizer'], '>', threshold[node])                                         
            cache.insert(depth, p)    
            traverse(right[node], left, right, features, threshold, depth+1, cache)
            prev_pred = cache.pop(depth)                                        
        else:                                                                   
            # node is a leaf node                                               
            if value[node][0][0] <= value[node][0][1]:
                rule_set.add_rule(Rule(cache[0:depth]))                                                                        
                print 'pos rule: ', cache[0:depth]                              
                                                                                
    traverse(0, left, right, features, threshold, 0, [])
    return rule_set 

def extract_pos_rules_from_rf(rf, feature_table):
    rule_sets = []                                                              
    for dt in rf.estimators_:                                                   
        rule_sets.append(extract_pos_rules_from_tree(dt, feature_table))
    return rule_sets
