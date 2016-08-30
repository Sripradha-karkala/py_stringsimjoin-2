
def execute_plan(plan, ltable, rtable, l_key_attr, r_key_attr,                            
                 l_match_attr, r_match_attr, feature_table, n_jobs=1):

    for child_node in plan.root.children:
        execute_node(child_node, ltable, rtable, l_key_attr, r_key_attr,                            
             l_match_attr, r_match_attr, rf, feature_table, n_jobs=1)

def execute_node(node, ltable, rtable, l_key_attr, r_key_attr,                            
                 l_match_attr, r_match_attr, rf, feature_table, n_jobs=1):
    
