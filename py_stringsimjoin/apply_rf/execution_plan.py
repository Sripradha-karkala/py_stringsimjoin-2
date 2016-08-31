
import copy

class Node:
    def __init__(self, node_type, predicate, parent=None):
        self.node_type = node_type
        self.predicate = predicate
        self.parent = parent
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

class Plan:
    def __init__(self):
        self.root = Node('ROOT', None)

    def merge_plan(self, plan_to_be_merged):
        curr_level_children = self.root.children
        node2 = plan_to_be_merged.root.children[0]
        while True:
            node_merged = False
            for child_node in curr_level_children:
                if nodes_can_be_merged(child_node, node2):
                    curr_threhsold = child_node.predicate.threshold            
                    if node2.node_type == 'JOIN':
                        if curr_threshold < node2.predicate.threshold:
                            new_select_node = Node('SELECT', node2.predicate, child_node)
                            new_select_node.add_child(node2.children[0])
                            child_node.add_child(new_select_node)
                    elif node2.node_type == 'FILTER':
                        if curr_threshold
                    break
                    node_merged = True
            if not node_merged:
                break
            node2 = new_node
            curr_level_children
        
            

        if not node_merged:
            self.root.add_child(plan_to_be_merged.root.children[0])                 
 

def nodes_can_be_merged(node1, node2):
    if node1.node_type != node2.node_type:
        return False
    if node1.predicate.feat_name != node2.predicate.feat_name:
        return False
    return are_comp_ops_compatible(node1.predicate.comp_op, 
                                   node2.predicate.comp_op)  

def are_comp_ops_compatible(comp_op1, comp_op2):
    if comp_op1 in ['<', '<='] and comp_op2 in ['>' '>=']:
        return False
    if comp_op1 in ['>', '>='] and comp_op2 in ['<', '<=']:
        return False
    return True        

def generate_execution_plan(rule_sets):
    ex_plan = Plan()
    for rule_set in rule_sets:
        for rule in rule_set.rules:
            plan_for_rule = get_optimal_plan_for_rule(rule)
            ex_plan.merge_plan(plan_for_rule)
    return ex_plan  

def get_optimal_plan_for_rule(rule):
    optimal_predicate_seq = get_optimal_predicate_seq(rule.predicates)

    plan = Plan()                                                               
    curr_node = plan.root                                                       
    join_pred = True        
    for predicate in optimal_predicate_seq:    
        if join_pred:
            new_node = Node('JOIN', predicate, curr_node)
            curr_node.add_child(new_node)
            curr_node = new_node
            join_pred = False
        else:
            new_node = Node('FILTER', predicate, curr_node)
            curr_node.add_child(new_node)
            curr_node = new_node
    curr_node.add_child(Node('OUTPUT', None, curr_node))
    return plan

def get_optimal_predicate_seq(predicates):
    valid_predicates = []                                                       
    invalid_predicates = []                                                     
    for predicate in predicates:                                           
        if predicate.is_valid_join_predicate():                                 
            valid_predicates.append(predicate)                                  
        else:                                                                   
            invalid_predicates.append(predicate)  

    optimal_predicate_seq = []                                                  
    selected_predicates = {}                                                    
    max_score = 0                                                               
    max_pred_index = -1                                                         
    prev_coverage = None                                                        
    for i in range(len(valid_predicates)):                                      
        pred_score = (1.0 - (sum(valid_predicates[i].coverage) /                
            len(valid_predicates[i].coverage))) / valid_predicates[i].cost      
                                                                                
        if pred_score > max_score:                                              
            max_score = pred_score                                              
            max_pred_index = i                                                  
                                                                                
    optimal_predicate_seq.append(valid_predicates[max_pred_index])              
    selected_predicates[max_pred_index] = True                                  
    prev_coverage = valid_predicates[max_pred_index].coverage                   
                                                                                
    while len(optimal_predicate_seq) != len(valid_predicates):                  
        max_score = -1                                                          
        max_pred_index = -1                                                     
                                                                                
        for i in range(len(valid_predicates)):                                  
            if selected_predicates.get(i) != None:                              
                continue                                                        
                                                                                
            combined_coverage = valid_predicates[i].coverage & prev_coverage         
            pred_score = (1.0 - (sum(combined_coverage) /                       
                len(combined_coverage))) / valid_predicates[i].cost             
            print pred_score, max_score, sum(prev_coverage), sum(combined_coverage)                                                                    
            if pred_score > max_score:                                          
                max_score = pred_score                                          
                max_pred_index = i                                              
                                                                                
        optimal_predicate_seq.append(valid_predicates[max_pred_index])          
        selected_predicates[max_pred_index] = True                              
        prev_coverage = prev_coverage & valid_predicates[max_pred_index].coverage

    optimal_predicate_seq.extend(invalid_predicates)
    return optimal_predicate_seq

def select_optimal_set_of_trees(rule_sets):
    num_trees = len(rule_sets)
    min_trees_to_apply = (num_trees / 2) + 1 
    min_score = 1000000
    min_subset_indices = None
    for comb in itertools.combinations(range(len(rule_sets)), min_trees_to_apply):
        score = compute_score_for_trees(map(lambda i: rule_sets[i], comb))            
        if score < min_score:
            min_score = score
            min_subset_indices = comb
    trees_to_apply_over_join = []
    trees_to_apply_over_candset = []
    for i in range(len(rule_sets)):
        if i in comb:
            trees_to_apply_over_join.append(rule_sets[i])
        else:
            trees_to_apply_over_candset.append(rule_sets[i])
    return (trees_to_apply_over_join, trees_to_apply_over_candset)

def compute_score_for_trees(rule_sets):
    return

def generate_greedy_execution_plan(rule_sets):
    naive_plan = generate_execution_plan(rule_sets)
    naive_plan_cost = compute_plan_cost(naive_plan.root, None)
    greedy_plan = Plan()
    max_reduction_pred = -1
    max_reduced_cost = naive_plan_cost
    predicate_dict = {}
    rule_dict = {}
    for rule_set in rule_sets:
        for rule in rule_set.rules:
            rule_dict[rule.name] = rule
            for predicate in rule.predicates:
                if predicate.is_valid_join_predicate():                
                    if pred_dict.get(predicate.feat_name) is None:
                        predicate_dict[predicate.feat_name] = []
                    predicate_dict[predicate.feat_name].append((rule_set.name,
                                                                rule.name,
                                                                predicate.name))
                    break
    
    for feat_name in predicate_dict.keys():
        rule_sets_copy = copy.deepcopy(rule_sets)
        new_rule_set
        for entry in predicate_dict.get(feat_name):
            new_rule_set = RuleSet()
                   
        

def compute_plan_cost(plan_node, coverage):
    if  plan_node.node_type == 'OUTPUT':
        return 0

    if plan_node.node_type == 'ROOT':
       cost = 0
       for child_node in plan_node.children:
           cost += compute_plan_cost(child_node, coverage)
       return cost       

    curr_coverage = plan_node.predicate.coverage
    if plan_node.parent.node_type != 'ROOT':
        curr_coverage = curr_coverage & coverage

    child_nodes_cost = 0
    for child_node in plan_node.children:
        child_nodes_cost += compute_plan_cost(child_node, curr_coverage)

    if plan_node.parent.node_type == 'ROOT':
        return plan_node.predicate.cost + child_nodes_cost
    else:
        sel = sum(coverage) / len(coverage)
        return sel * plan_node.predicate.cost + child_nodes_cost

def recursive_merge(plan_node, index):
    if plan_node.node_type == 'ROOT':
        for child_index in range(len(plan_node.children)):
            recursive_merge(plan_node.children, )
