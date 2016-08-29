
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
        self.root.add_child(plan_to_be_merged.root.children[0]) 
  

def generate_execution_plan(rule_sets):
    ex_plan = Plan()
    for rule_set in rule_sets:
        for rule in rule_set.rules:
            plan_for_rule = get_optimal_plan_for_rule(rule)
            ex_plan.merge_plan(plan_for_rule)
    return ex_plan  

def get_optimal_plan_for_rule(rule):
    valid_predicates = []
    invalid_predicates = []
    for predicate in rule.predicates:
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

            combined_cov = valid_predicates[i].coverage & prev_coverage
            pred_score = (1.0 - (sum(combined_coverage) /                     
                len(combined_coverage))) / valid_predicates[i].cost

            if pred_score > max_score:                                              
                max_score = pred_score                                              
                max_pred_index = i                                                  

        optimal_predicate_seq.append(valid_predicates[max_pred_index])                      
        selected_predicates[max_pred_index] = True                                  
        prev_coverage = prev_coverage & valid_predicates[max_pred_index].coverage

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
