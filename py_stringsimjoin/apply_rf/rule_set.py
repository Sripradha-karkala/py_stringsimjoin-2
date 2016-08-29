
import pandas as pd

class RuleSet:                                                                  
    def __init__(self, rules=None):                                             
        if rules is None:                                                       
            self.rules = []                                                     
        else:                                                                   
            self.rules = rules                                                  
                                                                                
    def add_rule(self, rule):                                                   
        self.rules.append(rule)                                                 

    def set_cost(self, cost):                                                   
        self.cost = cost                                                        
                                                                                                                                                               
    def set_coverage(self, coverage):                                           
        self.coverage = coverage                                                
        return True       
 
    def apply_tables(self, ltable, rtable, l_key_attr, r_key_attr,              
                     l_match_attr, r_match_attr, n_jobs=1):                     
        rule_outputs = []                                                       
        for rule in self.rules:                                                 
            rule_outputs.append(rule.apply_tables(ltable, rtable,               
                                    l_key_attr, r_key_attr,                     
                                    l_match_attr, r_match_attr, n_jobs)[['l_id', 'r_id']])
        output_df = pd.concat(rule_outputs)                                     
        return output_df.drop_duplicates()    
