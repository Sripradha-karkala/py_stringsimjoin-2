
import copy

import pandas as pd

from py_stringsimjoin.utils.generic_helper import COMP_OP_MAP
import py_stringsimjoin as ssj

class RuleSet:
    def __init__(self, rules=None):
        if rules is None:
            self.rules = []
        else:
            self.rules = rules

    def add_rule(self, rule):
        self.rules.append(rule)

    def apply_tables(self, ltable, rtable, l_key_attr, r_key_attr, 
                     l_match_attr, r_match_attr, n_jobs=1):
        rule_outputs = []
        for rule in self.rules:
            rule_outputs.append(rule.apply_tables(ltable, rtable, 
                                    l_key_attr, r_key_attr,
                                    l_match_attr, r_match_attr, n_jobs)[['l_id', 'r_id']])
        output_df = pd.concat(rule_outputs)
        return output_df.drop_duplicates()


def valid_predicate(predicate):
    if predicate.sim_measure_type in ['JACCARD', 'COSINE', 'DICE', 'OVERLAP',
                                      'OVERLAP_COEFFICIENT']:
        return predicate.comp_op in ['>', '>=', '=']
    elif predicate.sim_measure_type == 'EDIT_DISTANCE':
        return predicate.comp_op in ['<', '<=', '=']
    return False

class Rule:
    def __init__(self, predicates=None):
        if predicates is None:
            self.predicates = []
        else:
            self.predicates = predicates

    def add_predicate(self, predicate):
        self.predicates.append(predicate)

    def apply_tables(self, ltable, rtable, l_key_attr, r_key_attr,
                     l_match_attr, r_match_attr, n_jobs=1):
        selected_pred = None
        sel_pred_index = -1
        rem_preds = map(lambda p: p, self.predicates) 
        for i in xrange(0, len(self.predicates)):
            p = self.predicates[i]
            if valid_predicate(p):
                selected_pred=p
                sel_pred_index = i
                break
        del rem_preds[i]
        rem_rule = Rule(rem_preds)
        candset = selected_pred.apply_tables(ltable, rtable,
                                             l_key_attr, r_key_attr,
                                             l_match_attr, r_match_attr,
                                             n_jobs)
        ltable_copy = ltable.set_index(l_key_attr)
        rtable_copy = rtable.set_index(r_key_attr)
        return candset[candset.apply(lambda r: rem_rule.apply_pair(
            ltable_copy.ix[r['l_'+l_key_attr]][l_match_attr], 
            rtable_copy.ix[r['r_'+r_key_attr]][r_match_attr]), 1)]

    def apply_pair(self, string1, string2):
        for p in self.predicates:
            if not p.apply_pair(string1, string2):
                return False
        return True


class Predicate:
    def __init__(self, sim_measure_type, tokenizer_type, sim_function, 
                 tokenizer, comp_op, threshold):
        self.sim_measure_type = sim_measure_type
        self.tokenizer_type = tokenizer_type
        self.sim_function = sim_function
        self.tokenizer = tokenizer
        self.comp_op = comp_op
        self.threshold = threshold
        self.comp_fn = COMP_OP_MAP[self.comp_op]

    def apply_pair(self, string1, string2):
        val1 = string1
        val2 = string2
        if self.tokenizer_type is not None:
            val1 = self.tokenizer.tokenize(val1)
            val2 = self.tokenizer.tokenize(val2)
        return self.comp_fn(self.sim_function(val1, val2), self.threshold) 
        
    def apply_tables(self, ltable, rtable, l_key_attr, r_key_attr,              
                     l_match_attr, r_match_attr, n_jobs=1):
        if self.sim_measure_type == 'JACCARD':
            return ssj.jaccard_join(ltable, rtable, l_key_attr, r_key_attr,
                                    l_match_attr, r_match_attr, self.tokenizer, 
                                    self.threshold, comp_op=self.comp_op, 
                                    n_jobs=n_jobs)
        elif self.sim_measure_type == 'COSINE':
            return ssj.cosine_join(ltable, rtable, l_key_attr, r_key_attr,         
                                   l_match_attr, r_match_attr, self.tokenizer,     
                                   self.threshold, comp_op=self.comp_op, 
                                   n_jobs=n_jobs)   
        elif self.sim_measure_type == 'DICE':
            return ssj.dice_join(ltable, rtable, l_key_attr, r_key_attr,         
                                 l_match_attr, r_match_attr, self.tokenizer,     
                                 self.threshold, comp_op=self.comp_op, 
                                 n_jobs=n_jobs)   
        elif self.sim_measure_type == 'EDIT_DISTANCE':
            return ssj.edit_distance_join(ltable, rtable, 
                                          l_key_attr, r_key_attr,         
                                          l_match_attr, r_match_attr, 
                                          self.threshold, comp_op=self.comp_op, 
                                          n_jobs=n_jobs)   
        elif self.sim_measure_type == 'OVERLAP':
            return ssj.overlap_join(ltable, rtable, l_key_attr, r_key_attr,         
                                    l_match_attr, r_match_attr, self.tokenizer,     
                                    self.threshold, comp_op=self.comp_op, 
                                    n_jobs=n_jobs)   
        elif self.sim_measure_type == 'OVERLAP_COEFFICIENT':
            return ssj.overlap_coefficient_join(ltable, rtable, 
                                    l_key_attr, r_key_attr,         
                                    l_match_attr, r_match_attr, self.tokenizer,     
                                    self.threshold, comp_op=self.comp_op, 
                                    n_jobs=n_jobs)   
