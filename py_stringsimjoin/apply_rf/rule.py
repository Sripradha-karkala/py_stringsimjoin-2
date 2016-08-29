
import copy


class Rule:
    def __init__(self, predicates=None):
        if predicates is None:
            self.predicates = []
        else:
            self.predicates = predicates

    def add_predicate(self, predicate):
        self.predicates.append(predicate)

    def set_cost(self, cost):                                                   
        self.cost = cost                                                        
                                                                                
    def set_coverage(self, coverage):                                           
        self.coverage = coverage                                                
        return True       

    def apply_tables(self, ltable, rtable, l_key_attr, r_key_attr,
                     l_match_attr, r_match_attr, n_jobs=1):
        selected_pred = None
        sel_pred_index = -1
        rem_preds = map(lambda p: p, self.predicates) 
        for i in xrange(0, len(self.predicates)):
            p = self.predicates[i]
            if p.is_valid_join_predicate():
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


