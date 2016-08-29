
def compute_coverage(rule_sets, fvs):
    for rule_set in rule_sets:
        ruleset_cov = None
        first_rule = True
        for rule in rule_set.rules:
            rule_cov = None                                                         
            first_pred = True             
            for predicate in rule.predicates:
                pred_cov = fvs[predicate.feat_name].apply(lambda f: 
                               predicate.comp_fn(f, predicate.threshold), 1)
                predicate.set_coverage(pred_cov)
                if first_pred:
                    rule_cov = pred_cov
                    first_pred = False
                else:
                    rule_cov = rule_cov & pred_cov
            rule.set_coverage(rule_cov)
            if first_rule:
                ruleset_cov = rule_cov
                first_rule = False
            else:
                ruleset_cov = ruleset_cov | rule_cov
        rule_set.set_coverage(ruleset_cov)                 
