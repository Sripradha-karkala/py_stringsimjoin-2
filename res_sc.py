
import py_stringsimjoin as ssj
from py_stringsimjoin.labeler.labeler import *
from py_stringsimjoin.active_learner.active_learner import *
from sklearn.ensemble import RandomForestClassifier                             
from sklearn.linear_model import LogisticRegression
import pandas as pd
import py_stringmatching as sm
from py_stringsimjoin.sampler.sample import *
from py_stringsimjoin.feature.autofeaturegen import *
from py_stringsimjoin.feature.extractfeatures import *
from py_stringsimjoin.apply_rf.apply_rf import *
from py_stringsimjoin.apply_rf.estimate_parameters import *
from py_stringsimjoin.apply_rf.execution_plan import *
from py_stringsimjoin.apply_rf.extract_rules import *
from py_stringsimjoin.utils.tokenizers import *

ldf=pd.read_csv('http://pages.cs.wisc.edu/~anhai/data/corleone_data/restaurants/fodors.csv')
rdf=pd.read_csv('http://pages.cs.wisc.edu/~anhai/data/corleone_data/restaurants/zagats.csv')

ldf['match_col']=ldf['name']+ ' ' +ldf['addr']+ ' '+ldf['city']+' '+ldf['phone']+' '+ldf['type']+' '+ldf['class'].astype(str)
rdf['match_col']=rdf['name']+ ' ' +rdf['addr']+ ' '+rdf['city']+' '+rdf['phone']+' '+rdf['type']+' '+rdf['class'].astype(str)

seed=pd.read_csv('/scratch/resseed.csv')

print ('Performing sampling..')
c=sample_pairs(ldf,rdf,'id','id','match_col','match_col',1000,20,seed)

labeled_c = label_table_using_gold(c, 'l_id', 'r_id', '/scratch/resgold.csv')
print ('number of positives (after inverted_index sampling) : ', sum(labeled_c['label']))

ft=get_features()

print ('Extracting feature vectors..')
fvs = extract_feature_vecs(c, 'l_id','r_id',ldf,rdf,'id','id','match_col','match_col',ft,n_jobs=4)

print ('Computing feature costs..')
compute_feature_costs(c, 'l_id','r_id',ldf,rdf,'id','id','match_col','match_col',ft)

print ft

rf=RandomForestClassifier(5)
al=ActiveLearner(rf,20,20,'/scratch/resgold.csv',seed)
lp = al.learn(fvs, '_id', 'l_id','r_id')

rule_sets=extract_pos_rules_from_rf(al.matcher, ft)
for rule_set in rule_sets:
    rule_set._print()

compute_coverage(rule_sets, fvs)

output = apply_rulesets(ldf, rdf, 'id', 'id',                            
                        'match_col', 'match_col', rule_sets, n_jobs=4)
print len(output)
#compute_coverage(rule_sets, fvs)
#plan = generate_execution_plan(rule_sets)


