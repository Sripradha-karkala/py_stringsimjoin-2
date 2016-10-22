
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

ldf=pd.read_csv('/scratch/dblp_rahm.csv')
rdf=pd.read_csv('/scratch/google_scholar.csv')

l_match_col = []
for idx, row in ldf.iterrows():
    val = ''
    if pd.notnull(row['title']):
        val += row['title'].lower()
    if pd.notnull(row['authors']):                                                
        val += row['authors'].lower()    
    if pd.notnull(row['venue']):                                                
        val += row['venue'].lower()    
    if pd.notnull(row['year']):                                                
        val += str(row['year'])
    l_match_col.append(val)

r_match_col = []                                                                
for idx, row in rdf.iterrows():                                                 
    val = ''                                                                    
    if pd.notnull(row['title']):                                                
        val += row['title'].lower()                                             
    if pd.notnull(row['authors']):                                              
        val += row['authors'].lower()                                           
    if pd.notnull(row['venue']):                                                
        val += row['venue'].lower()                                             
    if pd.notnull(row['year']):                                                 
        val += str(row['year'])                                                 
    r_match_col.append(val)     
     
ldf['match_col']=l_match_col
rdf['match_col']=r_match_col

seed=pd.read_csv('/scratch/citseed.csv')

print ('Performing sampling..')
c=sample_pairs(ldf,rdf,'id','id','match_col','match_col',100000,20,seed)

labeled_c = label_table_using_gold(c, 'l_id', 'r_id', '/scratch/citgold.csv')
print ('number of positives (after inverted_index sampling) : ', sum(labeled_c['label']))

ft=get_features(['JACCARD', 'COSINE', 'DICE'])

print ('Extracting feature vectors..')
fvs = extract_feature_vecs(c, 'l_id','r_id',ldf,rdf,'id','id','match_col','match_col',ft,n_jobs=4)

print ('Computing feature costs..')
compute_feature_costs(c, 'l_id','r_id',ldf,rdf,'id','id','match_col','match_col',ft)

print ft

rf=RandomForestClassifier(n_estimators=5, min_samples_leaf=3)
al=ActiveLearner(rf,20,20,'/scratch/citgold.csv',seed)
lp = al.learn(fvs, '_id', 'l_id','r_id')

rule_sets=extract_pos_rules_from_rf(al.matcher, ft)
for rule_set in rule_sets:
    rule_set._print()

compute_coverage(rule_sets, fvs)
import pickle
pickle.dump(rule_sets, open('cit_rule_sets_tree_5_leaf_3', 'w'))
pickle.dump(al.matcher, open('rf_5_3', 'w'))                
#output = apply_rulesets(ldf, rdf, 'id', 'id',                            
#                        'match_col', 'match_col', rule_sets, n_jobs=4)
#print len(output)
#compute_coverage(rule_sets, fvs)
#plan = generate_execution_plan(rule_sets)


