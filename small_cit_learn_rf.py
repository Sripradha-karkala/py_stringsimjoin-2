
import py_stringsimjoin as ssj
from py_stringsimjoin.labeler.labeler import *
from py_stringsimjoin.activelearner.active_learner import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import py_stringmatching as sm
# import pyximport;
# pyximport.install()
from py_stringsimjoin.apply_rf.sample import sample_cython
from py_stringsimjoin.sampler.sample import *
from py_stringsimjoin.feature.autofeaturegen import *
from py_stringsimjoin.feature.extractfeatures import *
from py_stringsimjoin.apply_rf.apply_rf import *
from py_stringsimjoin.apply_rf.estimate_parameters import *
from py_stringsimjoin.apply_rf.execution_plan import *
from py_stringsimjoin.apply_rf.extract_rules import *
from py_stringsimjoin.utils.tokenizers import *
from py_stringsimjoin.exampleselector.entropy_selector import *
from py_stringsimjoin.labeler.cli_labeler import *;

base_path = '/Users/sripradha/Documents/Sem3/independent_study/smurf_datasets/'
# gold_path = base_path + 'cit_gold.csv'
out_path = '/Users/sripradha/Documents/Sem3/independent_study/smurf_datasets/'

ldf=pd.read_csv(base_path + 'cit_A.csv')
rdf=pd.read_csv(base_path + 'cit_B.csv')

seed=pd.read_csv(base_path + 'citations_seeds.csv')

print ('Performing sampling..')
c=sample_cython(ldf, rdf, 'id', 'id', 'citation_string', 'citation_string',
                100000, 20, seed)
c.to_csv(out_path + 'sample.csv', index=False)
c = pd.read_csv(out_path + 'sample.csv')

# labeled_c = label_table_using_gold(c, 'l_id', 'r_id', gold_path)
# print ('number of positives (after inverted_index sampling) : ', sum(labeled_c['label']))

ft=get_features(['JACCARD', 'COSINE', 'DICE', 'OVERLAP_COEFFICIENT', 'EDIT_DISTANCE', 'LEFT_LENGTH', 'RIGHT_LENGTH', 'LENGTH_SUM', 'LENGTH_DIFF'])
#ft=get_features(['JACCARD', 'COSINE', 'DICE', 'OVERLAP_COEFFICIENT', 'EDIT_DISTANCE'])

print ('Extracting feature vectors..')
fvs = extract_feature_vecs(c, 'l_id','r_id',ldf,rdf,'id','id',
                           'citation_string','citation_string',ft,n_jobs=4)

rf=RandomForestClassifier(n_estimators=10, min_samples_leaf=3)
al=ActiveLearner(rf, EntropySelector, CliLabeler, 20, 20)
lp = al.learn(fvs, seed)

import pickle
pickle.dump(al.matcher, open(out_path + 'small_cit_rf_10_trees_5_min_samples_leaf_for_big_cit_all_feat', 'w'))
'''
sample = c.sample(10000)
l_id = []
r_id = []

for idx, row in sample.iterrows():
    l_id.append(row['l_id'])
    r_id.append(row['r_id'])

pickle.dump(l_id, open('small_cit_sample_l_ids_1_min_samp', 'w'))
pickle.dump(r_id, open('small_cit_sample_r_ids_1_min_samp', 'w'))
'''
