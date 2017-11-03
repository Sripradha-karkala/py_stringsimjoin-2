
import py_stringsimjoin as ssj
from py_stringsimjoin.labeler.labeler import *
from py_stringsimjoin.activelearner.active_learner import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import py_stringmatching as sm
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


def sample_get_instruction_fn():
    banner_str = 'Select if the below shown pair is a match or not \n'
    return banner_str
    
def sample_get_example_display_fn(example_to_label, context):
    # Each of the example to label is a tuple
    fvs_A_id_attr = 'l_id'
    fvs_B_id_attr = 'r_id'
    A_id_attr = 'id'
    B_id_attr = 'id'
    
    A_out_attrs = ['id', 'citation_string']
    B_out_attrs = ['id', 'citation_string']
    
    #obtaining the raw representation
    table_A_id = example_to_label[fvs_A_id_attr]
    table_B_id = example_to_label[fvs_B_id_attr]
    
    raw_tuple_table_A = context['table_A'].where(context['table_A'][A_id_attr] == table_A_id).dropna().head(1)
    raw_tuple_table_B = context['table_B'].where(context['table_B'][B_id_attr] == table_B_id).dropna().head(1)
    
    #'id', 'id', 'l_id', 'r_id', ['citation_string'], ['citation_string']
    return str(raw_tuple_table_A[A_out_attrs]) + "\n" + str(raw_tuple_table_B[B_out_attrs]) + "\n"


if __name__ == '__main__':
	base_path = 'smurf_datasets/'
	# gold_path = base_path + 'cit_gold.csv'
	out_path = 'smurf_datasets/'
	
	ldf=pd.read_csv(base_path + 'cit_A.csv')
	rdf=pd.read_csv(base_path + 'cit_B.csv')
	
	seed=pd.read_csv(base_path + 'citations_seeds.csv')
	
	print ('Performing sampling..')
	#c=sample_cython(ldf, rdf, 'id', 'id', 'citation_string', 'citation_string',
	#                100000, 20, seed)
	#c.to_csv(out_path + 'sample.csv', index=False)
	c = pd.read_csv(out_path + 'sample.csv')
	ft=get_features(['JACCARD'])
	#, 'COSINE', 'DICE', 'OVERLAP_COEFFICIENT', 'EDIT_DISTANCE', 'LEFT_LENGTH', 'RIGHT_LENGTH', 'LENGTH_SUM', 'LENGTH_DIFF'])
	#ft=get_features(['JACCARD', 'COSINE', 'DICE', 'OVERLAP_COEFFICIENT', 'EDIT_DISTANCE'])
	
	print ('Extracting feature vectors..')
	#fvs = extract_feature_vecs(c, 'l_id','r_id',ldf,rdf,'id','id',
	#                           'citation_string','citation_string',ft,n_jobs=4)
	                           
	#fvs.to_csv(out_path + 'feature_vector.csv', index=False) 
	fvs = pd.read_csv(out_path + 'feature_vector.csv')                     
	print 'generating the seed information...'
	fvs = fvs.set_index('_id')
	print "Random forest classifier"
	rf=RandomForestClassifier(n_estimators=10, min_samples_leaf=3)
	print "Initializing active learning..."
	example_selector = EntropySelector()
	
	print 'Defining a labeler...'
	lables = {'0':0, '1':1 }
	labeler = CliLabeler(sample_get_instruction_fn, sample_get_example_display_fn, lables)
	context = {"table_A": ldf, "table_B":rdf}
	al=ActiveLearner(rf, example_selector, labeler, 4, 2)
	print 'Calling learn function'
	lp = al.learn(fvs, seed, context=context)
	
	import pickle
	pickle.dump(al.model, open(out_path + 'small_cit_rf_10_trees_5_min_samples_leaf_for_big_cit_all_feat', 'w'))
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
	
