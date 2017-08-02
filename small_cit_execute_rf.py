import pandas as pd
from py_stringsimjoin.apply_rf.executor import *
import pickle
from py_stringsimjoin.feature.autofeaturegen import get_features

base_path = '/scratch/small_cit/'                                               
run_path = 'small_cit_exp/run6/'                                                

def load_sample(sample_size):
    c = pd.read_csv(run_path + 'sample.csv')                                           
    sample = c.sample(sample_size)                                                        
    l_id = []                                                                       
    r_id = []                                                                       
                                                                                
    for idx, row in sample.iterrows():                                              
        l_id.append(row['l_id'])                                                    
        r_id.append(row['r_id']) 

    return (l_id, r_id)

rf=pickle.load(open(run_path + 'small_cit_rf_10_trees_3_min_samples_leaf','r'))

(l1, l2) = load_sample(10000)

ft=get_features(['JACCARD', 'COSINE', 'DICE', 'OVERLAP_COEFFICIENT', 'EDIT_DISTANCE', 'LEFT_LENGTH', 'RIGHT_LENGTH', 'LENGTH_SUM', 'LENGTH_DIFF'])
#ft=get_features(['JACCARD', 'COSINE', 'DICE', 'OVERLAP_COEFFICIENT', 'EDIT_DISTANCE'])

test_execute_rf(rf, ft, l1, l2, base_path + 'dblp.csv', 'citation_string', 
                base_path + 'google_scholar.csv', 'citation_string', 'cit_latest_10_tr_3_minsamp_leaf_run6_all_trees', 4)
