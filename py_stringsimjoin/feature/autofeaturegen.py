
from py_stringmatching.tokenizer.whitespace_tokenizer import WhitespaceTokenizer
import pandas as pd 

from py_stringsimjoin.utils.simfunctions import get_sim_function                

def get_features():
    features = []
    ws_tok = WhitespaceTokenizer(return_set=True)
    default_sim_measure_types = ['JACCARD', 'COSINE', 'DICE', 'OVERLAP', 
                                 'OVERLAP_COEFFICIENT']
    for sim_measure_type in default_sim_measure_types:
        features.append((sim_measure_type.lower()+'_ws', sim_measure_type, 
                         ws_tok, get_sim_function(sim_measure_type)))

    feature_table_header = ['feature_name', 'sim_measure_type', 'tokenizer',
                            'sim_function']
    feature_table = pd.DataFrame(features, columns=feature_table_header)
    feature_table = feature_table.set_index('feature_name')

    return feature_table     
