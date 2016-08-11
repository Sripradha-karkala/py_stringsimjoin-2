
from py_stringmatching.tokenizer.qgram_tokenizer import QgramTokenizer
from py_stringmatching.tokenizer.whitespace_tokenizer import WhitespaceTokenizer
import pandas as pd 

from py_stringsimjoin.utils.simfunctions import get_sim_function                

def get_features(sim_measures=None, tokenizers=None):
    features = []
    ws_tok = WhitespaceTokenizer(return_set=True)
    if sim_measures is None:
        sim_measures = ['JACCARD', 'COSINE', 'DICE', 'OVERLAP', 
                       'OVERLAP_COEFFICIENT']
    if tokenizers is None:
        tokenizers = {'ws': WhitespaceTokenizer(return_set=True),
                      'qg2': QgramTokenizer(qval=2, return_set=True),
                      'qg3': QgramTokenizer(qval=3, return_set=True)}
    for sim_measure_type in sim_measures:
        for tok_name in tokenizers.keys():
            features.append((sim_measure_type.lower()+'_'+tok_name, sim_measure_type, 
                             tokenizers[tok_name], get_sim_function(sim_measure_type)))

    feature_table_header = ['feature_name', 'sim_measure_type', 'tokenizer',
                            'sim_function']
    feature_table = pd.DataFrame(features, columns=feature_table_header)
    feature_table = feature_table.set_index('feature_name')

    return feature_table     
