import pandas as pd
from py_stringsimjoin.apply_rf.executor import ex_plan
import pickle
from py_stringsimjoin.apply_rf.execution_plan import *

ldf=pd.read_csv('/scratch/dblp_rahm.csv')                                       
rdf=pd.read_csv('/scratch/google_scholar.csv')                                  
'''                                                                            
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
'''
pls=pickle.load(open('ind_plans','r'))
rs=pickle.load(open('cit_rule_sets_tree_5_leaf_3','r'))
pl=generate_execution_plan1(pls, rs)
ex_plan(pl, rs, ldf, 'title', rdf, 'title', 't1', 4)     
