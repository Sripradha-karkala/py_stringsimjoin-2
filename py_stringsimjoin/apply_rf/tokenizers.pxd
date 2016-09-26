
from libcpp.vector cimport vector                                               
from libcpp.string cimport string  

cpdef void tokenize(vector[string]& lstrings, vector[string]& rstrings,         
                   const string& tok_type, const string& working_dir)

cpdef void load_tok(tok_type, path, vector[vector[int]]& ltokens, vector[vector[int]]& rtokens)
