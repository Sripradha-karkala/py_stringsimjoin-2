
from libcpp.vector cimport vector
from libcpp.set cimport set as oset                                             
from libcpp.string cimport string
from libcpp cimport bool

import re2

cdef extern from "string.h":                                                    
    char *strtok (char *inp_str, const char *delimiters)  

cdef class WhitespaceTokenizer:

    def __init__(self, bool return_set):
        self.return_set = return_set

    cdef vector[string] tokenize(self, const string& inp_string):
        cdef char* pch                                                              
        pch = strtok (<char*> inp_string.c_str(), " ") 
        cdef oset[string] tokens                                            
        cdef vector[string] out_tokens                                              
        if self.return_set:                             
            while pch != NULL:                                                          
                tokens.insert(string(pch))                                          
                pch = strtok (NULL, " ")
            for s in tokens:
                out_tokens.push_back(s)                                                
        else:
            while pch != NULL:                                                  
                out_tokens.push_back(string(pch))                                      
                pch = strtok (NULL, " ")                                        
        return out_tokens 

cdef class QgramTokenizer:
    def __init__(self, int qval, bool padding, char prefix_pad, char suffix_pad, 
                 bool return_set):                                        
        self.qval = qval
        self.padding = padding
        self.prefix_pad = prefix_pad
        self.suffix_pad = suffix_pad
        self.return_set = return_set                                            
                                                                                
    cdef vector[string] tokenize(self, const string& inp_string):               
        cdef string inp_str = inp_string;                                                         
        if self.padding:                                                                
            inp_str = string(self.qval - 1, self.prefix_pad) + inp_str + string(self.qval - 1, self.suffix_pad)
        cdef oset[string] tokens                                                
        cdef vector[string] out_tokens     
        cdef int i, n = inp_str.length() - self.qval + 1
        if self.return_set:
            for i in range(n):                                                  
                tokens.insert(inp_str.substr(i, self.qval))
            for s in tokens:
                out_tokens.push_back(s) 
        else:
            for i in range(n):
                out_tokens.push_back(inp_str.substr(i, self.qval))
        return out_tokens

class AlphabeticTokenizer:                                                                                                          
    def __init__(self, return_set):                                        
        self.regex = re2.compile('[a-zA-Z]+')
        self.return_set = return_set                                            
                                                                                
    def tokenize(self, const string& inp_string):               
        cdef oset[string] tokens                                                
        cdef vector[string] out_tokens                                          
        if self.return_set:
            for s in self.regex.findall(inp_string):                                                     
                tokens.insert(s)                                      
            for s in tokens:                                                    
                out_tokens.push_back(s)                                         
        else:                                                                   
            for s in self.regex.findall(inp_string):                            
                out_tokens.push_back(s)  
        return out_tokens 

class AlphanumericTokenizer:                                                      
    def __init__(self, return_set):                                             
        self.regex = re2.compile('[a-zA-Z0-9]+')                                   
        self.return_set = return_set                                            
                                                                                
    def tokenize(self, const string& inp_string):                               
        cdef oset[string] tokens                                                
        cdef vector[string] out_tokens                                          
        if self.return_set:                                                     
            for s in self.regex.findall(inp_string):                            
                tokens.insert(s)                                                
            for s in tokens:                                                    
                out_tokens.push_back(s)                                         
        else:                                                                   
            for s in self.regex.findall(inp_string):                            
                out_tokens.push_back(s)                                         
        return out_tokens  

class NumericTokenizer:                                                      
    def __init__(self, return_set):                                             
        self.regex = re2.compile('[0-9]+')                                   
        self.return_set = return_set                                            
                                                                                
    def tokenize(self, const string& inp_string):                               
        cdef oset[string] tokens                                                
        cdef vector[string] out_tokens                                          
        if self.return_set:                                                     
            for s in self.regex.findall(inp_string):                            
                tokens.insert(s)                                                
            for s in tokens:                                                    
                out_tokens.push_back(s)                                         
        else:                                                                   
            for s in self.regex.findall(inp_string):                            
                out_tokens.push_back(s)                                         
        return out_tokens  

def test_tok(df, attr):
    cdef int q=3
    ws = QgramTokenizer(q, True, ord('#'), ord('$'), True)
#    ws = AlphanumericTokenizer(True)
    cdef vector[string] strings
    convert_to_vector(df[attr], strings)                           
    cdef vector[string] t
    for s in strings:
        t = ws.tokenize(s)                
        print t                
 
cdef void convert_to_vector(string_col, vector[string]& string_vector):         
    for val in string_col:                                                      
        string_vector.push_back(val)

cdef vector[string] remove_duplicates(vector[string]& inp_vector):          
    cdef vector[string] out_tokens                                                    
    cdef oset[string] seen_tokens
    cdef string inp_str                                     
    for inp_str in inp_vector:
        if seen_tokens.find(inp_str) == seen_tokens.end():        
            out_tokens.push_back(inp_str)                                                
            seen_tokens.insert(inp_str)                                                                                                                       
    return out_tokens
