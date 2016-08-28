#pragma once
#include "Tokenizer.h"

class DelimiterTokenizer : public Tokenizer {
  public:
    string delimiters;
    bool return_set;

    DelimiterTokenizer(std::string delimiters, bool return_set);
    ~DelimiterTokenizer(void);
    vector<string> tokenize(const string& str);
};
