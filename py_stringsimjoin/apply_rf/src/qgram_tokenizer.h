#pragma once
#include "tokenizer.h"

class QgramTokenizer : public Tokenizer {
  public:
    int qval;
    bool return_set;

    QgramTokenizer(int qval, bool return_set);
    ~QgramTokenizer();

    vector<string> tokenize(const string& str);
};
