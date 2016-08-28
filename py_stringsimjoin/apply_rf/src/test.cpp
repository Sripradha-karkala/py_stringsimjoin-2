#include "delimiter_tokenizer.h"
#include <iostream>

int main(int argc, char* argv[]) {
  DelimiterTokenizer* dl = new DelimiterTokenizer(" ", true);
  vector<string> tokens = dl->tokenize(argv[1]);
  for(int i=0; i<tokens.size(); i++) {std::cout << tokens[i] << ",";}
  std::cout <<"\n";
} 
