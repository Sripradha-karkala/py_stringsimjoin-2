#include "qgram_tokenizer.h"
#include "utils.h"

#include <set>

QgramTokenizer::QgramTokenizer(
  int qval, bool return_set): qval(qval), return_set(return_set) {
}

QgramTokenizer::~QgramTokenizer() {}

vector<string> QgramTokenizer::tokenize(const string& str) {
  vector<string> tokens;
  return return_set ? remove_duplicates(tokens) : tokens;
}
