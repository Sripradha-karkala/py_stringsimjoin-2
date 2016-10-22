#include "predicatecpp.h"

Predicatecpp::Predicatecpp(std::string& sim_type, std::string& tok_type, std::string& cmp, double& t) {
  sim_measure_type = sim_type;
  tokenizer_type = tok_type;
  comp_op = cmp;
  threshold = t;
}

Predicatecpp::Predicatecpp() {}

Predicatecpp::~Predicatecpp() {} 
