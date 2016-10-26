#include "predicatecpp.h"

Predicatecpp::Predicatecpp(std::string& f_name, std::string& sim_type, std::string& tok_type, std::string& cmp, double& t) {
  feat_name = f_name;
  sim_measure_type = sim_type;
  tokenizer_type = tok_type;
  comp_op = cmp;
  threshold = t;
  cost = 0.0;
}

Predicatecpp::Predicatecpp() {}

Predicatecpp::~Predicatecpp() {} 
