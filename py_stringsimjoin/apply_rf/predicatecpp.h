#ifndef PREDICATE_H
#define PREDICATE_H

#include <string>

class Predicatecpp {
  public:
    std::string sim_measure_type, tokenizer_type, comp_op;                      
    double threshold;                                       

    Predicatecpp();
    Predicatecpp(std::string&, std::string&, std::string&, double&);
    ~Predicatecpp();
};

#endif
