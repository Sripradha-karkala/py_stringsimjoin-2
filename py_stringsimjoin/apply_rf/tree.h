#ifndef TREE_H                                                                  
#define TREE_H   

#include "rule.h"
#include <vector>

class Tree {
  public:
    std::vector<Rule> rules;
    
    Tree();
    Tree(std::vector<Rule>&);
    ~Tree();
};

#endif
