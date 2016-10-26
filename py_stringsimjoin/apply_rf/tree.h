#include "rule.h"
#include <vector>

class Tree {
  public:
    std::vector<Rule> rules;
    
    Tree();
    Tree(std::vector<Rule>&);
    ~Tree();
};
