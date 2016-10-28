#include "predicatecpp.h"
#include <string>
#include <vector>

class Node {
  public:
    std::vector<Predicatecpp> predicates;
    std::string node_type;                                       
    std::vector<Node> children;
    
    Node();
    Node(std::vector<Predicatecpp>&, std::string&, std::vector<Node>&);
    Node(std::vector<Predicatecpp>&, std::string&);         
    Node(std::string&);         
    ~Node();

    void add_child(Node);
};
