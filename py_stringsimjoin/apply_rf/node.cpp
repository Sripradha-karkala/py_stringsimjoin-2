#include "node.h"

Node::Node(std::vector<Predicatecpp>& preds, std::string& ntype, std::vector<Node>& child_nodes) {
  predicates = preds;
  node_type = ntype;
  children = child_nodes;
}

Node::Node() {}

Node::~Node() {} 
