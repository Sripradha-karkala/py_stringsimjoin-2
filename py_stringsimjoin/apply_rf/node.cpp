#include "node.h"

Node::Node(std::vector<Predicatecpp>& preds, std::string& ntype, std::vector<Node>& child_nodes) {
  predicates = preds;
  node_type = ntype;
  children = child_nodes;
}

Node::Node(std::string& ntype) {
  node_type = ntype;
}

Node::Node(std::vector<Predicatecpp>& preds, std::string& ntype) {
  predicates = preds;                                                           
  node_type = ntype; 
}

Node::Node() {}

Node::~Node() {}

void Node::add_child(Node n) {
  children.push_back(n);
} 
