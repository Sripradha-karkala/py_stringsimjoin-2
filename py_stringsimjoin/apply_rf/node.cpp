#include "node.h"

Node::Node(std::vector<Predicatecpp>& preds, std::string& ntype, std::vector<Node>& child_nodes, std::string& tid) {
  predicates = preds;
  node_type = ntype;
  children = child_nodes;
  tree_id = tid;
}

Node::Node() {}

Node::~Node() {} 
