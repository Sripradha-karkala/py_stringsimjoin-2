#include "cache.h"

Cache::Cache(int num_types) {
  int i;
  for (i=0; i<num_types; i++) {
    cache_map.push_back(std::map<std::pair<int, int>, double>());
  }
}

void Cache::add_entry(int tok_type, std::pair<int, int>& pair_id, double& overlap) {
  cache_map[tok_type][pair_id] = overlap;
}

double Cache::lookup(int tok_type, std::pair<int, int>& pair_id) {
  if (cache_map[tok_type].find(pair_id) == cache_map[tok_type].end()) {
    return -1;
  }
  return cache_map[tok_type][pair_id];
}

Cache::Cache() {}

Cache::~Cache() {} 
