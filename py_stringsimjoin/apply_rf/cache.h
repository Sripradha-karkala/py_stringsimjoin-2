#include <map>
#include <vector>

class Cache {
  public:
    std::vector<std::map<std::pair<int, int>, double> > cache_map;                       

    Cache(int);
    Cache();
    ~Cache();
    void add_entry(int, std::pair<int, int>&, double&);
    double lookup(int, std::pair<int, int>&);  
};
