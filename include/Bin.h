#ifndef Bin_h
#define Bin_h

#include <vector>

using namespace std;

class Bin{
  public:
    Bin();
    ~Bin();

    vector<int> elementBucket_;
    bool hasElements_;

};

#endif
