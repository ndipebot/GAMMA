#ifndef RadManager_h
#define RadManager_h
#include <vector>
#include <FluxManager.h>

class RadManager : public FluxManager
{
public:
  RadManager(
       float &ambient,
       float &epsilon,
       float &abszero,
       float &sigma,
       int *surfaceNodes,
       float * Nip,
       float * areaWeight,
       vector<float> &thetaN,
       vector<float> &rhs);

  virtual ~RadManager() {}

  float ambient_;
  float epsilon_;
  float sigma_;
  float abszero_;
  float *Nip_;
  float *areaWeight_;
  int *surfaceNodes_;

  vector<float> &thetaN_;
  vector<float> &rhs_;
  
  virtual void execute();

};

#endif
