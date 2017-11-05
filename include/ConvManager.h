#ifndef ConvManager_h
#define ConvManager_h
#include <vector>
#include <FluxManager.h>
#include <BCManager.h>

class ConvManager : public FluxManager
{
public:
  ConvManager(
       float &ambient,
       float &hconv,
       int *surfaceNodes,
       float * Nip,
       float * areaWeight,
       vector<float> &thetaN,
       vector<float> &rhs);

  virtual ~ConvManager() {}

  float ambient_;
  float hconv_;
  float *Nip_;
  float *areaWeight_;
  int *surfaceNodes_;

  vector<float> &thetaN_;
  vector<float> &rhs_;
  
  virtual void execute();

};

#endif
