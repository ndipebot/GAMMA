#ifndef MovingFlxManager_h
#define MovingFlxManager_h
#include <vector>
#include <FluxManager.h>

class MovingFlxManager : public FluxManager
{
public:
  MovingFlxManager(
       float *gpCoords,
       float *toolxyz,
       int &laserState,
       float &rBeam,
       float &Qin,
       int *surfaceNodes,
       float * Nip,
       float * areaWeight,
       vector<float> &thetaN,
       vector<float> &rhs);

  virtual ~MovingFlxManager() {}

  float *gpCoords_;
  float *toolxyz_;
  int &laserState_;
  float Qin_;
  float rBeam_;
  float *Nip_;
  float *areaWeight_;
  int *surfaceNodes_;

  vector<float> &thetaN_;
  vector<float> &rhs_;
  
  virtual void execute();

};

#endif
