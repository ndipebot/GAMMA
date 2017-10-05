#ifndef MovingFlxManager_h
#define MovingFlxManager_h
#include <vector>
#include <FluxManager.h>

class MovingFlxManager : public FluxManager
{
public:
  MovingFlxManager(
       double *gpCoords,
       double *toolxyz,
       double &rBeam,
       double &Qin,
       int *surfaceNodes,
       double * Nip,
       double * areaWeight,
       vector<double> &thetaN,
       vector<double> &rhs);

  virtual ~MovingFlxManager() {}

  double *gpCoords_;
  double *toolxyz_;
  double Qin_;
  double rBeam_;
  double *Nip_;
  double *areaWeight_;
  int *surfaceNodes_;

  vector<double> &thetaN_;
  vector<double> &rhs_;
  
  virtual void execute();

};

#endif
