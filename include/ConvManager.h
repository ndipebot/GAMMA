#ifndef ConvManager_h
#define ConvManager_h
#include <vector>
#include <FluxManager.h>
#include <BCManager.h>

class ConvManager : public FluxManager
{
public:
  ConvManager(
       double &ambient,
       double &hconv,
       int *surfaceNodes,
       double * Nip,
       double * areaWeight,
       vector<double> &thetaN,
       vector<double> &rhs);

  virtual ~ConvManager() {}

  double ambient_;
  double hconv_;
  double *Nip_;
  double *areaWeight_;
  int *surfaceNodes_;

  vector<double> &thetaN_;
  vector<double> &rhs_;
  
  virtual void execute();

};

#endif
