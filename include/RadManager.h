#ifndef RadManager_h
#define RadManager_h
#include <vector>
#include <FluxManager.h>

class RadManager : public FluxManager
{
public:
  RadManager(
       double &ambient,
       double &epsilon,
       double &abszero,
       double &sigma,
       int *surfaceNodes,
       double * Nip,
       double * areaWeight,
       vector<double> &thetaN,
       vector<double> &rhs);

  virtual ~RadManager() {}

  double ambient_;
  double epsilon_;
  double sigma_;
  double abszero_;
  double *Nip_;
  double *areaWeight_;
  int *surfaceNodes_;

  vector<double> &thetaN_;
  vector<double> &rhs_;
  
  virtual void execute();

};

#endif
