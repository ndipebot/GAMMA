#ifndef ThermalIsoManager_h
#define ThermalIsoManager_h
#include <vector>

#include <HeatSolverManager.h>
#include <MaterialManager.h>

class ThermalIsoManager : public MaterialManager
{
public:
  ThermalIsoManager(
       int *localNodes,
       double *Nip,
       double *condIp,
       double *cpIp,
       double &latent,
       double &liquidus,
       double &solidus,
       double &cp,
       vector <double> &thetaN);

  virtual ~ThermalIsoManager() {}

  int *localNodes_;
  double *Nip_;
  double *condIp_;
  double *cpIp_;
  double &latent_;
  double &liquidus_;
  double &solidus_;
  double &cp_;

  vector<double> &thetaN_;
  
  virtual void execute();

};

#endif
