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
       double *consolidFrac,
       double *solidRate,
       double &latent,
       double &liquidus,
       double &solidus,
       double &cp,
       double &dt,
       vector <double> &thetaN);

  virtual ~ThermalIsoManager() {}

  int *localNodes_;
  double *Nip_;
  double *condIp_;
  double *cpIp_;
  double *consolidFrac_;
  double *solidRate_;
  double *trackSrate_;
  bool checkSrate_[8];
  double &latent_;
  double &liquidus_;
  double &solidus_;
  double &cp_;
  double &dt_;

  vector<double> &thetaN_;
  
  virtual void execute();

};

#endif
