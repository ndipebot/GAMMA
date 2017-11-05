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
       float *Nip,
       float *condIp,
       float *cpIp,
       float *consolidFrac,
       float *solidRate,
       float &latent,
       float &liquidus,
       float &solidus,
       float &cp,
       float &dt,
       vector <float> &thetaN);

  virtual ~ThermalIsoManager() {}

  int *localNodes_;
  float *Nip_;
  float *condIp_;
  float *cpIp_;
  float *consolidFrac_;
  float *solidRate_;
  float *trackSrate_;
  bool checkSrate_[8];
  float &latent_;
  float &liquidus_;
  float &solidus_;
  float &cp_;
  float &dt_;

  vector<float> &thetaN_;
  
  virtual void execute();

};

#endif
