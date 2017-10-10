#ifndef HeatSolverManager_h
#define HeatSolverManager_h

#include <Mesh.h>
#include <Element.h>
#include <DomainManager.h>
#include <BCManager.h>
#include <Surface.h>
#include <MaterialManager.h>
#include <ThermalIsoManager.h>

class Mesh;
class DomainManager;
class Element;
class Surface;
class BCManager;

class HeatSolverManager
{
public:
  HeatSolverManager(
         DomainManager *domainMgr,
         Mesh *meshObj);

  virtual ~HeatSolverManager() {}
  // Input
  DomainManager *domainMgr_;
  Mesh *meshObj_;

  // Local
  vector <double> thetaNp1_;
  vector <double> thetaN_;
  vector <double> rhs_;
  vector <double> Mvec_;
  int ngp_, nn_, nel_;
  double dt_, initTheta_, probeTime_, sumEnergyNp1_, sumEnergyN_;
  double *Nip_;
  BCManager *heatBCManager_;
  
  // methods
  void assignGaussPoints();

  void initializeTemp();

  void initializeEleStiff();

  void getInternalForce();

  void initializeMass();

  void updateMassBirth();

  double getTimeStep();

  void initializeSystem();

  void pre_work();
 
  void updateCap();
  
  void integrateForce();

  void advance_time_step();

  void post_work();

  void outputProbeData();

  void outputEnergyData();

};


#endif
