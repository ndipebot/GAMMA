#ifndef BCManager_h
#define BCManager_h

#include <Element.h>
#include <Surface.h>
#include <Mesh.h>
#include <FluxManager.h>
#include <ConvManager.h>
#include <RadManager.h>
#include <DomainManager.h>

class DomainManager;
class Mesh;

class BCManager
{
public:
  BCManager(
      vector<double> &thetaNp1,
      Mesh *meshObj,
      DomainManager *domainMgr, 
      vector<double> &thetaN,
      vector<double> &rhs);

  virtual ~BCManager() {}

  // Input
  vector <double> &thetaNp1_;
  vector <double> &thetaN_;
  vector <double> &rhs_;
  Mesh *meshObj_;
  DomainManager *domainMgr_;

  // Internal to class
  vector <double> fixedNodeVals_;
  vector <int> fixedNodeIDs_;
  vector <Surface> staticSurfList_;
  vector <Surface> surfaceList_;
  double *tooltxyz_ = new double[3];
  int laserState_;

  // Might need to readjust these later
  vector <int> activeBirthSurfaces_;
  vector <int> activeSurfaces_;
  vector <int> birthSurfaceList_;
  
  void prescribeDirichletBC();

  void assignSurfGP(vector<Surface> &surfaceList);

  void assignFixedNodes();

  void attachBirthSurf();
  
  void attachStaticSurf();
 
  void attachSurfNset(vector<Surface> &surfaceList);
  
  void attachSurfBC(vector<Surface> &surfaceList);

  void assignSurfFluxAlg(vector<Surface> &surfaceList);
  
  void assignActiveStaticSurf();

  void updateBirthSurfaces();

  void updateTool();

  void initializeBoundaries();

  void applyFluxes();

  void assignUnitNorm(vector <Surface> &surfaceList);

};


#endif
