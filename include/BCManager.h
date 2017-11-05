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
      vector<float> &thetaNp1,
      Mesh *meshObj,
      DomainManager *domainMgr, 
      vector<float> &thetaN,
      vector<float> &rhs);

  virtual ~BCManager() {}

  // Input
  vector <float> &thetaNp1_;
  vector <float> &thetaN_;
  vector <float> &rhs_;
  Mesh *meshObj_;
  DomainManager *domainMgr_;

  // Internal to class
  vector <float> fixedNodeVals_;
  vector <int> fixedNodeIDs_;
  vector <Surface> staticSurfList_;
  vector <Surface> surfaceList_;
  float *tooltxyz_ = new float[3];
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
