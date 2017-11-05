#ifndef DomainManager_h
#define DomainManager_h

#include <vector>

#include <Element.h>
#include <Surface.h>
#include <Mesh.h>
#include <FluxManager.h>
#include <ConvManager.h>
#include <RadManager.h>
#include <Bin.h>
#include <Probe.h>

class Bin;
class Probe;

class DomainManager
{
public:
  DomainManager(Mesh *meshObj);


  virtual ~DomainManager() {}

  int nn_, nel_, nelactive_, nnactive_, nelactiveOld_, nnactiveOld_, 
      birthEleCt, birthNodeCt, binNx_, binNy_, binNz_;
  float currTime_, minElementVol_, xmin_, xmax_, 
         ymin_, ymax_, zmin_, zmax_, dxBin_, dyBin_, dzBin_;
  bool isInit_;

  map<int, int> node_global_to_local_;
  map<int, int> element_global_to_local_;

  vector< vector<int> > connElEl_;
  vector< vector<int> > connSurf_;

  vector<int> node_local_to_global_;
  vector<int> element_local_to_global_;
  vector<int> connVec_;
  vector<int> conn_to_el_Vec_;
  vector<int> connVecIndx_;
  vector<int> numElEl_;

  vector<int> activeElements_;
  vector<int> activeNodes_;
  vector<int> birthNodes_;
  vector<int> birthElements_;

  vector<float> birthNodeTimes_;

  vector<float> coordList_;

  vector< vector<float>> tooltxyz_;
  vector<int> laserOn_;

  vector<Element*> elementList_;
  vector<Bin*> binList_;
  vector<Probe*> probeList_;

  Mesh *meshObj_;

  void createNodeMap();

  void createReverseNodeMap();

  void createElementList();

  void assignElePID();

  void checkElementJac();

  void assignElementBirth();

  void assignElBirthList();

  void createElElConn();

  void createConnSurf();

  void sortIndxVec(vector<int> unsortVec, vector<int> &indx);

  void sortIndxVec_float(vector<float> unsortVec, vector<int> &indx);

  void initializeActiveElements();

  void updateActiveElements(vector<float> &thetaN, float initTemp);

  void getToolpath();

  void generateBins();

  void assignProbePoints();

  void initializeDomain();

  void freeMemory();

  float isInElement(vector< vector <float> > &elem_nodal_coor, 
		     float* point_coor, float* par_coor);

  bool within_tolerance( const float & val, const float & tol );
  
  float vector_norm_sq( const float * vect, int len );

  float parametric_distance(vector<float> x);

};

#endif
