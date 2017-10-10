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
  double currTime_, minElementVol_, xmin_, xmax_, 
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

  vector<double> birthNodeTimes_;

  vector<double> coordList_;

  vector< vector<double>> tooltxyz_;
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

  void sortIndxVec_double(vector<double> unsortVec, vector<int> &indx);

  void initializeActiveElements();

  void updateActiveElements(vector<double> &thetaN, double initTemp);

  void getToolpath();

  void generateBins();

  void assignProbePoints();

  void initializeDomain();

  void freeMemory();

  double isInElement(vector< vector <double> > &elem_nodal_coor, 
		     double* point_coor, double* par_coor);

  bool within_tolerance( const double & val, const double & tol );
  
  double vector_norm_sq( const double * vect, int len );

  double parametric_distance(vector<double> x);

};

#endif
