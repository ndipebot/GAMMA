#include "Element.h"
#include "Surface.h"
#include "Mesh.h"

void assignParameters (map<string, double> paramValues, double &Rabszero,
                       double &Rambient, double &Rboltz);

void createNodeMap(map<int, int> &node_global_to_local, vector<double> &coordList,
                   Mesh *meshObj);

void createElementList(Mesh *meshObj, vector<Element*> &elementList,
                       map<int, int> &element_global_to_local,
                       map<int, int> node_global_to_local);

void assignElePID(Mesh *meshObj,
                  map<int, int> element_global_to_local,
                  vector<Element*> &elementList);

void assignFixedNodes(Mesh *meshObj,
                      map<int, int> node_global_to_local,
                      vector<int> &fixedNodeIDs, vector<double> &fixedNodeVals);

void checkElementJac(vector<Element*> elementList, vector<double> coordList,
                     map<int, int> node_global_to_local);

void assignElementBirth(Mesh *meshObj,
                        map<int,int> element_global_to_local,
                        vector<Element*> &elementList);

void createElElConn(vector<Element*> elementList, 
                    vector<int> &connVec, vector <int> &conn_to_el_Vec,
                    vector<int> &connVecIndx, vector< vector<int> > &connElEl,
                    vector<int> &numElEl);

void createConnSurf(vector<Element*> elementList, 
                    vector<int> numElEl, vector< vector<int> > connElEl,
                    vector< vector<int> > &connSurf);

void attachBirthSurf(Mesh *meshObj,
                     map<int, int> element_global_to_local,
                     vector<Element*> elementList,
                     vector< vector<int> > connSurf,
                     vector <Surface> &surfaceList_);

void attachStaticSurf(vector<int> numElEl, vector< vector<int> > connSurf,
                      vector<Element*> elementList, vector<Surface> &staticSurfList);

void attachSurfNset(Mesh *meshObj, vector<Element*> elementList,
                    vector<int> node_local_to_global, vector<Surface> &surfaceList_);

void attachSurfBC(Mesh *meshObj, vector<Element*> elementList,
                  vector<Surface> &surfaceList_);

// Auxilary functions

void sortIndxVec(vector<int> unsortVec, vector<int> &indx);
