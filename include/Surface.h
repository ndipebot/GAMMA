#ifndef Surface_h
#define Surface_h
#include <vector>


using namespace std;

class FluxManager;

class Surface{
  public: 
    Surface();
    ~Surface();
    int birthElement_;  //
    int deathElement_;
    bool isDynamic_;
    bool isFixed_;
    bool isFlux_;
    double deathTime_;  //
    double birthTime_;  //
    int plane_;  //
    vector <int> setID_;
    double *areaWeight_;
    double *gpCoords_;
    double *unitNorm_;
    int *surfaceNodes_;
    vector<double> flux;
    vector<double> mappedCoords;
    
    void getMappedCoords(double boundCoords[4][3], double coordsMapped[4][2]);
    void getGradN(double chsi, double eta, double GN[2][4]);
    void getJacobian2D(double GN[2][4], double coordsMapped[4][2], double &detJac, double invJac[2][2]);
    void getShapeFcn(double *N, double chsi, double eta, double zeta);

    vector <FluxManager*> fluxManagerVec_;
    
};

#endif
