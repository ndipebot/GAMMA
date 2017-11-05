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
    float deathTime_;  //
    float birthTime_;  //
    int plane_;  //
    vector <int> setID_;
    float *areaWeight_;
    float *gpCoords_;
    float *unitNorm_;
    int *surfaceNodes_;
	vector<float> flux;
	vector<float> mappedCoords;
    
    void getMappedCoords(float boundCoords[4][3], float coordsMapped[4][2]);
    void getGradN(float chsi, float eta, float GN[2][4]);
    void getJacobian2D(float GN[2][4], float coordsMapped[4][2], float &detJac, float invJac[2][2]);
    void getShapeFcn(float *N, float chsi, float eta, float zeta);

    vector <FluxManager*> fluxManagerVec_;
    
};

#endif
