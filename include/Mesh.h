/*
 * Mesh.h
 *
 *  Created on: Oct 6, 2014
 *      Author: leonardo
 */

#ifndef MESH_H_
#define MESH_H_

#include <fstream>
#include <map>


using namespace std;



class Mesh {
public:
  Mesh(string fileName);
  virtual ~Mesh();
  
  map<int, vector<int> > PIDS_;
  map<int, vector<int> > ELEM_;
  map<int, vector<double> > NODES_;
  map<int, vector<int> > nodeSets_;
  map<int, vector<int> > loadSets_;
  map<int, double> initialCondition_;
  map<int, vector<double> > loadSetVals_;
  map<string, double> paramValues_;

  // Thermal material types
  map<int, vector<double> > PID_to_MAT_;
  map<int, int > PID_to_MAT_Type_;


  // Mechanical material types
  map<int, vector<double> > PID_to_matMech_;
  map<int, int > PID_to_matMech_Type_;


  vector<int> birthID_;
  vector<double> birthTime_;
  vector<double*> probePos_;
  vector<string> probeNames_;
  
  double inputDt_, finalTime_, outTime_, Rabszero_, Rambient_, Rboltz_,
         Qin_, Qeff_, rBeam_;

  string outFileName_;
  string toolFileName_;
  string energyFileName_;
  string probeExtension_ = ".txt";
  
  void getDomainInfo();
  void openNew(string fileName);
  void assignParameters();
  void freeMemory();

  bool isLENS_ = false;
  bool calcEnergy_ = false;
private:
  ifstream file;
  string getParam();
};

#endif /* MESH_H_ */

