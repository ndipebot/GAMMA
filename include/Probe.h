#ifndef Probe_h
#define Probe_h
#include <vector>
#include <string>
#include <iomanip>
#include <iostream>
#include <fstream>

using namespace std;

class Probe{
public: 
  Probe(double *physCoords);
  virtual ~Probe() {}
  
  double *parCoords_;
  double *physCoords_;
  int elementID_;
  ofstream outputFile_;
  bool hasElement_;

};

#endif
