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
  Probe(float *physCoords);
  virtual ~Probe() {}
  
  float *parCoords_;
  float *physCoords_;
  int elementID_;
  ofstream outputFile_;
  bool hasElement_;

};

#endif
