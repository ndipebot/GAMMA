#include <vector>
#include <iterator>
#include <sstream>
#include <algorithm>
#include <string>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <iomanip>

#include <BCManager.h>
#include <ConvManager.h>

///////////////////////////////////////////
//		Constructor		 //
///////////////////////////////////////////
ConvManager::ConvManager(
      float &ambient,
      float &hconv,
      int *surfaceNodes,
      float *Nip,
      float *areaWeight,
      vector<float> &thetaN,
      vector<float> &rhs)
      : ambient_(ambient),
        hconv_(hconv),
        surfaceNodes_(surfaceNodes),
        Nip_(Nip),
        areaWeight_(areaWeight),
        thetaN_(thetaN),
        rhs_(rhs)
{
}

///////////////////////////////////////////
//		execute			 //
///////////////////////////////////////////
void
ConvManager::execute()
{

  //calculate convection contribution 
  float qconv = 0.0;
  for (int ip = 0; ip < 4; ip++)
  {
    int offsetIp = ip * 4;
    float thetaIp = 0.0;
    //Calculate 
    for (int I = 0; I < 4; I++)
    {
      thetaIp += Nip_[offsetIp + I] * thetaN_[surfaceNodes_[I]];
    }// end for(I)
    qconv = -hconv_ * (thetaIp - ambient_) * areaWeight_[ip];
    // Scatter to global force vector
    for (int I = 0; I < 4; I++)
    {
      rhs_[surfaceNodes_[I]] += Nip_[offsetIp + I] * qconv;
    }//end for(I)

  }//end for(ip)

}// end execute()
