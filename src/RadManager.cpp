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
#include <RadManager.h>

///////////////////////////////////////////
//		Constructor		 //
///////////////////////////////////////////
RadManager::RadManager(
      double &ambient,
      double &epsilon,
      double &abszero,
      double &sigma,
      int *surfaceNodes,
      double *Nip,
      double *areaWeight,
      vector<double> &thetaN,
      vector<double> &rhs)
      : ambient_(ambient),
        epsilon_(epsilon),
        abszero_(abszero),
        sigma_(sigma),
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
RadManager::execute()
{

  //calculate convection contribution 
  double qrad = 0.0;
  for (int ip = 0; ip < 4; ip++)
  {
    int offsetIp = ip * 4;
    double thetaIp = 0.0;
    //Calculate 
    for (int I = 0; I < 4; I++)
    {
      thetaIp += Nip_[offsetIp + I] * thetaN_[surfaceNodes_[I]];
    }// end for(I)

    double ambient4 = pow( (ambient_ - abszero_), 4.0);
    double thetaIp4 = pow( (thetaIp  - abszero_), 4.0);
    qrad = -sigma_ * epsilon_ * (thetaIp4 - ambient4) * areaWeight_[ip];

    // Scatter to global force vector
    for (int I = 0; I < 4; I++)
    {
      rhs_[surfaceNodes_[I]] += Nip_[offsetIp + I] * qrad;
    }//end for(I)

  }//end for(ip)

}// end execute()
