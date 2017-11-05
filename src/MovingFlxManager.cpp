#include <vector>
#include <iterator>
#include <sstream>
#include <algorithm>
#include <string>
#include <fstream>
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <iomanip>

#include <BCManager.h>
#include <MovingFlxManager.h>

///////////////////////////////////////////
//		Constructor		 //
///////////////////////////////////////////
MovingFlxManager::MovingFlxManager(
      float *gpCoords,
      float *toolxyz,
      int &laserState,
      float &rBeam,
      float &Qin,
      int *surfaceNodes,
      float *Nip,
      float *areaWeight,
      vector<float> &thetaN,
      vector<float> &rhs)
      : gpCoords_(gpCoords),
        toolxyz_(toolxyz),
        rBeam_(rBeam),
        Qin_(Qin),
        surfaceNodes_(surfaceNodes),
        Nip_(Nip),
        areaWeight_(areaWeight),
        thetaN_(thetaN),
        rhs_(rhs),
        laserState_(laserState)
{
}

///////////////////////////////////////////
//		execute			 //
///////////////////////////////////////////
void
MovingFlxManager::execute()
{

  //calculate convection contribution 
  float small = 1.0e-8;
  float qmov = 0.0;
  for (int ip = 0; ip < 4; ip++)
  {
    int offsetIp = ip * 4;
    int offSet = ip * 3;
    //Calculate 
    float xip = gpCoords_[offSet+0];
    float yip = gpCoords_[offSet+1];
    float zip = gpCoords_[offSet+2];
    float r2 = ( ( xip - toolxyz_[0] ) * ( xip - toolxyz_[0] ) + 
                  ( yip - toolxyz_[1] ) * ( yip - toolxyz_[1] ) +
                  ( zip - toolxyz_[2] ) * ( zip - toolxyz_[2] ) );
    
    float rb2 = rBeam_ * rBeam_;
    if (laserState_ == 1) qmov = 3.0 * Qin_/(M_PI * rb2) * exp(-3.0 * r2 / rb2);
    else qmov = 0.0;
    // Scatter to global force vector
    for (int I = 0; I < 4; I++)
    {
      rhs_[surfaceNodes_[I]] += Nip_[offsetIp + I] * qmov * areaWeight_[ip];
    }//end for(I)

  }//end for(ip)

}// end execute()
