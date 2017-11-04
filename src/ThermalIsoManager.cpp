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

#include <ThermalIsoManager.h>

///////////////////////////////////////////
//		Constructor		 //
///////////////////////////////////////////
ThermalIsoManager::ThermalIsoManager(
         int *localNodes,
         double *Nip,
         double *condIp,
         double *cpIp,
         double *consolidFrac,
         double *solidRate,
         double &latent,
         double &liquidus,
         double &solidus,
         double &cp,
         double &dt,
         vector<double> &thetaN)
         : localNodes_(localNodes),
           Nip_(Nip),
           condIp_(condIp),
           cpIp_(cpIp),
           latent_(latent),
           liquidus_(liquidus),
           solidus_(solidus),
           cp_(cp),
           thetaN_(thetaN),
           consolidFrac_(consolidFrac),
           solidRate_(solidRate),
           dt_(dt)
{

  trackSrate_ = new double[8];

  for (int ip = 0; ip < 8; ip++)
  {
    trackSrate_[ip] = 0.0;
    checkSrate_[ip] = false;
  }
}

///////////////////////////////////////////
//		execute			 //
///////////////////////////////////////////
void
ThermalIsoManager::execute()
{
  double small = 1.0e-16;
  double thetaNodes[8];
  // gather variables
  for (int I = 0; I < 8; I++)
  {
    int nid = localNodes_[I];
    thetaNodes[I] = thetaN_[nid];
  }//end for(I)

  //calculate convection contribution 
  for (int ip = 0; ip < 8; ip++)
  {
    int offsetIp = ip * 8;
    double thetaIp = 0.0;

    // Calculate ip values
    for (int I = 0; I < 8; I++)
    {
      thetaIp += Nip_[offsetIp + I] * thetaNodes[I];
    }
    

    double cpPoint;
    if ( thetaIp >= solidus_ && thetaIp <= liquidus_)
    {
      cpPoint = cp_ + latent_ / ( liquidus_ - solidus_ );
    }
    else
    {
      cpPoint = cp_;
    }

    if (fabs(cpPoint - cpIp_[ip]) > small)
    {
      cpIp_[ip] = cpPoint;
      updateMass_ = true;
    }

    // calculate consolidation fraction
    /*if ( consolidFrac_[ip] <= 1.0 - small )
    {
      if ( thetaIp > solidus_ )
      {
        if (thetaIp < liquidus_)
        {
	  consolidFrac_[ip] = ( thetaIp - solidus_ ) / (liquidus_ - solidus_);
        }
        else consolidFrac_[ip] = 1.0;
      }
    }*/

    // calculate solidification cooling rates
    if ( thetaIp > liquidus_)
    {
      solidRate_[ip] = 0.0;
      checkSrate_[ip] = false;
      trackSrate_[ip] = 0.0;
    }
    else if ( thetaIp < liquidus_ && 
         thetaIp > solidus_ && 
         !checkSrate_[ip] )
    {
      checkSrate_[ip] = true; 
      trackSrate_[ip] = 0.0;
      solidRate_[ip] = thetaIp;
    }
    else if ( checkSrate_[ip] && thetaIp > solidus_)
    {
      trackSrate_[ip] += dt_;
    }
    else if (thetaIp < solidus_ && checkSrate_[ip])
    { 
      // adjust if skipped over solidification..
      if ( trackSrate_[ip] < small )
      {
        solidRate_[ip] = (solidus_ - liquidus_)/dt_;
      }
      else
      {
        solidRate_[ip] = (thetaIp - solidRate_[ip]) / trackSrate_[ip];
      }
      checkSrate_[ip] = false;  
    }

  }//end for(ip)

}// end execute()
