#include <math.h>

#include "Surface.h"

using namespace std;

//==========================================================================
// Class Definition
//==========================================================================
// Holder for member variables and methods for each voxel
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------

Surface::Surface()
 : isDynamic_(false),
   birthElement_(0),
   deathElement_(0),
   deathTime_(0.0),
   isFlux_(false),
   isFixed_(false)
{
  //nothing to do
}

Surface::~Surface()
{
  // nothing to do
}

//////////////////////////////////////////////////////
//		getMappedCoords	                    //
//////////////////////////////////////////////////////
void
Surface::getMappedCoords(double boundCoords[4][3], double coordsMapped[4][2])
{
  double u[3], v[3], w[3];

  for (int ii = 0; ii < 3; ii++)
  {
    u[ii] = boundCoords[1][ii] - boundCoords[0][ii];
    v[ii] = boundCoords[2][ii] - boundCoords[1][ii];
    w[ii] = boundCoords[3][ii] - boundCoords[0][ii];
  }//end for(ii)

  for (int ii = 0; ii < 4; ii++)
  {
    for (int jj = 0; jj < 2; jj++)
    {
      coordsMapped[ii][jj] = 0.0;
    }//end for (jj)
  }//end for(ii)

  double l1 = sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2]);
  double l2 = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
  double l4 = sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]);

  double cos12 = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2])/(l1*l2);
  double cos14 = (u[0]*w[0] + u[1]*w[1] + u[2]*w[2])/(l1*l4);

  double sin12 = sqrt(1.0 - cos12 * cos12);
  double sin14 = sqrt(1.0 - cos14 * cos14);

  coordsMapped[1][0] = l1;
  coordsMapped[2][0] = l1 + l2*cos12;
  coordsMapped[2][1] = l2 * sin12;
  coordsMapped[3][0] = l4 * cos14;
  coordsMapped[3][1] = l4 * sin14;
}//end getMappedCoords

//////////////////////////////////////////////////////
//		getGradN	                    //
//////////////////////////////////////////////////////
void 
Surface::getGradN(double chsi, double eta, double GN[2][4])
{
  // w.r.t chsi
  GN[0][0] =  0.25 * (eta - 1.0);
  GN[0][1] =  0.25 * (1.0 - eta);
  GN[0][2] =  0.25 * (1.0 + eta);
  GN[0][3] = -0.25 * (1.0 + eta);

  GN[1][0] =  0.25 * (chsi - 1.0);
  GN[1][1] = -0.25 * (1.0 + chsi);
  GN[1][2] =  0.25 * (1.0 + chsi);
  GN[1][3] =  0.25 * (1.0 - chsi);
}

//////////////////////////////////////////////////////
//		getJacobian2D	                    //
//////////////////////////////////////////////////////
void 
Surface::getJacobian2D(double GN[2][4], double coordsMapped[4][2], double &detJac, double invJac[2][2])
{
  double Jac[2][2];
  Jac[0][0] = 0.0;
  Jac[0][1] = 0.0;
  Jac[1][0] = 0.0;
  Jac[1][1] = 0.0;

  for (int ii = 0; ii < 2; ii++)
  {
    for (int I = 0; I < 4; I++)
    {
      for (int jj = 0; jj < 2; jj++)
      {
        Jac[ii][jj] += GN[ii][I] * coordsMapped[I][jj];
      }//end for(jj)
    }//end for(I)
  }//end for(ii)

  detJac = Jac[0][0]*Jac[1][1] - Jac[1][0]*Jac[0][1];

  invJac[0][0] = (1.0/detJac) * Jac[1][1];
  invJac[1][0] = (1.0/detJac) * (-Jac[1][0]);
  invJac[0][1] = (1.0/detJac) * (-Jac[0][1]);
  invJac[1][1] = (1.0/detJac) * Jac[0][0];
}

//////////////////////////////////////////////////////
//		getShapeFcn	                    //
//////////////////////////////////////////////////////
void 
Surface::getShapeFcn(double *N, double chsi, double eta, double zeta)
{
   N[0] = 0.125*(1.0 - chsi)*(1.0 - eta)*(1.0 - zeta);
   N[3] = 0.125*(1.0 - chsi)*(1.0 + eta)*(1.0 - zeta);
   N[2] = 0.125*(1.0 + chsi)*(1.0 + eta)*(1.0 - zeta);
   N[1] = 0.125*(1.0 + chsi)*(1.0 - eta)*(1.0 - zeta);
   N[4] = 0.125*(1.0 - chsi)*(1.0 - eta)*(1.0 + zeta);
   N[7] = 0.125*(1.0 - chsi)*(1.0 + eta)*(1.0 + zeta);
   N[6] = 0.125*(1.0 + chsi)*(1.0 + eta)*(1.0 + zeta);
   N[5] = 0.125*(1.0 + chsi)*(1.0 - eta)*(1.0 + zeta);
}//end getShapeFcn
