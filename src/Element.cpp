/*=======================================================================
                        Element
              Definition of Element base class.
  =======================================================================*/
#include "Element.h"
#include <assert.h>
#include <algorithm>
#define __max(a,b)  (((a) > (b)) ? (a) : (b))
#define __min(a,b)  (((a) < (b)) ? (a) : (b))

/*-----------------
    Constructor
  -----------------*/
Element::Element()
{
  //nothing for now
}

/*-----------------
    Destructor
  -----------------*/
Element::~Element()
{
  // nothing for now
}


/*=======================================================================
                           Hexahedral Element
                   Declaration of Element base class.
=======================================================================*/

/*---------------
   - Constructor
  ---------------*/
HexahedralElement::HexahedralElement()
{
  nodesPerElement_ = 8;
  nID_ = new int[8];
  PID_ = -1;
  birth_ = false;
  //stiffMatrix_ = NULL;
}



/*--------------
   - Destructor
  --------------*/
HexahedralElement::~HexahedralElement()
{
  if (nID_)
    delete nID_;
  if (stiffMatrix_)
    delete stiffMatrix_;
}

/*---------------------
   - Jacobian
     coords[24]=[8][3]
     deriv[24] = [3][8];
  ---------------------*/
void
HexahedralElement::
Jacobian(float* deriv, float* coords, float* iJac, float &detJac)
{
  int count = 0;
  float Jac[3][3];
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      Jac[i][j] = 0.0;

  for (int k = 0; k < 3; k++) {
    for (int j = 0; j < 3; j++) {
      for (int i = 0; i < 8; i++) {
        Jac[k][j] += deriv[k * 8 + i] * coords[i * 3 + j];
      }
    }
  }

  // calculate determinant
  detJac = Jac[0][0] * Jac[1][1] * Jac[2][2] + Jac[0][1] * Jac[1][2] * Jac[2][0] +
           Jac[1][0] * Jac[2][1] * Jac[0][2] - Jac[0][2] * Jac[1][1] * Jac[2][0] -
           Jac[0][0] * Jac[1][2] * Jac[2][1] - Jac[0][1] * Jac[1][0] * Jac[2][2];
  assert(detJac >= 0.0);

  // cofactor matrix of Jacobian matrix
  iJac[0] = Jac[1][1] * Jac[2][2] - Jac[1][2] * Jac[2][1];
  iJac[1] = Jac[1][0] * Jac[2][2] - Jac[1][2] * Jac[2][0];
  iJac[2] = Jac[1][0] * Jac[2][1] - Jac[1][1] * Jac[2][0];

  iJac[3] = Jac[0][1] * Jac[2][2] - Jac[0][2] * Jac[2][1];
  iJac[4] = Jac[0][0] * Jac[2][2] - Jac[0][2] * Jac[2][0];
  iJac[5] = Jac[0][0] * Jac[2][1] - Jac[0][1] * Jac[2][0];

  iJac[6] = Jac[0][1] * Jac[1][2] - Jac[0][2] * Jac[1][1];
  iJac[7] = Jac[0][0] * Jac[1][2] - Jac[0][2] * Jac[1][0];
  iJac[8] = Jac[0][0] * Jac[1][1] - Jac[0][1] * Jac[1][0];

  for (unsigned char i = 0; i < 3; i++) {
    for (unsigned char j = 0; j < 3; j++) {
      iJac[i * 3 + j] *= std::pow(-1.0, i + j) / detJac;
    }
  }
  // inverse matrix of Jacobian matrix
  float swap;
  swap = iJac[1];
  iJac[1] = iJac[3];
  iJac[3] = swap;

  swap = iJac[2];
  iJac[2] = iJac[6];
  iJac[6] = swap;

  swap = iJac[5];
  iJac[5] = iJac[7];
  iJac[7] = swap;
}
/*----------------------------------------------------------
   - derivative_of_shape_function_about_real_coords
   gradN[24]=[3][8]
  ----------------------------------------------------------*/
void
HexahedralElement::derivative_of_shape_fuction_about_real_coords
(float* deriv, float* iJac, float* gradN)
{
  for (int k = 0; k < 8; k++)
    for (int j = 0; j < 3; j++)
    {
      for (int i = 0; i < 3; i++)
        gradN[j*8 + k] += iJac[j*3 + i] * deriv[i*8 + k];
    }
}

/*--------------------------------------------------------
  - gradient_of_shape_function_about_parametic_coordinates
  -------------------------------------------------------*/
void
HexahedralElement::derivative_of_shape_function_about_parametic_coords
(float* parCoord, float* deriv)
{
  float oneMinusChsi = 1.0 - parCoord[0];
  float onePlusChsi  = 1.0 + parCoord[0];
  float oneMinusEta  = 1.0 - parCoord[1];
  float onePlusEta   = 1.0 + parCoord[1];
  float oneMinusZeta = 1.0 - parCoord[2];
  float onePlusZeta  = 1.0 + parCoord[2];
  
  // with respect to chsi
  deriv[0] = -0.1250 * oneMinusEta * oneMinusZeta;
  deriv[2] =  0.1250 * onePlusEta * oneMinusZeta;
  deriv[4] = -0.1250 * oneMinusEta * onePlusZeta;
  deriv[6] =  0.1250 * onePlusEta * onePlusZeta;
  for (int i = 0; i < 4; i++)
    deriv[i * 2 + 1] = -deriv[i * 2];

  // with respect to eta
  deriv[0 + 8] = -0.1250 * oneMinusChsi * oneMinusZeta;
  deriv[3 + 8] = -deriv[8];
  deriv[1 + 8] = -0.1250 * onePlusChsi * oneMinusZeta;
  deriv[2 + 8] = -deriv[9];
  deriv[4 + 8] = -0.1250 * oneMinusChsi * onePlusZeta;
  deriv[7 + 8] = -deriv[12];
  deriv[5 + 8] = -0.1250 * onePlusChsi * onePlusZeta;
  deriv[6 + 8] = -deriv[13];

  // with respect to zeta
  deriv[4 + 16] = 0.1250 * oneMinusChsi * oneMinusEta;
  deriv[5 + 16] = 0.1250 * onePlusChsi * oneMinusEta;
  deriv[6 + 16] = 0.1250 * onePlusChsi * onePlusEta;
  deriv[7 + 16] = 0.1250 * oneMinusChsi * onePlusEta;
  for (int i = 0; i < 4; i++)
    deriv[i + 16] = -deriv[i + 20];
  
}

/*---------------
    shape_fcn
  ---------------*/
void
HexahedralElement::shape_fcn(float* parCoord, float* N)
{
  float chsi = parCoord[0];
  float eta = parCoord[1];
  float zeta = parCoord[2];

  N[0] = 0.125*(1.0 - chsi)*(1.0 - eta)*(1.0 - zeta);
  N[3] = 0.125*(1.0 - chsi)*(1.0 + eta)*(1.0 - zeta);
  N[2] = 0.125*(1.0 + chsi)*(1.0 + eta)*(1.0 - zeta);
  N[1] = 0.125*(1.0 + chsi)*(1.0 - eta)*(1.0 - zeta);
  N[4] = 0.125*(1.0 - chsi)*(1.0 - eta)*(1.0 + zeta);
  N[7] = 0.125*(1.0 - chsi)*(1.0 + eta)*(1.0 + zeta);
  N[6] = 0.125*(1.0 + chsi)*(1.0 + eta)*(1.0 + zeta);
  N[5] = 0.125*(1.0 + chsi)*(1.0 - eta)*(1.0 + zeta);
}

/*------------------------
   characteristic_length
  ------------------------*/
float
HexahedralElement::charateristic_length(float* coordinates)
{
  float characterLength;
  // FIXME: replace xyz by coordinates
  float xyz[3][8];
  for (int i = 0; i < 8; i++)
    for (int j = 0; j < 3; j++)
    {
      xyz[j][i] = coordinates[i * 3 + j];
    }
  unsigned char fac[4][6] = { { 0,4,0,1,2,3 },{ 1,5,1,2,3,0 },
                              { 2,6,5,6,7,4 },{ 3,7,4,5,6,7 } };
  float dt, at;
  float e, f, g, atest, x13[3], x24[3], fs[3], ft[3];
  float areal = 1.0e20, aream = 0.0;
  unsigned char k1, k2, k3, k4;

  // Calculate area of each surface
  for (unsigned char j = 0; j<6; j++)
  {
    for (unsigned char i = 0; i<3; i++)
    {
      k1 = fac[0][j];
      k2 = fac[1][j];
      k3 = fac[2][j];
      k4 = fac[3][j];

      x13[i] = xyz[i][k3] - xyz[i][k1];
      x24[i] = xyz[i][k4] - xyz[i][k2];

      fs[i] = x13[i] - x24[i];
      ft[i] = x13[i] + x24[i];
    }// end loop i

    e = fs[0] * fs[0] + fs[1] * fs[1] + fs[2] * fs[2];
    f = fs[0] * ft[0] + fs[1] * ft[1] + fs[2] * ft[2];
    g = ft[0] * ft[0] + ft[1] * ft[1] + ft[2] * ft[2];

    atest = e*g - f*f;     // area/4  (4*area)^2

    aream = __max(atest, aream);
  }// end loop j

  characterLength = 4 * (volume_) / sqrt(aream);

  return characterLength;
}

/*----------------------------------
   element_level_stiffness_matrix
  ----------------------------------*/
void
HexahedralElement::
element_level_stiffness_matrix(float* coordinates,float* stiffMatrix, float* condIp)
{
  // Gauss points
  float parCoords[8][3] = { {-1.0, -1.0, -1.0},{ 1.0, -1.0, -1.0},
                             { 1.0,  1.0, -1.0},{-1.0,  1.0, -1.0},
                             {-1.0, -1.0,  1.0},{ 1.0, -1.0,  1.0},
                             { 1.0,  1.0,  1.0},{-1.0,  1.0,  1.0} };
  for (int j = 0; j < 8; j++)
    for (int i = 0; i < 3; i++)
      parCoords[j][i] *= 0.5773502692;
  // weights
  float weight[8] = { 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0 };

  // loop over all the Gauss points
  for (int nGaussPoint = 0; nGaussPoint < 8; nGaussPoint++)
  {
    float deriv[24],gradN[24]; // [3][8]
    float detJac=0.0, iJac[9];
    float kappa = condIp[nGaussPoint];
    for (int i = 0; i < 24; i++)
    {
      deriv[i] = 0.0;
      gradN[i] = 0.0;
      if (i<9)
        iJac[i] = 0.0;
    }
    // get derivative of shape function with respect to parametic coordinates
    derivative_of_shape_function_about_parametic_coords(parCoords[nGaussPoint],
                                                        deriv);
    // get inverse Jacob matrix and its determinant
    Jacobian(deriv, coordinates, iJac, detJac);

    // get derivative of shape functin with respect to real coordinates
    derivative_of_shape_fuction_about_real_coords(deriv, iJac, gradN);

    // assemble the stiffness matrix of element
    int count = 0;
    for (int I=0; I<8; I++)
    {
      for (int J = I; J < 8; J++)
      {
        for (int i = 0; i < 3; i++)
        {
          stiffMatrix[count] += gradN[i * 8 + I] * kappa * gradN[i * 8 + J] * detJac*weight[nGaussPoint];
        }
        count++;
      }//end for(J)
    }//end for(I)
  }
}

/*--------------------------
   element_level_time_step
  --------------------------*/
float
HexahedralElement::element_level_time_step()
{
  float dT;
  // FIXME: element with different \rho*Cp
  return dT;
}
