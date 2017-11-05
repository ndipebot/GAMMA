#ifndef ELEMENT_H
#define ELEMENT_H

/*=======================================================================
                      Element
     Define the element class base for FEM
  =======================================================================*/
#include <vector>
#include <math.h>

class MaterialManager;

class Element
{
public:
  // Constructor
  Element();

  // Destructor
  virtual ~Element();

  /*-------------------------
      Basic Attributes
    -------------------------*/
  int numIntPoints_;
    // number of integration points
  int nodesPerElement_;
    // number of nodes cosisting the element
  int* nID_;
    // ID of nodes that connect to this element
  int globId;
  bool birth_;
    // active state
  bool liquid_;
    // phase state
  float volume_;
    // volume of the element
  float* stiffMatrix_;
    // element-level stiffness matrix
  float* massMatrix_;
    // element-level mass matrix
  float birthTime_;
    // element birth time (LENS)
  int PID_;
    // Element part id
  int matID_, userID_;
  float cond_;
  float cp_;
  float rho_;
  float latent_;
  float liquidus_;
  float solidus_;
  float *condIp_;
  float *cpIp_;
  float *rhoIp_;
  float *volWeight_;
  float *consolidFrac_;
  float *solidRate_;

  MaterialManager* matManager_;

  /*---------------------------------------
     Basic methods associated with element
     type and geometry
    ---------------------------------------*/
  virtual void Jacobian(float* deriv, float* coords,
                        float* iJac, float &detJac)=0;
    // Jacobian determinant and inverse Jacobian matrix calculation
  virtual void shape_fcn(float* parCoord, float* shapeFcn)=0;
    // shape functin of each node at a given point
  virtual float charateristic_length(float* coordinates)=0;
    // calculate charateristic length of element
  
  /*---------------------------------------
    Basic methods associated with FEM
    ---------------------------------------*/
  virtual void element_level_stiffness_matrix(float* nodalCoords, 
                                              float* eleStiffMatrix,
                                              float* condIp)=0;
    // stiffness matrix of element level
  virtual float element_level_time_step()=0;
    // time step of element level

};

/*=======================================================================
Hexahedral Element
Define the element class base for FEM
=======================================================================*/
class HexahedralElement: public Element
{
public:
  HexahedralElement();
  ~HexahedralElement();

  /*---------------------------------------
     Basic methods associated with element
     type and geometry
    ---------------------------------------*/
  void Jacobian(float* deriv, float* coords, 
                float* iJac, float &detJac);
  // Jacobian determinant and inverse Jacobian matrix calculation
  void shape_fcn(float* parCoord, float* shapeFcn);
  // shape functin of each node at a given point

  void derivative_of_shape_fuction_about_real_coords(float* deriv,
                                                     float* iJac,
                                                     float* gradN);
  void derivative_of_shape_function_about_parametic_coords(float* parCoord, 
                                                           float* deriv);
  float charateristic_length(float* coordinates);
  // calculate charateristic length of element

  /*---------------------------------------
  Basic methods associated with FEM
  ---------------------------------------*/
  void element_level_stiffness_matrix(float* nodalCoords,float* eleStiffMatrix,
                                      float* condIp);
  // stiffness matrix of element level
  float element_level_time_step();
  // time step of element level
};
#endif
