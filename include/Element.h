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
  double volume_;
    // volume of the element
  double* stiffMatrix_;
    // element-level stiffness matrix
  double* massMatrix_;
    // element-level mass matrix
  double birthTime_;
    // element birth time (LENS)
  int PID_;
    // Element part id
  double cond_;
  double cp_;
  double rho_;
  double latent_;
  double liquidus_;
  double solidus_;
  double *condIp_;
  double *cpIp_;
  double *rhoIp_;
  double *volWeight_;
  double *consolidFrac_;
  double *solidRate_;

  MaterialManager* matManager_;

  /*---------------------------------------
     Basic methods associated with element
     type and geometry
    ---------------------------------------*/
  virtual void Jacobian(double* deriv, double* coords,
                        double* iJac, double &detJac)=0;
    // Jacobian determinant and inverse Jacobian matrix calculation
  virtual void shape_fcn(double* parCoord, double* shapeFcn)=0;
    // shape functin of each node at a given point
  virtual double charateristic_length(double* coordinates)=0;
    // calculate charateristic length of element
  
  /*---------------------------------------
    Basic methods associated with FEM
    ---------------------------------------*/
  virtual void element_level_stiffness_matrix(double* nodalCoords, 
                                              double* eleStiffMatrix,
                                              double* condIp)=0;
    // stiffness matrix of element level
  virtual double element_level_time_step()=0;
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
  void Jacobian(double* deriv, double* coords, 
                double* iJac, double &detJac);
  // Jacobian determinant and inverse Jacobian matrix calculation
  void shape_fcn(double* parCoord, double* shapeFcn);
  // shape functin of each node at a given point

  void derivative_of_shape_fuction_about_real_coords(double* deriv,
                                                     double* iJac,
                                                     double* gradN);
  void derivative_of_shape_function_about_parametic_coords(double* parCoord, 
                                                           double* deriv);
  double charateristic_length(double* coordinates);
  // calculate charateristic length of element

  /*---------------------------------------
  Basic methods associated with FEM
  ---------------------------------------*/
  void element_level_stiffness_matrix(double* nodalCoords,double* eleStiffMatrix,
                                      double* condIp);
  // stiffness matrix of element level
  double element_level_time_step();
  // time step of element level
};
#endif
