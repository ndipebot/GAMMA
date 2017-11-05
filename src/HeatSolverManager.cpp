#include <vector>
#include <stdio.h>
#include <iterator>
#include <sstream>
#include <algorithm>
#include <string>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <iomanip>
#include "time.h"

// user-defined headers
#include <Mesh.h>
#include <Element.h>
#include <dynaInput.h>
#include <Surface.h>
#include <BCManager.h>
#include <HeatSolverManager.h>
#include <MaterialManager.h>
#include <ThermalIsoManager.h>

//////////////////////////////////////////////////////
//		Constructor			    //
//////////////////////////////////////////////////////
HeatSolverManager::HeatSolverManager(DomainManager *domainMgr,
                                     Mesh *meshObj)
                                     : domainMgr_(domainMgr),
                                       meshObj_(meshObj),
                                       probeTime_(0.0)
{
  dt_ = 1.0;
}

//////////////////////////////////////////////////////
//		assignGaussPoints		    //
//////////////////////////////////////////////////////
void 
HeatSolverManager::assignGaussPoints()
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
  ngp_ = 2;
}//end assignGaussPoints

//////////////////////////////////////////////////////
//		initializeTemp			    //
//////////////////////////////////////////////////////
void 
HeatSolverManager::initializeTemp()
{
  // HeatSolver Manager(both)
  for (map<int,float>::iterator it = meshObj_->initialCondition_.begin();
       it != meshObj_->initialCondition_.end(); it++)
  {
    initTheta_ = it->second;
  }//end for(it)

  for (int ii = 0; ii < nn_; ii++)
  {
    thetaN_[ii] = initTheta_;
    thetaNp1_[ii] = initTheta_;
  }//end for(ii)
}// end initializeTemp

//////////////////////////////////////////////////////
//		initializeEleStiff		    //
//////////////////////////////////////////////////////
void 
HeatSolverManager::initializeEleStiff()
{
  HexahedralElement *ele = new HexahedralElement;
  for (int ie = 0; ie < domainMgr_->elementList_.size(); ie++)
  {
    Element *element = domainMgr_->elementList_[ie];
    element->stiffMatrix_ = new float[36];
    int *localNodes = element->nID_;
    float kappa = element->cond_;
    float rhs_e[8];
    float nodalCoords[24];
    // unpack local data
    element->condIp_ = new float[8];
    for (int inode = 0; inode < 8; inode++)
    {
      int nid = localNodes[inode];
      rhs_e[inode] = 0.0;
      element->condIp_[inode] = element->cond_;
      for (int kk = 0; kk < 3; kk++)
      {
	nodalCoords[kk+inode*3] = domainMgr_->coordList_[nid*3 + kk];
      }
    }
 
    for (int iCom = 0; iCom < 36; iCom++) element->stiffMatrix_[iCom] = 0.0;

    ele->element_level_stiffness_matrix(nodalCoords, element->stiffMatrix_, element->condIp_);

  }//end for(ie)

}//end initializeEleStiff

//////////////////////////////////////////////////////
//		getInternalForce		    //
//////////////////////////////////////////////////////
void 
HeatSolverManager::getInternalForce()
{
  int mapIndex[8][8] = { { 0,  1,  2,  3,  4,  5,  6,  7 },
                         { 1,  8,  9, 10, 11, 12, 13, 14 },
                         { 2,  9, 15, 16, 17, 18, 19, 20 },
                         { 3, 10, 16, 21, 22, 23, 24, 25 },
                         { 4, 11, 17, 22, 26, 27, 28, 29 },
                         { 5, 12, 18, 23, 27, 30, 31, 32 },
                         { 6, 13, 19, 24, 28, 31, 33, 34 },
                         { 7, 14, 20, 25, 29, 32, 34, 35 } };

  for (int ie = 0; ie < domainMgr_->activeElements_.size(); ie++)
  {
      int eID = domainMgr_->activeElements_[ie];
      Element * element = domainMgr_->elementList_[eID]; 
      
      //grab local nodes for this element
      int *localNodes = element->nID_;
      float *eleStiff = element->stiffMatrix_;
//      float *massMat = element->massMatrix_;

      for (int I = 0; I < 8; I++)
      {
	int IG = localNodes[I];
	for (int J = 0; J < 8; J++)
	{
	  int JG = localNodes[J];
	  int stiffInd = mapIndex[I][J];
	  rhs_[IG] -= eleStiff[stiffInd] * thetaN_[JG];
//          Mvec_[IG] += massMat[stiffInd];
	}//end for(J)
      }//end for(I)

  }//end for(ie)
}//end getInternalForce

//////////////////////////////////////////////////////
//		initializeMass			    //
//////////////////////////////////////////////////////
void 
HeatSolverManager::initializeMass()
{
  Nip_ = new float[8 * 8];
  float parCoords[8][3] = { {-1.0, -1.0, -1.0},{ 1.0, -1.0, -1.0},
                             { 1.0,  1.0, -1.0},{-1.0,  1.0, -1.0},
                             {-1.0, -1.0,  1.0},{ 1.0, -1.0,  1.0},
                             { 1.0,  1.0,  1.0},{-1.0,  1.0,  1.0} };

  for (int j = 0; j < 8; j++)
    for (int i = 0; i < 3; i++)
      parCoords[j][i] *= 0.5773502692;

  // Pre-calculate shape fcn at ip
  for (int ip = 0; ip < 8; ip++)
  {
    HexahedralElement *HexTemp = new HexahedralElement;
    float N[8];
    HexTemp->shape_fcn(parCoords[ip], N);
    int offSetIp = ip * 8;
    for (int I = 0; I < 8; I++)
    {
      Nip_[offSetIp+I] = N[I];
    }//end for(I)
  }//end for(ip)


  float weight[8] = { 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0 };

  for (int ie = 0; ie < domainMgr_->activeElements_.size(); ie++)
  {
    int eID = domainMgr_->activeElements_[ie];
    Element * element = domainMgr_->elementList_[eID];
    HexahedralElement *HexTemp = new HexahedralElement;
    int *localNodes = element->nID_;
    float rho = element->rho_;
    float cp = element->cp_;  
    element->cpIp_ = new float[8];
    element->rhoIp_ = new float[8];
    element->volWeight_ = new float[8];
    element->massMatrix_ = new float[36];
    element->consolidFrac_ = new float[8];
    element->solidRate_ = new float[8];
    float nodalCoords[24];
    for (int iNode = 0; iNode < 8; iNode++)
    {
      int nid = localNodes[iNode];
      for (int kk = 0; kk < 3; kk++)
      {
        nodalCoords[kk + iNode*3] = domainMgr_->coordList_[nid*3+kk];
      }//end for(kk)
    }//end for(iNode)

    for (int ip = 0; ip < 8; ip++)
    { 
      element->consolidFrac_[ip] = 0.0;
      element->solidRate_[ip] = 0.0;
      float deriv[24],gradN[24], iJac[9];
      float detJac = 0.0;
      float *N = new float[8];
      int offSetIp = ip * 8;

      // zero everything out for this ip
      for (int i = 0; i < 24; i++)
      {
        deriv[i] = 0.0;
        gradN[i] = 0.0;
        if (i<9) iJac[i] = 0.0;
      }//end for(i)
     
      N = &Nip_[offSetIp];
      HexTemp->derivative_of_shape_function_about_parametic_coords(parCoords[ip], deriv);
      HexTemp->Jacobian(deriv, nodalCoords, iJac, detJac);
      element->volWeight_[ip] = detJac * weight[ip];
      //FIXME: Need to implement a couple of material managers here
      element->matManager_ = new ThermalIsoManager(element->nID_, Nip_, element->rhoIp_,
				     element->cpIp_, element->consolidFrac_, element->solidRate_,
				     element->latent_, element->liquidus_, element->solidus_, 
				     element->cp_, dt_, thetaN_);

      element->matManager_->execute();
      float cpIp = element->cpIp_[ip];

      // Calculate saved mass matrix
      int count = 0;
      for (int I = 0; I < 8; I++)
      {
        for (int J = I; J < 8; J++)
        {
          element->massMatrix_[count] += N[I] * rho * cpIp * N[J] * detJac * weight[ip];
          count++;
        }//end for(J)
      }//end for(I)


      // sum to global
      for (int I = 0; I < 8; I++)
      {
        int IG = localNodes[I];
        for (int J = 0; J < 8; J++)
        {
          Mvec_[IG] += N[I] * rho * cpIp * N[J] * detJac * weight[ip];
        }//end for(J)
      }//end for(I)

    }//end for(ip)

  }//end for(ie)

}//end initializeMass

//////////////////////////////////////////////////////
//		updateMassBirth			    //
//////////////////////////////////////////////////////
void 
HeatSolverManager::updateMassBirth()
{
  float parCoords[8][3] = { {-1.0, -1.0, -1.0},{ 1.0, -1.0, -1.0},
                             { 1.0,  1.0, -1.0},{-1.0,  1.0, -1.0},
                             {-1.0, -1.0,  1.0},{ 1.0, -1.0,  1.0},
                             { 1.0,  1.0,  1.0},{-1.0,  1.0,  1.0} };

  for (int j = 0; j < 8; j++)
    for (int i = 0; i < 3; i++)
      parCoords[j][i] *= 0.5773502692;

  float weight[8] = { 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0 };
  HexahedralElement *HexTemp = new HexahedralElement;

  for (int ie = domainMgr_->nelactiveOld_; ie < domainMgr_->activeElements_.size(); ie++)
  {
    int eID = domainMgr_->activeElements_[ie];
    Element * element = domainMgr_->elementList_[eID];
    int *localNodes = element->nID_;
    float rho = element->rho_;
    float cp = element->cp_;  
    element->cpIp_ = new float[8];
    element->rhoIp_ = new float[8];
    element->volWeight_ = new float[8];
    element->massMatrix_ = new float[36];
    element->consolidFrac_ = new float[8];
    element->solidRate_ = new float[8];
    element->matManager_ = new ThermalIsoManager(element->nID_, Nip_, element->condIp_,
				   element->cpIp_, element->consolidFrac_, element->solidRate_,
                                   element->latent_, element->liquidus_, element->solidus_, 
                                   element->cp_, dt_, thetaN_);
    element->matManager_->execute();
    float nodalCoords[24];

    for (int iCom = 0; iCom < 36; iCom++) element->massMatrix_[iCom] = 0.0;
 
    for (int iNode = 0; iNode < 8; iNode++)
    {
      int nid = localNodes[iNode];
      for (int kk = 0; kk < 3; kk++)
      {
        nodalCoords[kk + iNode*3] = domainMgr_->coordList_[nid*3+kk];
      }//end for(kk)
    }//end for(iNode)

    for (int ip = 0; ip < 8; ip++)
    { 
      element->consolidFrac_[ip] = 0.0;
      element->solidRate_[ip] = 0.0;
      float deriv[24], N[8], gradN[24], iJac[9];
      float detJac = 0.0;

      // zero everything out for this ip
      for (int i = 0; i < 24; i++)
      {
        deriv[i] = 0.0;
        gradN[i] = 0.0;
        if (i<9) iJac[i] = 0.0;
      }//end for(i)
      HexTemp->shape_fcn(parCoords[ip], N);
      HexTemp->derivative_of_shape_function_about_parametic_coords(parCoords[ip], deriv);
      HexTemp->Jacobian(deriv, nodalCoords, iJac, detJac);
      element->volWeight_[ip] = detJac * weight[ip];
      float cpIp = element->cpIp_[ip];
      for (int I = 0; I < 8; I++)
      {
	int IG = localNodes[I];
        for (int J = 0; J < 8; J++)
        {
          Mvec_[IG] += N[I] * rho * cpIp * N[J] * detJac * weight[ip];
        }//end for(I)
      }//end for(J)

      // Calculate saved mass matrix
      int count = 0;
      for (int I = 0; I < 8; I++)
      {
        for (int J = I; J < 8; J++)
        {
          element->massMatrix_[count] += N[I] * rho * cpIp * N[J] * detJac * weight[ip];
          count++;
        }//end for(J)
      }//end for(I)

    }//end for(ip)

  }//end for(ie)
}//end updateMassBirth

//////////////////////////////////////////////////////
//		getTimeStep			    //
//////////////////////////////////////////////////////
float 
HeatSolverManager::getTimeStep()
{
  float dt = 1.0e8;
  float defaultFac = 0.95;
  float parCoords[8][3] = { {-1.0, -1.0, -1.0},{ 1.0, -1.0, -1.0},
                             { 1.0,  1.0, -1.0},{-1.0,  1.0, -1.0},
                             {-1.0, -1.0,  1.0},{ 1.0, -1.0,  1.0},
                             { 1.0,  1.0,  1.0},{-1.0,  1.0,  1.0} };
 
  int ngp2D = 4;
  float gPoints2D [4][2] = { {-1.0/sqrt(3.0), -1.0/sqrt(3.0)},
                              {-1.0/sqrt(3.0),  1.0/sqrt(3.0)},
                              { 1.0/sqrt(3.0), -1.0/sqrt(3.0)}, 
                              { 1.0/sqrt(3.0),  1.0/sqrt(3.0)} };
  float faceCoords[4][3];
  float mappedCoords[4][2]; 
  float surfNodes[4];
  float *Nsurf = new float[8];
  float gpNsurf[4][8];

  for (int j = 0; j < 8; j++)
    for (int i = 0; i < 3; i++)
      parCoords[j][i] *= 0.5773502692;

  float weight[8] = { 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0 };


  for (int ie = 0; ie < domainMgr_->elementList_.size(); ie++)
  {
    Element * element = domainMgr_->elementList_[ie];
    HexahedralElement *HexTemp = new HexahedralElement;
    Surface *SurfTemp = new Surface;
    int *localNodes = element->nID_;
    float rho = element->rho_;
    float cp = element->cp_;  
    float cond = element->cond_;  
    float nodalCoords[24];
    float eleVol = 0.0;
    for (int iNode = 0; iNode < 8; iNode++)
    {
      int nid = localNodes[iNode];
      for (int kk = 0; kk < 3; kk++)
      {
        nodalCoords[kk + iNode*3] = domainMgr_->coordList_[nid*3+kk];
      }//end for(kk)
    }//end for(iNode)

    // Calculate element volumes
    for (int ip = 0; ip < 8; ip++)
    { 
      float deriv[24], N[8], gradN[24], iJac[9];
      float detJac = 0.0;

      // zero everything out for this ip
      for (int i = 0; i < 24; i++)
      {
        deriv[i] = 0.0;
        gradN[i] = 0.0;
        if (i<9) iJac[i] = 0.0;
      }//end for(i)
     
      HexTemp->shape_fcn(parCoords[ip], N);
      HexTemp->derivative_of_shape_function_about_parametic_coords(parCoords[ip], deriv);
      HexTemp->Jacobian(deriv, nodalCoords, iJac, detJac);
      eleVol += detJac;

    }//end for(ip)

    // Calculate element surface areas
    float areaMax = 0.;
    for (int ll = 0; ll < 6; ll++)
    {
      float areaFace = 0.0;
      if (ll == 0)
      {
	surfNodes[0] = localNodes[7];
	surfNodes[1] = localNodes[6];
	surfNodes[2] = localNodes[5];
	surfNodes[3] = localNodes[4];
	for (int ip = 0; ip < ngp2D; ip++)
	{
	  SurfTemp->getShapeFcn(Nsurf, gPoints2D[ip][0], gPoints2D[ip][1], 1.0);
	  for (int I = 0; I < 8; I++) gpNsurf[ip][I] = Nsurf[I];
	}//end for(ip)

      }
      else if(ll == 1)
      {
	surfNodes[0] = localNodes[0];
	surfNodes[1] = localNodes[1];
	surfNodes[2] = localNodes[2];
	surfNodes[3] = localNodes[3];
	for (int ip = 0; ip < ngp2D; ip++)
	{
	  SurfTemp->getShapeFcn(Nsurf, gPoints2D[ip][0], gPoints2D[ip][1], -1.0);
	  for (int I = 0; I < 8; I++) gpNsurf[ip][I] = Nsurf[I];
	}//end for(ip)
      }
      else if(ll == 2)
      {
	surfNodes[0] = localNodes[0];
	surfNodes[1] = localNodes[4];
	surfNodes[2] = localNodes[5];
	surfNodes[3] = localNodes[1];
	for (int ip = 0; ip < ngp2D; ip++)
	{
	  SurfTemp->getShapeFcn(Nsurf, gPoints2D[ip][0], -1.0, gPoints2D[ip][1]);
	  for (int I = 0; I < 8; I++) gpNsurf[ip][I] = Nsurf[I];
	}//end for(ip)
      }
      else if(ll == 3)
      {
	surfNodes[0] = localNodes[3];
	surfNodes[1] = localNodes[2];
	surfNodes[2] = localNodes[6];
	surfNodes[3] = localNodes[7];
	for (int ip = 0; ip < ngp2D; ip++)
	{
	  SurfTemp->getShapeFcn(Nsurf, gPoints2D[ip][0], 1.0, gPoints2D[ip][1]);
	  for (int I = 0; I < 8; I++) gpNsurf[ip][I] = Nsurf[I];
	}//end for(ip)
      }
      else if(ll == 4)
      {
	surfNodes[0] = localNodes[0];
	surfNodes[1] = localNodes[4];
	surfNodes[2] = localNodes[7];
	surfNodes[3] = localNodes[3];
	for (int ip = 0; ip < ngp2D; ip++)
	{
	  SurfTemp->getShapeFcn(Nsurf, -1.0, gPoints2D[ip][0], gPoints2D[ip][1]);
	  for (int I = 0; I < 8; I++) gpNsurf[ip][I] = Nsurf[I];
	}//end for(ip)
      }
      else if(ll == 5)
      {
	surfNodes[0] = localNodes[1];
	surfNodes[1] = localNodes[2];
	surfNodes[2] = localNodes[6];
	surfNodes[3] = localNodes[5];
	for (int ip = 0; ip < ngp2D; ip++)
	{
	  SurfTemp->getShapeFcn(Nsurf,  1.0, gPoints2D[ip][0], gPoints2D[ip][1]);
	  for (int I = 0; I < 8; I++) gpNsurf[ip][I] = Nsurf[I];
	}//end for(ip)
      }//end if

      // pick out nodal coordinates
      for (int jj = 0; jj < 4; jj++)
      {
	int nid = surfNodes[jj];
	for (int kk = 0; kk < 3; kk++)
	{
	  faceCoords[jj][kk] = domainMgr_->coordList_[nid*3 + kk];
	}//end for(kk)
      }//end for(jj)

      float GN[2][4], invJac[2][2], detJac;

      // Calculate area weights for each gauss point
      float xp, yp, zp;
      for (int ip = 0; ip < ngp2D; ip++)
      {
	float gp1 = gPoints2D[ip][0];
	float gp2 = gPoints2D[ip][1];
	SurfTemp->getMappedCoords(faceCoords, mappedCoords);
	SurfTemp->getGradN(gp1, gp2, GN);
	SurfTemp->getJacobian2D(GN, mappedCoords, detJac, invJac);
       
        areaFace += detJac;
      }//end for(ip)
      areaMax = max (areaMax, areaFace);

    }//end for(ll)

    float length = eleVol / areaMax;
    float dt_ele = rho * cp *  length * length / (2.0 * cond);
    dt = std::min(dt_ele, dt);

  }//end for(ie)
  dt = dt * defaultFac;
  return dt;

}//end getTimeStep

//////////////////////////////////////////////////////
//		initializeSystem		    //
//////////////////////////////////////////////////////
void 
HeatSolverManager::initializeSystem()
{

  const std::string SecondEleWidth_ = "  ";
  const std::string ThirdEleWidth_  = "    ";
  const std::string FourthEleWidth_ = "      ";
  const std::string FifthEleWidth_  = "        ";
  const std::string SixthEleWidth_  = "          ";

  float starttime, endtime;

  // Grab and initialize matrices
  nn_ = domainMgr_->nn_;
  nel_ = domainMgr_->nel_;
  thetaNp1_.resize(nn_);
  thetaN_.resize(nn_);
  rhs_.resize(nn_);
  Mvec_.resize(nn_);

  // BC Manager
  starttime = clock();
  heatBCManager_ = new BCManager(thetaNp1_, meshObj_,
                                 domainMgr_,thetaN_, rhs_);

  heatBCManager_->assignFixedNodes();
  
  heatBCManager_->initializeBoundaries();
  endtime = clock();
  cout << SecondEleWidth_ << "Timer for setting up boundary system "
       << (float) (endtime - starttime) / CLOCKS_PER_SEC << endl;

  // Initial Conditions
  initializeTemp();

  // Element Stiffness matrix
  starttime = clock();
  initializeEleStiff();
  endtime = clock();
  cout << SecondEleWidth_ << "Timer for calculating initial element level stiff matrices "
       << (float) (endtime - starttime) / CLOCKS_PER_SEC << endl;

  // Zero out force vector
  fill(rhs_.begin(), rhs_.end(), 0.0);

  // Construct mass matrix
  fill(Mvec_.begin(), Mvec_.end(), 0.0);
  starttime = clock();
  initializeMass();
  endtime = clock();
  cout << SecondEleWidth_ << "Timer for assembling Lumped mass "
       << (float) (endtime - starttime) / CLOCKS_PER_SEC << endl;

  // Find minimum time step size
  starttime = clock();
  dt_ = getTimeStep();
  if (meshObj_->inputDt_ < dt_)
  {
    dt_ = meshObj_->inputDt_;
    cout << "      Using user-defined time step, smaller than critical timestep\n";
  }
  endtime = clock();
  cout << SecondEleWidth_ << "Timer for calculating critical timestep "
       << (float) (endtime - starttime) / CLOCKS_PER_SEC << endl;
}//end initializeSystem

//////////////////////////////////////////////////////
//    		    pre_work   		      	    //
//////////////////////////////////////////////////////
void 
HeatSolverManager::pre_work()
{
  // Zero out
  fill(rhs_.begin(), rhs_.end(), 0.0);
//  fill(Mvec_.begin(), Mvec_.end(), 0.0);

  domainMgr_->updateActiveElements(thetaN_, initTheta_);

  updateMassBirth();

  heatBCManager_->updateBirthSurfaces();

  heatBCManager_->updateTool();

}//end pre_work()

//////////////////////////////////////////////////////
//    		    updateCap   		    //
//////////////////////////////////////////////////////
void 
HeatSolverManager::updateCap()
{
  int mapIndex[8][8] = { { 0,  1,  2,  3,  4,  5,  6,  7 },
                         { 1,  8,  9, 10, 11, 12, 13, 14 },
                         { 2,  9, 15, 16, 17, 18, 19, 20 },
                         { 3, 10, 16, 21, 22, 23, 24, 25 },
                         { 4, 11, 17, 22, 26, 27, 28, 29 },
                         { 5, 12, 18, 23, 27, 30, 31, 32 },
                         { 6, 13, 19, 24, 28, 31, 33, 34 },
                         { 7, 14, 20, 25, 29, 32, 34, 35 } };

  float parCoords[8][3] = { {-1.0, -1.0, -1.0},{ 1.0, -1.0, -1.0},
                             { 1.0,  1.0, -1.0},{-1.0,  1.0, -1.0},
                             {-1.0, -1.0,  1.0},{ 1.0, -1.0,  1.0},
                             { 1.0,  1.0,  1.0},{-1.0,  1.0,  1.0} };
  for (int j = 0; j < 8; j++)
    for (int i = 0; i < 3; i++)
      parCoords[j][i] *= 0.5773502692;

  float weight[8] = { 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0 };

  for (int ie = 0; ie < domainMgr_->activeElements_.size(); ie++)
  {
    int eID = domainMgr_->activeElements_[ie];
    Element * element = domainMgr_->elementList_[eID];
    element->matManager_->execute();
    int *localNodes = element->nID_;

    if (element->matManager_->updateMass_)
    {
      float *massTemp = new float [36];
      element->matManager_->updateMass_ = false;
      for (int iCom = 0; iCom < 36; iCom++) massTemp[iCom] = 0.0;

      float nodalCoords[24];

      for (int iNode = 0; iNode < 8; iNode++)
      {
	int nid = localNodes[iNode];
	for (int kk = 0; kk < 3; kk++)
	{
	  nodalCoords[kk + iNode*3] = domainMgr_->coordList_[nid*3+kk];
	}//end for(kk)
      }//end for(iNode)

      for (int ip = 0; ip < 8; ip++)
      { 
	float deriv[24],gradN[24], iJac[9];
	float detJac = 0.0;
	float *N = new float[8];
	int offSetIp = ip * 8;

	// zero everything out for this ip
	for (int i = 0; i < 24; i++)
	{
	  deriv[i] = 0.0;
	  gradN[i] = 0.0;
	  if (i<9) iJac[i] = 0.0;
	}//end for(i)

	N = &Nip_[offSetIp];
	float cpIp = element->cpIp_[ip];
        float rhoIp = element->rho_;

	// Calculate saved mass matrix
	int count = 0;
	for (int I = 0; I < 8; I++)
	{
	  for (int J = I; J < 8; J++)
	  {
	    massTemp[count] += N[I] * rhoIp * cpIp * N[J] * element->volWeight_[ip];
	    count++;
	  }//end for(J)
	}//end for(I)

      }//end for(ip)

      // Add in differences
      for (int I = 0; I < 8; I++)
      {
	int IG = localNodes[I];
	for (int J = 0; J < 8; J++)
	{
	  int stiffInd = mapIndex[I][J];
	  Mvec_[IG] += massTemp[stiffInd] - element->massMatrix_[stiffInd];
	}//end for(J)
      }//end for(I)
      element->massMatrix_ = massTemp;
    }//end if check
  }//end for(ie)
}//end updateCap()

//////////////////////////////////////////////////////
//    		    integrateForce   		    //
//////////////////////////////////////////////////////
void 
HeatSolverManager::integrateForce()
{
  // Diffusion
  getInternalForce();

  // External fluxes
  heatBCManager_->applyFluxes();

}//end integrateForce

//////////////////////////////////////////////////////
//            advance_time_step   		    //
//////////////////////////////////////////////////////
void 
HeatSolverManager::advance_time_step()
{
  for (int ii = 0; ii < domainMgr_->activeNodes_.size(); ii++)
  {
    int IG = domainMgr_->activeNodes_[ii];
    thetaNp1_[IG] = thetaN_[IG] + dt_ * (rhs_[IG])/Mvec_[IG];

    if (  std::isnan(thetaNp1_[IG]) )
    {
      cout << "****************************************************************\n";
      cout << "WARNING! Nan value occured\n";
      cout << "Node: " << IG << " RHS Value: " << rhs_[IG] << " Mass Value: " << Mvec_[IG] << "\n";
      cout << "****************************************************************\n";
      exit(EXIT_FAILURE);
    }
  }

}//end advance_time_step

//////////////////////////////////////////////////////
//    		    post_work   		    //
//////////////////////////////////////////////////////
void 
HeatSolverManager::post_work()
{
  heatBCManager_->prescribeDirichletBC();

  thetaN_.swap(thetaNp1_);
 
  outputProbeData();

  if ( meshObj_->calcEnergy_) outputEnergyData();

}//end post_work()


//////////////////////////////////////////////////////
//		outputProbeData		            //
//////////////////////////////////////////////////////
void 
HeatSolverManager::outputProbeData()
{
  float starttime = clock();
  HexahedralElement *HexTemp = new HexahedralElement;
  float N[8];
  float thetaNodes[8];
  for (int iprobe = 0; iprobe < domainMgr_->probeList_.size(); iprobe++)
  {
    // calculate temperature data
    int eID = domainMgr_->probeList_[iprobe]->elementID_;
    Element *elementProbe = domainMgr_->elementList_[eID];
    int *localNodes = elementProbe->nID_;
    for (int I = 0; I < 8; I++)
    {
      int nid = localNodes[I];
      thetaNodes[I] = thetaN_[nid];
    }
    HexTemp->shape_fcn(domainMgr_->probeList_[iprobe]->parCoords_, N);
    
    float thetaIp = 0.0;
    for (int I = 0; I < 8; I++) thetaIp += N[I] * thetaNodes[I];

    ofstream outFile;
    string fileName = meshObj_->probeNames_[iprobe];
    // print out data
    if (domainMgr_->isInit_)
    {
      outFile.open(fileName, ios::out);
    }
    else
    {
      outFile.open(fileName, ios::app);
    }
    outFile << domainMgr_->currTime_ << " , " << thetaIp << endl;
    outFile.close(); 
  }//end for(iprobe)
  float endtime = clock();
  probeTime_ += (float) (endtime - starttime) / CLOCKS_PER_SEC;
}//end outputProbeData

//////////////////////////////////////////////////////
//		outputEnergyData	            //
//////////////////////////////////////////////////////
void 
HeatSolverManager::outputEnergyData()
{

  float sumEnergy = 0.0;
  for (int ii = 0; ii < domainMgr_->activeNodes_.size(); ii++)
  {
    int IG = domainMgr_->activeNodes_[ii];
    // ** swapped rigiht before this, so need T^n - T^(n+1)
    sumEnergy += (thetaN_[IG]- thetaNp1_[IG]) / dt_ * Mvec_[IG];
  }


  // print out data
  string fileName = meshObj_->energyFileName_;
  ofstream outFile;
  if (domainMgr_->isInit_)
  {
    outFile.open(fileName, ios::out);
  }
  else
  {
    outFile.open(fileName, ios::app);
  }

  outFile << domainMgr_->currTime_ << " , " << sumEnergy << endl;
  outFile.close(); 
}//end outputEnergyData
