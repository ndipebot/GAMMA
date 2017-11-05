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

// user-defined headers
#include <Mesh.h>
#include <Element.h>
#include "dynaInput.h"
#include <Surface.h>
#include <BCManager.h>
#include <FluxManager.h>
#include <ConvManager.h>
#include <MovingFlxManager.h>
#include <RadManager.h>
#include <DomainManager.h>

//////////////////////////////////////////////////////
//		Constructor			    //
//////////////////////////////////////////////////////
BCManager::BCManager(vector<float> &thetaNp1,
                     Mesh *meshObj,
                     DomainManager *domainMgr,
                     vector<float> &thetaN,
                     vector<float> &rhs)
                     : thetaNp1_(thetaNp1),
                       meshObj_(meshObj),
                       domainMgr_(domainMgr),
 		       thetaN_(thetaN),
                       rhs_(rhs)
{
}

//////////////////////////////////////////////////////
//		prescribeDirichletBC		    //
//////////////////////////////////////////////////////
void 
BCManager::prescribeDirichletBC()
{
  for (int jj = 0; jj < fixedNodeIDs_.size(); jj++)
  {
    int nid = fixedNodeIDs_[jj];
    thetaNp1_[nid] = fixedNodeVals_[jj];
  }//end for(jj)
}//end prescribeDirichletBC

//////////////////////////////////////////////////////
//		assignSurfGP			    //
//////////////////////////////////////////////////////
void 
BCManager::assignSurfGP(vector<Surface> &surfaceList)
{
  // Calculate mapped areas for each GP point of surfaces
  int ngp2D = 4;
  float gPoints2D [4][2] = { {-1.0/sqrt(3.0), -1.0/sqrt(3.0)},
                              {-1.0/sqrt(3.0),  1.0/sqrt(3.0)},
                              { 1.0/sqrt(3.0), -1.0/sqrt(3.0)}, 
                              { 1.0/sqrt(3.0),  1.0/sqrt(3.0)} };
  float weights[4] = {1.0, 1.0, 1.0, 1.0};
  for (int ii = 0; ii < surfaceList.size(); ii++)
  {
    Surface surfI;
    int plane = surfaceList[ii].plane_;
    int birthElement = surfaceList[ii].birthElement_;
    int *surfNodes = new int[4];
    float faceCoords[4][3];
    float elemCoords[8][3];
    float mappedCoords[4][2]; 
    Element *element = domainMgr_->elementList_[birthElement];
    int *localNodes = element->nID_;
    float *Nsurf = new float[8];
    surfaceList[ii].gpCoords_ = new float[ngp2D*3];
    surfaceList[ii].areaWeight_ = new float[4];
    float gpNsurf[4][8];

    for (int iNode = 0; iNode < 8; iNode++)
    {
      int nid = localNodes[iNode];
      for (int kk = 0; kk < 3; kk++)
      {
        elemCoords[iNode][kk] = domainMgr_->coordList_[nid*3 + kk];
      }//end for(kk)
    }//end for(iNode)
    
    if (plane == 0)
    {
      surfNodes[0] = localNodes[7];
      surfNodes[1] = localNodes[6];
      surfNodes[2] = localNodes[5];
      surfNodes[3] = localNodes[4];
      for (int ip = 0; ip < ngp2D; ip++)
      {
        surfI.getShapeFcn(Nsurf, gPoints2D[ip][0], gPoints2D[ip][1], 1.0);
        for (int I = 0; I < 8; I++) gpNsurf[ip][I] = Nsurf[I];
      }//end for(ip)

    }
    else if(plane == 1)
    {
      surfNodes[0] = localNodes[0];
      surfNodes[1] = localNodes[1];
      surfNodes[2] = localNodes[2];
      surfNodes[3] = localNodes[3];
      for (int ip = 0; ip < ngp2D; ip++)
      {
        surfI.getShapeFcn(Nsurf, gPoints2D[ip][0], gPoints2D[ip][1], -1.0);
        for (int I = 0; I < 8; I++) gpNsurf[ip][I] = Nsurf[I];
      }//end for(ip)
    }
    else if(plane == 2)
    {
      surfNodes[0] = localNodes[0];
      surfNodes[1] = localNodes[4];
      surfNodes[2] = localNodes[5];
      surfNodes[3] = localNodes[1];
      for (int ip = 0; ip < ngp2D; ip++)
      {
        surfI.getShapeFcn(Nsurf, gPoints2D[ip][0], -1.0, gPoints2D[ip][1]);
        for (int I = 0; I < 8; I++) gpNsurf[ip][I] = Nsurf[I];
      }//end for(ip)
    }
    else if(plane == 3)
    {
      surfNodes[0] = localNodes[3];
      surfNodes[1] = localNodes[2];
      surfNodes[2] = localNodes[6];
      surfNodes[3] = localNodes[7];
      for (int ip = 0; ip < ngp2D; ip++)
      {
        surfI.getShapeFcn(Nsurf, gPoints2D[ip][0], 1.0, gPoints2D[ip][1]);
        for (int I = 0; I < 8; I++) gpNsurf[ip][I] = Nsurf[I];
      }//end for(ip)
    }
    else if(plane == 4)
    {
      surfNodes[0] = localNodes[0];
      surfNodes[1] = localNodes[4];
      surfNodes[2] = localNodes[7];
      surfNodes[3] = localNodes[3];
      for (int ip = 0; ip < ngp2D; ip++)
      {
        surfI.getShapeFcn(Nsurf, -1.0, gPoints2D[ip][0], gPoints2D[ip][1]);
        for (int I = 0; I < 8; I++) gpNsurf[ip][I] = Nsurf[I];
      }//end for(ip)
    }
    else if(plane == 5)
    {
      surfNodes[0] = localNodes[1];
      surfNodes[1] = localNodes[2];
      surfNodes[2] = localNodes[6];
      surfNodes[3] = localNodes[5];
      for (int ip = 0; ip < ngp2D; ip++)
      {
        surfI.getShapeFcn(Nsurf,  1.0, gPoints2D[ip][0], gPoints2D[ip][1]);
        for (int I = 0; I < 8; I++) gpNsurf[ip][I] = Nsurf[I];
      }//end for(ip)
    }

    // pick out nodal coordinates
    for (int jj = 0; jj < 4; jj++)
    {
      int nid = surfNodes[jj];
      for (int kk = 0; kk < 3; kk++)
      {
        faceCoords[jj][kk] = domainMgr_->coordList_[nid*3 + kk];
      }//end for(kk)
    }//end for(jj)

    // save off surface nodes
    surfaceList[ii].surfaceNodes_ = surfNodes;

    float GN[2][4], invJac[2][2], detJac;

    // Calculate area weights for each gauss point
    float xp, yp, zp;
    for (int ip = 0; ip < ngp2D; ip++)
    {
      xp = 0.0;
      yp = 0.0;
      zp = 0.0;
      float gp1 = gPoints2D[ip][0];
      float gp2 = gPoints2D[ip][1];
      surfI.getMappedCoords(faceCoords, mappedCoords);

      //save mapped coords
      for(int e = 0; e < 4; ++e)
    	  for(int b = 0; b < 2; b++)
			  surfaceList[ii].mappedCoords.push_back(mappedCoords[e][b]);

      //
      surfI.getGradN(gp1, gp2, GN);
      surfI.getJacobian2D(GN, mappedCoords, detJac, invJac);
      surfaceList[ii].areaWeight_[ip] = detJac * weights[ip];

      for (int I = 0; I < 8; I++)
      { 
        xp += gpNsurf[ip][I] * elemCoords[I][0];
        yp += gpNsurf[ip][I] * elemCoords[I][1];
        zp += gpNsurf[ip][I] * elemCoords[I][2];
      }//end for(I)

      int offSet = ip * 3;
      surfaceList[ii].gpCoords_[offSet+0] = xp; 
      surfaceList[ii].gpCoords_[offSet+1] = yp; 
      surfaceList[ii].gpCoords_[offSet+2] = zp; 
      
    }//end for(ip)

  }//end for(ii)

}//end assignSurfGP

//////////////////////////////////////////////////////
//		assignFixedNodes		    //
//////////////////////////////////////////////////////
void 
BCManager::assignFixedNodes()
{
  for (map<int, vector<int> >::iterator it = meshObj_->nodeSets_.begin();
       it != meshObj_->nodeSets_.end(); it++)
  {
    int setID = it->first;
    vector<int> loadIDs = meshObj_->loadSets_[setID];
    for (int ii = 0; ii < loadIDs.size(); ii++)
    {
      if (loadIDs[ii] == 1)
      {
        vector<int> fixedSets = it->second;
        for (int jj = 0; jj < fixedSets.size(); jj++)
        {
          int gnid = fixedSets[jj];
          int lnid = domainMgr_->node_global_to_local_[gnid];
          vector <float> temp = meshObj_->loadSetVals_[setID];
          fixedNodeIDs_.push_back(lnid);
          fixedNodeVals_.push_back(temp[0]);
        }//end for(Jj)
      }//end if check
    }//end for(ii)
  }//end for(it)

}//end assignFixedNodes

//////////////////////////////////////////////////////
//		attachBirthSurf			    //
//////////////////////////////////////////////////////
void 
BCManager::attachBirthSurf()
{

  // Generate surface lists and their death times
  // ** NOTE ** birthElement_ will own surface of interest
  for (int ii = 0; ii < meshObj_->birthID_.size(); ii++)
  {
    int gbID = meshObj_->birthID_[ii];
    int lbID = domainMgr_->element_global_to_local_[gbID];
    Element *element = domainMgr_->elementList_[lbID];
    float birthTimeCurrent = element->birthTime_;
    for (int jj = 0; jj < 6; jj++)
    {
      int eIDCheck = domainMgr_->connSurf_[lbID][jj];
      if (eIDCheck != -1)
      { 
        Element *eleNeigh = domainMgr_->elementList_[eIDCheck];
        bool isbirth =  eleNeigh->birth_;
        if (isbirth)
        {
          float birthTimeNeighbor = eleNeigh->birthTime_;
          if (birthTimeNeighbor > birthTimeCurrent)
          {
            Surface surfI;
            surfI.birthElement_ = lbID;
            surfI.deathElement_ = eIDCheck;
            surfI.deathTime_ = birthTimeNeighbor;
            surfI.birthTime_ = birthTimeCurrent;
            surfI.plane_ = jj;
            surfI.isDynamic_ = true;
            surfaceList_.push_back(surfI);
          }
        }
        else
        {
	  Surface surfI;
	  surfI.birthElement_ = lbID;
	  surfI.deathElement_ = lbID;
	  surfI.deathTime_ = birthTimeCurrent;
	  surfI.birthTime_ = 0.0;
	  surfI.isDynamic_ = true;
          surfI.plane_ = jj;
	  surfaceList_.push_back(surfI);
        }//end if(isBirth)
      }
      else
      {
	Surface surfI;
	surfI.birthElement_ = lbID;
	surfI.deathElement_ = -1;
	surfI.deathTime_ = 1.0e10;
	surfI.birthTime_ = birthTimeCurrent;
	surfI.plane_ = jj;
	surfI.isDynamic_ = true;
	surfaceList_.push_back(surfI);
      }//end if(eIDCheck)
    }//end for(jj)
  }//end for(ii)

	//sort surfaces by birthTime
	std::sort(surfaceList_.begin(), surfaceList_.end(), [](Surface a, Surface b) {return a.birthTime_ < b.birthTime_; });


  for (int I = 0; I < surfaceList_.size(); I++)
  {
    birthSurfaceList_.push_back(I);
  }//end for(I)

}//attachBirthSurf

//////////////////////////////////////////////////////
//		attachStaticSurf	    	    //
//////////////////////////////////////////////////////
void 
BCManager::attachStaticSurf()
{
  for (int ii = 0; ii < domainMgr_->numElEl_.size(); ii++)
  {
    int numNeigh = domainMgr_->numElEl_[ii];
    Element *element = domainMgr_->elementList_[ii];
    if (numNeigh != 26 && !element->birth_)
    {
      for (int jj = 0; jj < 6; jj++)
      {
        int neighElement = domainMgr_->connSurf_[ii][jj];
        if (neighElement == -1)
        {
          Surface surfI;
          surfI.birthElement_ = ii;
          surfI.deathElement_ = -1;
          surfI.birthTime_ = 0.0;
          surfI.deathTime_ = 1.0e10;
          surfI.plane_ = jj;
          surfI.isDynamic_ = false;
          staticSurfList_.push_back(surfI);
        }
      }//end for(jj)
    }
  }//end for(ii)
}//end attachStaticSurf

//////////////////////////////////////////////////////
//		attachSurfNset		    	    //
//////////////////////////////////////////////////////
void 
BCManager::attachSurfNset(vector <Surface> &surfaceList)
{
  // assign node set ids to surfaces
  for (map<int, vector<int> >::iterator it = meshObj_->nodeSets_.begin();
       it != meshObj_->nodeSets_.end(); it++)
  {
    int nSetID            = it->first;
    vector <int> nSetList = it->second;

    sort(nSetList.begin(), nSetList.end());
    for (int ii = 0; ii < surfaceList.size(); ii++)
    {
      int surfCt = 0;
      Surface surfI = surfaceList[ii];
      int birthElement = surfI.birthElement_;
      Element *element = domainMgr_->elementList_[birthElement];
      int *localNodes = element->nID_;
      int plane = surfI.plane_;
      int surfaceNodes[4];
      if (plane == 0)
      {
        surfaceNodes[0] = 4;
        surfaceNodes[1] = 5;
        surfaceNodes[2] = 6;
        surfaceNodes[3] = 7;
      }
      else if (plane == 1)
      {
        surfaceNodes[0] = 0;
        surfaceNodes[1] = 1;
        surfaceNodes[2] = 2;
        surfaceNodes[3] = 3;
      }
      else if (plane == 2)
      {
        surfaceNodes[0] = 0;
        surfaceNodes[1] = 1;
        surfaceNodes[2] = 4;
        surfaceNodes[3] = 5;
      }
      else if (plane == 3)
      {
        surfaceNodes[0] = 3;
        surfaceNodes[1] = 2;
        surfaceNodes[2] = 6;
        surfaceNodes[3] = 7;
      }
      else if (plane == 4)
      {
        surfaceNodes[0] = 0;
        surfaceNodes[1] = 3;
        surfaceNodes[2] = 7;
        surfaceNodes[3] = 4;
      }
      else if (plane == 5)
      {
        surfaceNodes[0] = 1;
        surfaceNodes[1] = 2;
        surfaceNodes[2] = 6;
        surfaceNodes[3] = 5;
      }
      
      for (int jj = 0; jj < 4; jj++)
      {
        int nodeCheck = domainMgr_->node_local_to_global_[ localNodes[ surfaceNodes[jj] ] ];
        if ( binary_search(nSetList.begin(), nSetList.end(), nodeCheck) )
        { 
          surfCt++;
        }
      }//end for(jj)

      if (surfCt == 4)
      {
        surfaceList[ii].setID_.push_back(nSetID);
      }
    }//end for(ii)
  }//end for(it)
}//end attachSurfNset

//////////////////////////////////////////////////////
//		attachSurfBC		    	    //
//////////////////////////////////////////////////////
void 
BCManager::attachSurfBC(vector<Surface> &surfaceList)
{
  for (int ii = 0; ii < surfaceList.size(); ii++)
  {
    Surface surfI = surfaceList[ii]; 
    vector<int> surfSetID = surfI.setID_;
    for (int jj = 0; jj < surfSetID.size(); jj ++)
    {
      vector<int> bcIDVec = meshObj_->loadSets_[surfSetID[jj]];
      for (int kk = 0; kk < bcIDVec.size(); kk++)
      {
        int bcID = bcIDVec[kk];
        if (bcID == 1)
        {
          surfaceList[ii].isFixed_ = true;
          surfaceList[ii].isFlux_ = false;
        } 
        else
        {
          surfaceList[ii].isFixed_ = false;
          surfaceList[ii].isFlux_ = true;
        }
      }//end for(kk)
    }//end for(jj)
  }//end for(ii)
}//end attachSurfBC

//////////////////////////////////////////////////////
//		assignSurfFluxAlg    	    	    //
//////////////////////////////////////////////////////
void
BCManager::assignSurfFluxAlg(vector<Surface> &surfaceList)
{
  // Set up BC managers  
  float small = 1.0e-8;
  float *Nip = new float[16];
  float gPoints2D [4][2] = { {-1.0/sqrt(3.0), -1.0/sqrt(3.0)},
                              {-1.0/sqrt(3.0),  1.0/sqrt(3.0)},
                              { 1.0/sqrt(3.0), -1.0/sqrt(3.0)}, 
                              { 1.0/sqrt(3.0),  1.0/sqrt(3.0)} };
  int ngp2D = 4;
  for (int ip = 0; ip < ngp2D; ip++)
  {
    float chsi = gPoints2D[ip][0];
    float eta  = gPoints2D[ip][1];
    Nip[ip*4 + 0] = 0.25 * (1 - chsi) * ( 1 - eta);
    Nip[ip*4 + 1] = 0.25 * (1 + chsi) * ( 1 - eta);
    Nip[ip*4 + 2] = 0.25 * (1 + chsi) * ( 1 + eta);
    Nip[ip*4 + 3] = 0.25 * (1 - chsi) * ( 1 + eta);

  }//end for(ip)
  
  for (int ii = 0; ii < surfaceList.size(); ii++)
  {
	/*
	 * Ebot Mod
	 *
	 * set flux conditions
	 */
	 surfaceList[ii].flux.resize(5,0);
	///

    int *surfaceNodes;
    surfaceNodes = surfaceList[ii].surfaceNodes_;
    float *areaWeight = surfaceList[ii].areaWeight_;
    vector <int> setID = surfaceList[ii].setID_;
    if (surfaceList[ii].isFlux_)
    {
      for (int jj = 0; jj < setID.size(); jj++)
      {
        vector<int> loadID = meshObj_->loadSets_[setID[jj]];
        vector<float> loadVals = meshObj_->loadSetVals_[setID[jj]];
        for (int kk = 0; kk < loadID.size(); kk++)  
        {
          if (loadID[kk] == 3) //natural convection
          {
        	//set flux type indicator
        	surfaceList[ii].flux[0] = 1.;
            float hconv = loadVals[kk];
            surfaceList[ii].flux[1] = hconv;
	    FluxManager *fluxAlg = new ConvManager(meshObj_->Rambient_, hconv, surfaceNodes, Nip,
						   areaWeight, thetaN_, rhs_);
	    surfaceList[ii].fluxManagerVec_.push_back(fluxAlg);

          }
          if (loadID[kk] == 4) //radiative convection
          {
        	surfaceList[ii].flux[2] = 1.;
            float epsilon = loadVals[kk];
            surfaceList[ii].flux[3] = epsilon;
	    FluxManager *fluxAlg = new RadManager(meshObj_->Rambient_, epsilon, meshObj_->Rabszero_, 
                                                  meshObj_->Rboltz_, surfaceNodes, Nip, areaWeight, thetaN_, rhs_);
	    surfaceList[ii].fluxManagerVec_.push_back(fluxAlg);
          }
          if (loadID[kk] == 5) //moving flux (laser)
          {
          
            if ( fabs(surfaceList[ii].unitNorm_[2]) > small)
            {
            	surfaceList[ii].flux[4] = 1.;
      	   float Qin = meshObj_->Qin_ * meshObj_->Qeff_;
      	   float rBeam = meshObj_->rBeam_;
	      FluxManager *fluxAlg = new MovingFlxManager(surfaceList[ii].gpCoords_, tooltxyz_, 
                                                          laserState_, rBeam, Qin, surfaceNodes, 
                                                          Nip, areaWeight, thetaN_, rhs_);
	      surfaceList[ii].fluxManagerVec_.push_back(fluxAlg);
            }
          }//end if loadID
        }//end for(kk)
      }//end for(jj)
    }//if isflux
  }//end for(ii)

}//end assignSurfBC

//////////////////////////////////////////////////////
//		assignActiveStaticSurf	    	    //
//////////////////////////////////////////////////////
void
BCManager::assignActiveStaticSurf()
{
  for (int I = 0; I < staticSurfList_.size(); I++)
  {
    if (staticSurfList_[I].isFlux_)
    {
      activeSurfaces_.push_back(I);
    }
  }//end for(I)

}//end assignActiveStaticSurf

//////////////////////////////////////////////////////
//		updateBirthSurfaces	    	    //
//////////////////////////////////////////////////////
void
BCManager::updateBirthSurfaces()
{

  vector<int> ::iterator itBirth;
  vector<int> ::iterator itDeath;

  // Update birth
  for (itBirth = birthSurfaceList_.begin(); itBirth != birthSurfaceList_.end();)
  {
    int surfID = *itBirth;
    if (surfaceList_[surfID].birthTime_ <= domainMgr_->currTime_)
    {
      activeBirthSurfaces_.push_back(surfID);
      itBirth = birthSurfaceList_.erase(itBirth);
    }
    else
    {
      itBirth++;
    }
  }//end for(itBirth)

  // Update Death
  for (itDeath = activeBirthSurfaces_.begin(); itDeath != activeBirthSurfaces_.end();)
  {
    int surfID = *itDeath;
    if (surfaceList_[surfID].deathTime_ <= domainMgr_->currTime_ &&
        surfaceList_[surfID].deathTime_ > 0.0)
    {
      itDeath = activeBirthSurfaces_.erase(itDeath);
    }
    else
    {
      itDeath++;
    }
  }//end for(itDeath)


}


//////////////////////////////////////////////////////
//	   	   updateTool	     	    	    //
//////////////////////////////////////////////////////
void
BCManager::updateTool()
{
  float tx, ty, tz;
  for (int ii = 1; ii < domainMgr_->tooltxyz_.size(); ii++)
  {
    float *txyzNp1 = &domainMgr_->tooltxyz_[ii][0];
    float laserTimeNp1 = txyzNp1[0];
    if (domainMgr_->currTime_ <= laserTimeNp1)
    {
      float *txyzN = &domainMgr_->tooltxyz_[ii-1][0];
      float laserTimeN = txyzN[0];
      float num = domainMgr_->currTime_ - laserTimeN;
      float den = laserTimeNp1 - laserTimeN;
      float rat = num/den;
      tx = rat * (txyzNp1[1] - txyzN[1]) + txyzN[1];
      ty = rat * (txyzNp1[2] - txyzN[2]) + txyzN[2];
      tz = rat * (txyzNp1[3] - txyzN[3]) + txyzN[3];
      laserState_ = domainMgr_->laserOn_[ii];
      break;
    }//end if
  }//end for(ii)
/*  float velocity = 0.0;
  float xc = 0.5;
  float yc = 0.25;
  laserState_ = 1;

  tooltxyz_[0] = xc + velocity * domainMgr_->currTime_;
  tooltxyz_[1] = yc;*/
  tooltxyz_[0] = tx;
  tooltxyz_[1] = ty;
  tooltxyz_[2] = tz;
}//end updateTool

//////////////////////////////////////////////////////
//		initializeBoundaries	    	    //
//////////////////////////////////////////////////////
void
BCManager::initializeBoundaries()
{

  attachBirthSurf();

  attachStaticSurf();

  attachSurfNset(surfaceList_);
  attachSurfNset(staticSurfList_);

  attachSurfBC(surfaceList_);
  attachSurfBC(staticSurfList_);

  // Attach surface nodes and gauss points
  assignSurfGP(surfaceList_);
  assignSurfGP(staticSurfList_);

  // find unit normals of each surface
  assignUnitNorm(surfaceList_);
  assignUnitNorm(staticSurfList_);

  // Push back flux managers
  assignSurfFluxAlg(staticSurfList_);
  assignSurfFluxAlg(surfaceList_);

  // find active static surfaces
  assignActiveStaticSurf();

}

//////////////////////////////////////////////////////
//		   applyFluxes		    	    //
//////////////////////////////////////////////////////
void
BCManager::applyFluxes()
{
  // Boundary conditions (static surfaces)
  for (int I = 0; I < activeSurfaces_.size(); I++)
  {
    int surfID = activeSurfaces_[I];
    if (staticSurfList_[surfID].isFlux_)
    {
      for (int kk = 0; kk < staticSurfList_[surfID].fluxManagerVec_.size(); kk++)
      {
	staticSurfList_[surfID].fluxManagerVec_[kk]->execute();
      }
    }
  }//end for(I)


  // Boundary conditions (birth surfaces)
  for (int I = 0; I < activeBirthSurfaces_.size(); I++)
  {
    int surfID = activeBirthSurfaces_[I];
    if (surfaceList_[surfID].isFlux_)
    {
      for (int kk = 0; kk < surfaceList_[surfID].fluxManagerVec_.size(); kk++)
      {
	surfaceList_[surfID].fluxManagerVec_[kk]->execute();
      }
    }
  }//end for(I)
}

//////////////////////////////////////////////////////
//		   assignUnitNorm		    //
//////////////////////////////////////////////////////
void
BCManager::assignUnitNorm(vector <Surface> &surfaceList)
{
  float surfCoords [4][3];
  float *v1, *v2, *v3, *e1, *e2, *nvec;
  e1 = new float [3];
  e2 = new float [3];

  for (int ii = 0; ii < surfaceList.size(); ii++)
  {
    int *surfaceNodes;
    surfaceNodes = surfaceList[ii].surfaceNodes_;
    nvec = new float[3];

    // unpack coordinates
    for (int I = 0; I < 4; I++)
    {
      int IG = surfaceNodes[I];
      for (int kk = 0; kk < 3; kk++)
      {
	surfCoords[I][kk] = domainMgr_->coordList_[IG*3 + kk];
      }//end for(kk)
    }//end for(I)

    v1 = surfCoords[0];
    v2 = surfCoords[1]; 
    v3 = surfCoords[2];

    // calculate edges, hold v1 as origin
    for (int kk = 0; kk < 3; kk++)
    {
      e2[kk] = v2[kk] - v1[kk];
      e1[kk] = v3[kk] - v1[kk];
    }//end for(kk)

    // calculate normal (cross product)
    float normVec = 0.0;
    nvec[0] = e1[1] * e2[2] - e1[2] * e2[1];
    nvec[1] = - ( e1[0] * e2[2] - e1[2] * e2[0]);
    nvec[2] = e1[0] * e2[1] - e1[1] * e2[0];

    normVec = sqrt ( nvec[0] * nvec[0] + 
                     nvec[1] * nvec[1] +
                     nvec[2] * nvec[2] );
    // Normalize
    for (int kk = 0; kk < 3; kk ++)
    {
      nvec[kk] = nvec[kk]/normVec;
    }

    surfaceList[ii].unitNorm_ = nvec;

  }//end for(ii)

}
