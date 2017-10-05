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
#include "Mesh.h"
#include "Element.h"
#include "dynaInput.h"
#include "Surface.h"

//////////////////////////////////////////////////////
//		createNodeMap			    //
//////////////////////////////////////////////////////
void createNodeMap(map<int, int> &node_global_to_local, vector<double> &coordList,
                   Mesh * meshObj)
{

  // Create local-global node mapping
  int nodeCt = 0;
  for ( map<int, vector<double> >::iterator it = meshObj->NODES_.begin();
        it != meshObj->NODES_.end(); it++)
  {
    int global_nid = it->first;
    vector <double> gCoords = it->second;
    node_global_to_local[global_nid] = nodeCt;
    for (int ii = 0; ii < gCoords.size(); ii++)
    {
      coordList.push_back(gCoords[ii]);
    }
    nodeCt++;
  }

}//end createNodeMap

//////////////////////////////////////////////////////
//		createElementList		    //
//////////////////////////////////////////////////////
void createElementList(Mesh *meshObj, vector<Element*> &elementList,
                       map<int, int> &element_global_to_local,
                       map<int, int> node_global_to_local)
{
  // Create element classes
  int eleCnt = 0;
  for ( map<int, vector<int> >::iterator it = meshObj->ELEM_.begin();
        it != meshObj->ELEM_.end(); it++)
  {
    HexahedralElement* ele = new HexahedralElement;
    int elementID = it->first;
    vector<int> localNode = it->second;
    element_global_to_local[elementID] = eleCnt;
    for (int ii = 0; ii<localNode.size(); ii++)
    {
      int global_nid = localNode[ii];
      ele->nID_[ii] = node_global_to_local[global_nid];
    }//end for(ii)
    ele->birthTime_ = 0.0;
    ele->birth_=false;
    elementList.push_back(ele);
    eleCnt++;
  }//end for(it)

}//end createElementList


//////////////////////////////////////////////////////
//		assignElePID		    	    //
//////////////////////////////////////////////////////
void assignElePID(Mesh *meshObj, 
                  map<int, int> element_global_to_local,
                  vector<Element*> &elementList)
{
  for (map<int, vector<int> >::iterator it = meshObj->PIDS_.begin();
       it != meshObj->PIDS_.end(); it++)
  {
    int PID = it->first;
    vector <double> eleMat = meshObj->PID_to_MAT_[PID];
    vector<int> pidEleList = it->second;
    for (int ii = 0; ii < pidEleList.size(); ii++)
    {
      int elePID = pidEleList[ii];
      int localEle = element_global_to_local[elePID];
      Element *element = elementList[localEle];
      // assign material properties to element
      element->PID_ = PID;
      element->cond_ = eleMat[5];
      element->rho_ = eleMat[0];
      element->cp_ = eleMat[4];
      element->liquidus_ = eleMat[2];
      element->solidus_ = eleMat[1];
      element->latent_ = eleMat[3];
    }//end for(ii)
  }//end for(it)
}//end assignElePID


//////////////////////////////////////////////////////
//		checkElementJac			    //
//////////////////////////////////////////////////////
void checkElementJac(vector<Element*> elementList, vector<double> coordList,
                     map<int, int> node_global_to_local)
{
  for (int ii = 0; ii < elementList.size(); ii++)
  {
    double nodalCoords[24];
    
    //unpack local nodal coordinates.
    Element * element = elementList[ii];
    HexahedralElement* HexTemp = new HexahedralElement;
    int *localNodes = element->nID_;
    for (int jj = 0; jj < 8; jj++)
    {
      int nid = localNodes[jj];
      for (int kk = 0; kk < 3; kk++)
      {
        nodalCoords[kk+jj*3] = coordList[nid*3 + kk];
      }//end for(kk)
    }//end for(jj)

    // Calculate element jacobian
    double deriv[24], detJac, iJac[9];
    double parCoords[3] = {0.0, 0.0, 0.0};
    HexTemp->derivative_of_shape_function_about_parametic_coords(parCoords, deriv);
    HexTemp->Jacobian(deriv, nodalCoords, iJac, detJac);
    if (detJac < 0.0)
    {
      cout << "WARNING!!!! Negative Jacobian! Element Number: " << ii << endl;
    }

  }//end for(it)
}//end checkElementJac

//////////////////////////////////////////////////////
//		assignElementBirth		    //
//////////////////////////////////////////////////////
void assignElementBirth(Mesh *meshObj,
                        map<int,int> element_global_to_local,
                        vector<Element*> &elementList)
{

  // Assign element birth times
  for (int ii = 0; ii < meshObj->birthID_.size(); ii++)
  {
    int gbID = meshObj->birthID_[ii];
    int lbID = element_global_to_local[meshObj->birthID_[ii]];
    double bTime = meshObj->birthTime_[ii];
    elementList[lbID]->birthTime_ = bTime;
    elementList[lbID]->birth_ = true;
 
  }//end for(ii)

}//end assignElementBirth

//////////////////////////////////////////////////////
//		createElElConn			    //
//////////////////////////////////////////////////////
void createElElConn(vector<Element*> elementList, 
                    vector<int> &connVec, vector <int> &conn_to_el_Vec,
                    vector<int> &connVecIndx, vector< vector<int> > &connElEl,
                    vector<int> &numElEl)
{
  //***********************************//
  //	# of neighbors:
  //  	  edges = 11 elements
  //	  surfaces = 17 elements
  //	  corners = 7 elements
  //	  interior = 26 elements
  //***********************************//
  int idxCt = 0;
  for (int ii = 0; ii < elementList.size(); ii++)
  { 
    Element *element = elementList[ii];
    int *localNodes = element->nID_;
    for (int jj = 0; jj < 8; jj++)
    {
      connVec.push_back(localNodes[jj]);
      conn_to_el_Vec.push_back(ii);
      connVecIndx.push_back(idxCt);
      idxCt++;
    }//end for(jj)
    vector<int> rowEl;
    rowEl.resize(26);
    for (int mm = 0; mm < 26; mm++) rowEl[mm] = -1;
    connElEl.push_back(rowEl);
    numElEl.push_back(0);
  }//end for(ii)

  // sort flat connectivity indices 
  sortIndxVec(connVec, connVecIndx);

  // sort flat connectivity
  sort(connVec.begin(), connVec.end());

  for (int ii = 0; ii < elementList.size(); ii++)
  {
    Element *element = elementList[ii];
    int *localNodes = element->nID_;
    for (int jj = 0; jj < 8; jj++)
    {
      int searchNode = localNodes[jj];
      vector<int>::iterator low  = lower_bound(connVec.begin(), connVec.end(), searchNode);
      vector<int>::iterator high = upper_bound(connVec.begin(), connVec.end(), searchNode);
      int lowIndx   = low - connVec.begin();
      int highIndx  = high - connVec.begin();
      for (int kk = lowIndx; kk < highIndx; kk++)
      {
        int nodeEleKK = connVecIndx[kk];
        if (ii != conn_to_el_Vec[nodeEleKK] )
        {
          for ( int ll = 0; ll < 26; ll++)
          {
            if(connElEl[ii][ll] == conn_to_el_Vec[nodeEleKK]) break;

            if(connElEl[ii][ll]==-1)
            {
              connElEl[ii][numElEl[ii]] = conn_to_el_Vec[nodeEleKK];
              numElEl[ii] = numElEl[ii] + 1;
              break;
            }
          }//end for(ll)
        }//end if
      }//end for(kk)
    }//end for(jj)
  }//end for(ii)

}//end createElElConn

//////////////////////////////////////////////////////
//		createConnSurf			    //
//////////////////////////////////////////////////////
void createConnSurf(vector<Element*> elementList, 
                    vector<int> numElEl, vector< vector<int> > connElEl,
                    vector< vector<int> > &connSurf)
{
  // set up surface connections between elements
  // Element Orientation
 /*     local node ordering is 
	CONNECTIVITY ORDERING
  connSurf storage order:
	xy(+), xy(-), xz(-), xz (+), yz(-), yz(+)
#
#			  7  o-------------- o 6
#			    /|		    /|
#			   / |		   / |
#			  /  |		  /  |
#			 /   |		 /   |
#		      4	o---------------o 5  |
#	z		|    o----------|----o
#	^    y		|   / 3		|   / 2
#	|   /		|  /		|  /
#	|  /		| /		| /
#	| /		|/		|/
#	|/	      0	o---------------o 1
#	----------->x				*/
 

  int surfCt = 0;
  int surfIndx[4];
  for (int ii = 0; ii < elementList.size(); ii++)
  {
    vector<int> rowConn(6);

    for (int mm = 0; mm < 6; mm++) rowConn[mm] = -1;

    Element *element = elementList[ii];
    int *localNodes = element->nID_;

    for (int jj = 0; jj < numElEl[ii]; jj++)
    { 
      surfCt = 0;
      Element *elementCompare = elementList[connElEl[ii][jj]];
      int *compareNodes = elementCompare->nID_;
      for (int kk = 0; kk < 8; kk++)
      {
        for (int ll = 0; ll < 8; ll++)
        {
          if (localNodes[kk] == compareNodes[ll])
          {
            surfIndx[surfCt] = kk;
            surfCt++;
            break;
          }//end if
        }//end for(ll)
      }//end for(kk)
      // Surface connections
      if (surfCt > 2)
      {
        // xy(+)
        if ( surfIndx[0] == 4 &&
             surfIndx[1] == 5 &&
             surfIndx[2] == 6 &&
             surfIndx[3] == 7)
        {
          rowConn[0] = connElEl[ii][jj];
        }
        // xy(-)
        if ( surfIndx[0] == 0 &&
             surfIndx[1] == 1 &&
             surfIndx[2] == 2 &&
             surfIndx[3] == 3)
        {
          rowConn[1] = connElEl[ii][jj];
        }
        // xz(-)
        if ( surfIndx[0] == 0 &&
             surfIndx[1] == 1 &&
             surfIndx[2] == 4 &&
             surfIndx[3] == 5)
        {
          rowConn[2] = connElEl[ii][jj];
        }
        // xz(+)
        if ( surfIndx[0] == 2 &&
             surfIndx[1] == 3 &&
             surfIndx[2] == 6 &&
             surfIndx[3] == 7)
        {
          rowConn[3] = connElEl[ii][jj];
        }
        // yz(-)
        if ( surfIndx[0] == 0 &&
             surfIndx[1] == 3 &&
             surfIndx[2] == 4 &&
             surfIndx[3] == 7)
        {
          rowConn[4] = connElEl[ii][jj];
        }
        // yz(+)
        //
        if ( surfIndx[0] == 1 &&
             surfIndx[1] == 2 &&
             surfIndx[2] == 5 &&
             surfIndx[3] == 6)
        {
          rowConn[5] = connElEl[ii][jj];
        }
      }

    }//end for(jj)
    connSurf.push_back(rowConn);

  }//end for(ii)
}//end createConnSurf

//////////////////////////////////////////////////////
//		attachBirthSurf			    //
//////////////////////////////////////////////////////
void attachBirthSurf(Mesh *meshObj,
                     map<int, int> element_global_to_local,
                     vector<Element*> elementList,
                     vector< vector<int> > connSurf,
                     vector <Surface> &surfaceList_)
{

  // Generate surface lists and their death times
  // ** NOTE ** birthElement_ will own surface of interest
  for (int ii = 0; ii < meshObj->birthID_.size(); ii++)
  {
    int gbID = meshObj->birthID_[ii];
    int lbID = element_global_to_local[gbID];
    Element *element = elementList[lbID];
    double birthTimeCurrent = element->birthTime_;
    for (int jj = 0; jj < 6; jj++)
    {
      int eIDCheck = connSurf[lbID][jj];
      if (eIDCheck != -1)
      { 
        Element *eleNeigh = elementList[eIDCheck];
        bool isbirth =  eleNeigh->birth_;
        if (isbirth)
        {
          double birthTimeNeighbor = eleNeigh->birthTime_;
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

}//attachBirthSurf

//////////////////////////////////////////////////////
//		attachStaticSurf	    	    //
//////////////////////////////////////////////////////
void attachStaticSurf(vector<int> numElEl, vector< vector<int> > connSurf,
                      vector<Element*> elementList, vector<Surface> &staticSurfList)
{
  for (int ii = 0; ii < numElEl.size(); ii++)
  {
    int numNeigh = numElEl[ii];
    Element *element = elementList[ii];
    if (numNeigh != 26 && !element->birth_)
    {
      for (int jj = 0; jj < 6; jj++)
      {
        int neighElement = connSurf[ii][jj];
        if (neighElement == -1)
        {
          Surface surfI;
          surfI.birthElement_ = ii;
          surfI.deathElement_ = -1;
          surfI.birthTime_ = 0.0;
          surfI.deathTime_ = 1.0e10;
          surfI.plane_ = jj;
          surfI.isDynamic_ = false;
          staticSurfList.push_back(surfI);
        }
      }//end for(jj)
    }
  }//end for(ii)
}//end attachStaticSurf

//////////////////////////////////////////////////////
//		attachSurfNset		    	    //
//////////////////////////////////////////////////////
void attachSurfNset(Mesh *meshObj, vector<Element*> elementList,
                    vector<int> node_local_to_global, vector<Surface> &surfaceList_)
{
  // assign node set ids to surfaces
  for (map<int, vector<int> >::iterator it = meshObj->nodeSets_.begin();
       it != meshObj->nodeSets_.end(); it++)
  {
    int nSetID            = it->first;
    vector <int> nSetList = it->second;

    sort(nSetList.begin(), nSetList.end());
    for (int ii = 0; ii < surfaceList_.size(); ii++)
    {
      int surfCt = 0;
      Surface surfI = surfaceList_[ii];
      int birthElement = surfI.birthElement_;
      Element *element = elementList[birthElement];
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
        int nodeCheck = node_local_to_global[ localNodes[ surfaceNodes[jj] ] ];
        if ( binary_search(nSetList.begin(), nSetList.end(), nodeCheck) )
        { 
          surfCt++;
        }
      }//end for(jj)

      if (surfCt == 4)
      {
        surfaceList_[ii].setID_.push_back(nSetID);
      }
    }//end for(ii)
  }//end for(it)
}//end attachSurfNset

//////////////////////////////////////////////////////
//		attachSurfBC		    	    //
//////////////////////////////////////////////////////
void attachSurfBC(Mesh* meshObj, vector<Element*> elementList,
                  vector<Surface> &surfaceList_)
{
  for (int ii = 0; ii < surfaceList_.size(); ii++)
  {
    Surface surfI = surfaceList_[ii]; 
    vector<int> surfSetID = surfI.setID_;
    for (int jj = 0; jj < surfSetID.size(); jj ++)
    {
      vector<int> bcIDVec = meshObj->loadSets_[surfSetID[jj]];
      for (int kk = 0; kk < bcIDVec.size(); kk++)
      {
        int bcID = bcIDVec[kk];
        if (bcID == 1)
        {
          surfaceList_[ii].isFixed_ = true;
          surfaceList_[ii].isFlux_ = false;
        } 
        else
        {
          surfaceList_[ii].isFixed_ = false;
          surfaceList_[ii].isFlux_ = true;
        }
      }//end for(kk)
    }//end for(jj)
  }//end for(ii)
}//end attachSurfBC

//////////////////////////////////////////////////////
//		sortIndxVec		    	    //
//////////////////////////////////////////////////////
void sortIndxVec(vector<int> unsortVec, vector<int> &indx)
{
  sort(indx.begin(), indx.end(), 
       [&unsortVec](int i1, int i2) {return unsortVec[i1]< unsortVec[i2];});

}//end sortIndxVec

