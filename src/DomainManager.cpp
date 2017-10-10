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
#include "time.h"

// user-defined headers
#include <Mesh.h>
#include <Element.h>
#include "dynaInput.h"
#include <Surface.h>
#include <DomainManager.h>
#include <FluxManager.h>
#include <ConvManager.h>
#include <RadManager.h>
#include <Probe.h>

//////////////////////////////////////////////////////
//		Constructor			    //
//////////////////////////////////////////////////////
DomainManager::DomainManager(Mesh *meshObj)
                            : meshObj_(meshObj),
                              isInit_(true)
{
}

//////////////////////////////////////////////////////
//		createNodeMap			    //
//////////////////////////////////////////////////////
void 
DomainManager::createNodeMap()
{

  // Create local-global node mapping
  int nodeCt = 0;
  for ( map<int, vector<double> >::iterator it = meshObj_->NODES_.begin();
        it != meshObj_->NODES_.end(); it++)
  {
    int global_nid = it->first;
    vector <double> gCoords = it->second;
    node_global_to_local_[global_nid] = nodeCt;
    for (int ii = 0; ii < gCoords.size(); ii++)
    {
      coordList_.push_back(gCoords[ii]);
    }
    nodeCt++;
  }

}//end createNodeMap

//////////////////////////////////////////////////////
//		createReverseNodeMap		    //
//////////////////////////////////////////////////////
void 
DomainManager::createReverseNodeMap()
{
  for  (map<int,int>::iterator it = node_global_to_local_.begin();
        it != node_global_to_local_.end(); it++)
  {
    node_local_to_global_.push_back(it->first);
  }
}//end createReverseNodeMap

//////////////////////////////////////////////////////
//		createElementList		    //
//////////////////////////////////////////////////////
void 
DomainManager::createElementList()
{
  // Create element classes
  int eleCnt = 0;
  for ( map<int, vector<int> >::iterator it = meshObj_->ELEM_.begin();
        it != meshObj_->ELEM_.end(); it++)
  {
    HexahedralElement* ele = new HexahedralElement;
	int elementID = it->first;
	ele->globId= elementID;
	element_global_to_local_[elementID] = eleCnt;
	element_local_to_global_.push_back(elementID);
    vector<int> localNode = it->second;
    for (int ii = 0; ii<localNode.size(); ii++)
    {
      int global_nid = localNode[ii];
      ele->nID_[ii] = node_global_to_local_[global_nid];
    }//end for(ii)
    ele->birthTime_ = 0.0;
    ele->birth_=false;
    elementList_.push_back(ele);
	eleCnt++;
  }//end for(it)

  assignElementBirth();
  std::sort(elementList_.begin(), elementList_.end(), [](Element* a, Element* b) {return a->birthTime_ < b->birthTime_; });

  element_global_to_local_.clear();
  element_local_to_global_.clear();
  // assign globaltolocal and localtoglobal
  eleCnt = 0;
  for (std::vector<Element*>::iterator it = elementList_.begin(); it != elementList_.end(); ++it) {
	  Element * element = *it;
	  element_global_to_local_[element->globId] = eleCnt;
	  element_local_to_global_.push_back(element->globId);
	  eleCnt++;
  }


}//end createElementList

//////////////////////////////////////////////////////
//		assignElePID		    	    //
//////////////////////////////////////////////////////
void 
DomainManager::assignElePID()
{
  for (map<int, vector<int> >::iterator it = meshObj_->PIDS_.begin();
       it != meshObj_->PIDS_.end(); it++)
  {
    int PID = it->first;
    vector <double> eleMat = meshObj_->PID_to_MAT_[PID];
    vector<int> pidEleList = it->second;
    for (int ii = 0; ii < pidEleList.size(); ii++)
    {
      int elePID = pidEleList[ii];
      int localEle = element_global_to_local_[elePID];
      Element *element = elementList_[localEle];
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
void 
DomainManager::checkElementJac()
{
  minElementVol_ = 10000000000000.0;
  double xCoords[8];
  double yCoords[8];
  double zCoords[8];
  xmin_ = minElementVol_;
  ymin_ = minElementVol_;
  zmin_ = minElementVol_;
  xmax_ = -10000000000000.0;
  ymax_ = -10000000000000.0;
  zmax_ = -10000000000000.0;
  HexahedralElement* HexTemp = new HexahedralElement;

  for (int ii = 0; ii < elementList_.size(); ii++)
  {
    double nodalCoords[24];
    double volCheck = 0.0;
    //unpack local nodal coordinates.
    Element * element = elementList_[ii];
    int *localNodes = element->nID_;
    for (int jj = 0; jj < 8; jj++)
    {
      int nid = localNodes[jj];
      xCoords[jj] = coordList_[nid*3 + 0];
      yCoords[jj] = coordList_[nid*3 + 1];
      zCoords[jj] = coordList_[nid*3 + 2];
      for (int kk = 0; kk < 3; kk++)
      {
        nodalCoords[kk+jj*3] = coordList_[nid*3 + kk];
      }//end for(kk)
    }//end for(jj)

    // get max/mins of domain
    xmin_ = std::min( xmin_, *min_element(xCoords, xCoords + 8) );
    xmax_ = std::max( xmax_, *max_element(xCoords, xCoords + 8) );
    ymin_ = std::min( ymin_, *min_element(yCoords, yCoords + 8) );
    ymax_ = std::max( ymax_, *max_element(yCoords, yCoords + 8) );
    zmin_ = std::min( zmin_, *min_element(zCoords, zCoords + 8) );
    zmax_ = std::max( zmax_, *max_element(zCoords, zCoords + 8) );

    // Calculate element jacobian
    double deriv[24], detJac, iJac[9];
    double parCoords[3] = {0.0, 0.0, 0.0};
    HexTemp->derivative_of_shape_function_about_parametic_coords(parCoords, deriv);
    HexTemp->Jacobian(deriv, nodalCoords, iJac, detJac);
    if (detJac < 0.0)
    {
      cout << "WARNING!!!! Negative Jacobian! Element Number: " << ii << endl;
      exit(EXIT_FAILURE);
    }
    volCheck = detJac * 8.0; //adjust for gaussian weight^3
    minElementVol_ = std::min(volCheck, minElementVol_);

  }//end for(it)
}//end checkElementJac

//////////////////////////////////////////////////////
//		assignElementBirth		    //
//////////////////////////////////////////////////////
void 
DomainManager::assignElementBirth()
{

  // Assign element birth times
  for (int ii = 0; ii < meshObj_->birthID_.size(); ii++)
  {
    int gbID = meshObj_->birthID_[ii];
    int lbID = element_global_to_local_[meshObj_->birthID_[ii]];
    double bTime = meshObj_->birthTime_[ii];
    elementList_[lbID]->birthTime_ = bTime;
    elementList_[lbID]->birth_ = true;
    //birthElements_.push_back(lbID);
  }//end for(ii)

}//end assignElementBirth

void DomainManager::assignElBirthList()
{

	// Assign element birth times
	for (int ii = 0; ii < meshObj_->birthID_.size(); ii++)
	{
		int gbID = meshObj_->birthID_[ii];
		int lbID = element_global_to_local_[meshObj_->birthID_[ii]];
		birthElements_.push_back(lbID);
	}//end for(ii)

}//end assignElementBirth

//////////////////////////////////////////////////////
//		createElElConn			    //
//////////////////////////////////////////////////////
void 
DomainManager::createElElConn()
{
  //***********************************//
  //	# of neighbors:
  //  	  edges = 11 elements
  //	  surfaces = 17 elements
  //	  corners = 7 elements
  //	  interior = 26 elements
  //***********************************//
  int idxCt = 0;
  for (int ii = 0; ii < elementList_.size(); ii++)
  { 
    Element *element = elementList_[ii];
    int *localNodes = element->nID_;
    for (int jj = 0; jj < 8; jj++)
    {
      connVec_.push_back(localNodes[jj]);
      conn_to_el_Vec_.push_back(ii);
      connVecIndx_.push_back(idxCt);
      idxCt++;
    }//end for(jj)
    vector<int> rowEl;
    rowEl.resize(26);
    for (int mm = 0; mm < 26; mm++) rowEl[mm] = -1;
    connElEl_.push_back(rowEl);
    numElEl_.push_back(0);
  }//end for(ii)

  // sort flat connectivity indices 
  sortIndxVec(connVec_, connVecIndx_);

  // sort flat connectivity
  sort(connVec_.begin(), connVec_.end());

  for (int ii = 0; ii < elementList_.size(); ii++)
  {
    Element *element = elementList_[ii];
    int *localNodes = element->nID_;
    for (int jj = 0; jj < 8; jj++)
    {
      int searchNode = localNodes[jj];
      vector<int>::iterator low  = lower_bound(connVec_.begin(), connVec_.end(), searchNode);
      vector<int>::iterator high = upper_bound(connVec_.begin(), connVec_.end(), searchNode);
      int lowIndx   = low - connVec_.begin();
      int highIndx  = high - connVec_.begin();
      for (int kk = lowIndx; kk < highIndx; kk++)
      {
        int nodeEleKK = connVecIndx_[kk];
        if (ii != conn_to_el_Vec_[nodeEleKK] )
        {
          for ( int ll = 0; ll < 26; ll++)
          {
            if(connElEl_[ii][ll] == conn_to_el_Vec_[nodeEleKK]) break;

            if(connElEl_[ii][ll]==-1)
            {
              connElEl_[ii][numElEl_[ii]] = conn_to_el_Vec_[nodeEleKK];
              numElEl_[ii] = numElEl_[ii] + 1;
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
void 
DomainManager::createConnSurf()
{
  // set up surface connections between elements
  // Element Orientation
 /*     local node ordering is 
	CONNECTIVITY ORDERING
  connSurf_ storage order:
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
  for (int ii = 0; ii < elementList_.size(); ii++)
  {
    vector<int> rowConn(6);

    for (int mm = 0; mm < 6; mm++) rowConn[mm] = -1;

    Element *element = elementList_[ii];
    int *localNodes = element->nID_;

    for (int jj = 0; jj < numElEl_[ii]; jj++)
    { 
      surfCt = 0;
      Element *elementCompare = elementList_[connElEl_[ii][jj]];
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
          rowConn[0] = connElEl_[ii][jj];
        }
        // xy(-)
        if ( surfIndx[0] == 0 &&
             surfIndx[1] == 1 &&
             surfIndx[2] == 2 &&
             surfIndx[3] == 3)
        {
          rowConn[1] = connElEl_[ii][jj];
        }
        // xz(-)
        if ( surfIndx[0] == 0 &&
             surfIndx[1] == 1 &&
             surfIndx[2] == 4 &&
             surfIndx[3] == 5)
        {
          rowConn[2] = connElEl_[ii][jj];
        }
        // xz(+)
        if ( surfIndx[0] == 2 &&
             surfIndx[1] == 3 &&
             surfIndx[2] == 6 &&
             surfIndx[3] == 7)
        {
          rowConn[3] = connElEl_[ii][jj];
        }
        // yz(-)
        if ( surfIndx[0] == 0 &&
             surfIndx[1] == 3 &&
             surfIndx[2] == 4 &&
             surfIndx[3] == 7)
        {
          rowConn[4] = connElEl_[ii][jj];
        }
        // yz(+)
        //
        if ( surfIndx[0] == 1 &&
             surfIndx[1] == 2 &&
             surfIndx[2] == 5 &&
             surfIndx[3] == 6)
        {
          rowConn[5] = connElEl_[ii][jj];
        }
      }

    }//end for(jj)
    connSurf_.push_back(rowConn);

  }//end for(ii)
}//end createConnSurf

//////////////////////////////////////////////////////
//		sortIndxVec		    	    //
//////////////////////////////////////////////////////
void 
DomainManager::sortIndxVec(vector<int> unsortVec, vector<int> &indx)
{
  sort(indx.begin(), indx.end(), 
       [&unsortVec](int i1, int i2) {return unsortVec[i1]< unsortVec[i2];});

}//end sortIndxVec

//////////////////////////////////////////////////////
//		sortIndxVec_double	    	    //
//////////////////////////////////////////////////////
void 
DomainManager::sortIndxVec_double(vector<double> unsortVec, vector<int> &indx)
{
  sort(indx.begin(), indx.end(), 
       [&unsortVec](int i1, int i2) {return unsortVec[i1]< unsortVec[i2];});

}//end sortIndxVec

//////////////////////////////////////////////////////
//		initializeActiveElements	    //
//////////////////////////////////////////////////////
void 
DomainManager::initializeActiveElements()
{
  vector <double> birthTmpNodeTimes_;
  double big = 1.0e10;
  birthTmpNodeTimes_.resize(nn_);
  fill(birthTmpNodeTimes_.begin(), birthTmpNodeTimes_.end(), big + 1.0);
  double small = 1.0e-10;

  // Find active elements
  for (int ie = 0; ie < nel_; ie++)
  {
    Element *element = elementList_[ie];
    int *localNodes = element->nID_;
    if (!element->birth_)
    {
      activeElements_.push_back(ie);
      for (int I = 0; I < 8; I++)
      {
        int IG = localNodes[I];
	birthTmpNodeTimes_[IG] = -1.0;
      }//end for(I)
    } 
  }//end for(ie)

  // Find birth elements
  for (int ie = 0; ie < birthElements_.size(); ie++)
  {
    int eID = birthElements_[ie];
    Element *element = elementList_[eID];
    double birthTime = element->birthTime_;
    int *localNodes = element->nID_;
    for (int I = 0; I < 8; I++)
    {
      int IG = localNodes[I];
      if (birthTmpNodeTimes_[IG] >= -small)
      {
	if (birthTmpNodeTimes_[IG] > birthTime || 
	    birthTmpNodeTimes_[IG] > big)
	{
	  birthTmpNodeTimes_[IG] = birthTime;
	}//end if
      }//end if
    }//end for(I)
  
  }//end for(ie)

  for (int I = 0; I < nn_; I++)
  {
    if (birthTmpNodeTimes_[I] > -small)
    {
      birthNodes_.push_back(I);
      birthNodeTimes_.push_back(birthTmpNodeTimes_[I]);
    }
    else
    {
      activeNodes_.push_back(I);
    }
  }//end for(I)

  birthTmpNodeTimes_.clear();
  
  nelactive_ = activeElements_.size();
  nnactive_ = activeNodes_.size();
  birthEleCt = 0;
  birthNodeCt = 0;

  int numBirthNodes = birthNodeTimes_.size();
  vector <int> birthTimeIndx, holdBirthNodes;
  birthTimeIndx.resize(numBirthNodes);
  holdBirthNodes.resize(numBirthNodes);

  // Sort nodal birth by time:
  for (int I = 0; I < numBirthNodes; I++)
  {
    birthTimeIndx[I] = I;
    holdBirthNodes[I] = birthNodes_[I];
  }
  sortIndxVec_double(birthNodeTimes_, birthTimeIndx);
  sort(birthNodeTimes_.begin(), birthNodeTimes_.end());

  // reposition indices based on sort
  for (int I = 0; I < numBirthNodes; I++)
  {
    birthNodes_[I] = holdBirthNodes[birthTimeIndx[I]];
  }

  birthTimeIndx.clear();
  holdBirthNodes.clear();
}//end initializeActiveElements

//////////////////////////////////////////////////////
//		updateActiveElements	    	    //
//////////////////////////////////////////////////////
void 
DomainManager::updateActiveElements( vector<double> &thetaN, double initTemp)
{
  // update active element list
  vector<int> ::iterator it;
  vector<int> ::iterator itNode;
  vector<double> :: iterator itNodeTime;
  nelactiveOld_ = activeElements_.size();
  int eraseIt_ele = 0;
  int eraseIt_node = 0;
  for (it = birthElements_.begin(); it != birthElements_.end(); )
  {
    int eID = *it;
    Element * element = elementList_[eID];
    if (element->birthTime_ <= currTime_)
    {
      activeElements_.push_back(eID);
      it = birthElements_.erase(it);
    }
    else
    {
      break;
    }
  }//end for(I)

  // update active surface list
  int delIndx;
  itNodeTime = birthNodeTimes_.begin();
  for (itNode = birthNodes_.begin(); itNode != birthNodes_.end(); )
  {
    int IG = *itNode;
    double birthTime = *itNodeTime;
    if ( (birthTime <= currTime_ ) )
    {
      activeNodes_.push_back(IG);
      thetaN[IG] = initTemp;
      itNodeTime = birthNodeTimes_.erase(itNodeTime);
      itNode = birthNodes_.erase(itNode); 
//      delIndx = I;
    } 
    else
    {
//      delIndx = I;
      break;
    }
  }//end for(I)

//  birthNodes_.erase( birthNodes_.begin(), birthNodes_.begin() + delIndx);
//  birthNodeTimes_.erase( birthNodeTimes_.begin(), birthNodeTimes_.begin() + delIndx);

  // update sizes
  nnactive_ = activeNodes_.size();
  nelactive_ = activeElements_.size();


}//end updateActiveElements

//////////////////////////////////////////////////////
//		getToolpath()			    //
//////////////////////////////////////////////////////
void
DomainManager::getToolpath()
{
  ifstream file;
  file.open(meshObj_->toolFileName_.c_str());
  string line;
  if (file.is_open())
  {
    while(getline(file, line))
    {
      istringstream lines(line);
      vector<double> coords((istream_iterator<double>(lines)), istream_iterator<double>());
      vector<double> txyz(coords.begin(), coords.begin() + 4);
      int state = (int)coords[4];
      laserOn_.push_back(state);
      tooltxyz_.push_back(txyz);
    }// end while
  }//end if
}//end getToolpath

//////////////////////////////////////////////////////
//		generateBins()			    //
//////////////////////////////////////////////////////
void
DomainManager::generateBins()
{
  const std::string SecondEleWidth_ = "  ";
  const std::string ThirdEleWidth_  = "    ";
  const std::string FourthEleWidth_ = "      ";
  const std::string FifthEleWidth_  = "        ";
  const std::string SixthEleWidth_  = "          ";

  double charLength = pow(minElementVol_, 1.0/3.0);
  dxBin_ = 10.0 * charLength;
  dyBin_ = 10.0 * charLength;
  dzBin_ = 10.0 * charLength;
  binNx_ = ceil( (xmax_ - xmin_)/dxBin_);
  binNy_ = ceil( (ymax_ - ymin_)/dyBin_);
  binNz_ = ceil( (zmax_ - zmin_)/dzBin_);
  cout << FourthEleWidth_ << "Number of bins created: "
       << binNx_ * binNy_ * binNz_ << endl;

  binList_.resize(binNx_ * binNy_ * binNz_);
  for (int ii = 0; ii < binNx_*binNy_*binNz_; ii++) 
  {
    binList_[ii] = new Bin;
  }
  
  // Assign elements to bins
  double xCoords[8];
  double yCoords[8];
  double zCoords[8];
  for (int ie = 0; ie < elementList_.size(); ie++)
  {
    //unpack coordinates
    Element * element = elementList_[ie];
    int *localNodes = element->nID_;
    for (int jj = 0; jj < 8; jj++)
    {
      int nid = localNodes[jj];
      xCoords[jj] = coordList_[nid*3 + 0];
      yCoords[jj] = coordList_[nid*3 + 1];
      zCoords[jj] = coordList_[nid*3 + 2];
    }//end for(jj)

    // get max/mins of domain
    double eleMinX = *min_element(xCoords, xCoords + 8);
    double eleMaxX = *max_element(xCoords, xCoords + 8);
    double eleMinY = *min_element(yCoords, yCoords + 8);
    double eleMaxY = *max_element(yCoords, yCoords + 8);
    double eleMinZ = *min_element(zCoords, zCoords + 8);
    double eleMaxZ = *max_element(zCoords, zCoords + 8);

    // find bounds for bins
    int imin = floor( (eleMinX - xmin_)/dxBin_ );
    int imax = ceil( (eleMaxX - xmin_)/dxBin_ );
    int jmin = floor( ( eleMinY - ymin_)/dyBin_ );
    int jmax = ceil( (eleMaxY - ymin_)/dyBin_ );
    int kmin = floor( ( eleMinZ - zmin_)/dzBin_ );
    int kmax = ceil( (eleMaxZ - zmin_)/dzBin_ );
    for (int kk = kmin; kk < kmax; kk++)
    {
      for (int jj = jmin; jj < jmax; jj++)
      {
        for (int ii = imin; ii < imax; ii++)
        {
          int ind = ii + jj * binNx_ + kk * binNy_ * binNx_;
          binList_[ind]->hasElements_ = true;
          binList_[ind]->elementBucket_.push_back(ie);
        }//end for(ii)
      }//end for(jj)
    }//end for (kk)
  }//end for(ie)

}//end generateBins


//////////////////////////////////////////////////////
//		assignProbePoints()	            //
//////////////////////////////////////////////////////
void
DomainManager::assignProbePoints()
{
  const std::string SecondEleWidth_ = "  ";
  const std::string ThirdEleWidth_  = "    ";
  const std::string FourthEleWidth_ = "      ";
  for (int ii = 0; ii < meshObj_->probeNames_.size(); ii++)
  {
    string probeName = meshObj_->probeNames_[ii];
    double *probeCoords = meshObj_->probePos_[ii];
    Probe * probe_I = new Probe(probeCoords);
    probeList_.push_back(probe_I);
  }


  // Fine elements that own probes
  double xCoords[8];
  double yCoords[8];
  double zCoords[8];
  double tol = 1.0e-8;
  vector<vector<double>> elem_nodal_coor;
  for (int i = 0; i < 8; i++)
  {
    vector<double> temp;
    temp.resize(3);
    elem_nodal_coor.push_back(temp);
  }  

  for (int iprobe = 0; iprobe < probeList_.size(); iprobe++)
  {
    double xp = probeList_[iprobe]->physCoords_[0];
    double yp = probeList_[iprobe]->physCoords_[1];
    double zp = probeList_[iprobe]->physCoords_[2];
   
    int imin = floor( (xp - xmin_)/dxBin_ );
    int imax = ceil( (xp - xmin_)/dxBin_ );
    int jmin = floor( (yp - ymin_)/dyBin_ );
    int jmax = ceil( (yp - ymin_)/dyBin_ );
    int kmin = floor( (zp - zmin_)/dzBin_ );
    int kmax = ceil( (zp - zmin_)/dzBin_ );

    // Adjust i indices
    if (imin == imax && imin > 0 && imax < binNx_)
    {
      imax = imax + 1;
      imin = imax - 1;
    }
    if (imax == 0)
    {
      imax = 1;
    }
    if (imax == binNx_ && imin > 0)
    {
      imin = imin - 1;
    }
    if (imin < 0)
    {
      imin = 0;
    }

    // Adjust j indices
    if (jmin == jmax && jmin > 0 && jmax < binNy_)
    {
      jmax = jmax + 1;
      jmin = jmax - 1;
    }
    if (jmax == 0)
    {
      jmax = 1;
    }
    if (jmax == binNy_ && jmin > 0)
    {
      jmin = jmin - 1;
    }
    if (jmin < 0)
    {
      jmin = 0;
    }

    // Adjust k indices
    if (kmin == kmax && kmin > 0 && kmax < binNz_)
    {
      kmax = kmax + 1;
      kmin = kmax - 1;
    }
    if (kmax == 0)
    {
      kmax = 1;
    }
    if (kmax == binNz_ && kmin > 0)
    {
      kmin = kmin - 1;
    }
    if (kmin < 0)
    {
      kmin = 0;
    }
    for (int kk = kmin; kk < kmax; kk++)
    {
      if(probeList_[iprobe]->hasElement_) break;
      for (int jj = jmin; jj < jmax; jj++)
      {
	if(probeList_[iprobe]->hasElement_) break;
        for (int ii = imin; ii < imax; ii++)
        {
	  if(probeList_[iprobe]->hasElement_) break;
          int ind = ii + jj * binNx_ + kk * binNy_ * binNx_;
          if (binList_[ind]->hasElements_) 
          {
            for (int ie = 0; ie < binList_[ind]->elementBucket_.size(); ie++)
            {
	      //unpack coordinates
	      int eID = binList_[ind]->elementBucket_[ie];
	      Element * element = elementList_[eID];
	      int *localNodes = element->nID_;
	      for (int mm = 0; mm < 8; mm++)
	      {
		int nid = localNodes[mm];
		xCoords[mm] = coordList_[nid*3 + 0];
		yCoords[mm] = coordList_[nid*3 + 1];
		zCoords[mm] = coordList_[nid*3 + 2];
                elem_nodal_coor[mm][0] = xCoords[mm];
                elem_nodal_coor[mm][1] = yCoords[mm];
                elem_nodal_coor[mm][2] = zCoords[mm];
	      }//end for(jj)

	      // get max/mins of domain
	      double eleMinX = *min_element(xCoords, xCoords + 8);
	      double eleMaxX = *max_element(xCoords, xCoords + 8);
	      double eleMinY = *min_element(yCoords, yCoords + 8);
	      double eleMaxY = *max_element(yCoords, yCoords + 8);
	      double eleMinZ = *min_element(zCoords, zCoords + 8);
	      double eleMaxZ = *max_element(zCoords, zCoords + 8);

              if ( xp >= eleMinX && xp <= eleMaxX &&
                   yp >= eleMinY && yp <= eleMaxY &&
                   zp >= eleMinZ && zp <= eleMaxZ)
              {
		double *parCoords = new double[3];
                double dist = isInElement(elem_nodal_coor, 
                               probeList_[iprobe]->physCoords_, parCoords);
                if (dist <= 1.0 + tol)
                {
                  probeList_[iprobe]->parCoords_ = parCoords;
                  probeList_[iprobe]->elementID_ = eID;
                  probeList_[iprobe]->hasElement_ = true;
		  cout << FourthEleWidth_ 
                       << "Element found for probe ["<< iprobe + 1 <<"] : "
                       << eID << endl;
                  break;
                }
              }
		    
            }//end for(ie)
          }//end if
        }//end for(ii)
      }//end for(jj)
    }//end for(kk)
  }//end for(iprobe)

}//end assignProbePoints

//////////////////////////////////////////////////////
//		initializeDomain	    	    //
//////////////////////////////////////////////////////
void 
DomainManager::initializeDomain()
{
  // Vtk indentation rule
  const std::string SecondEleWidth_ = "  ";
  const std::string ThirdEleWidth_  = "    ";
  const std::string FourthEleWidth_ = "      ";
  const std::string FifthEleWidth_  = "        ";
  const std::string SixthEleWidth_  = "          ";
  double starttime, endtime, preprocessTimer;

  preprocessTimer = 0.0;
  cout <<  "===============================================================\n";  
  cout <<  "TIMING INDIVIDUAL PREPROCESSES\n\n";

  starttime = clock();
  getToolpath();
  endtime = clock();

  starttime = clock();
  createNodeMap();
  createReverseNodeMap();
  endtime = clock();
  cout << SecondEleWidth_ << "Time generating node mapping: "
       << (double) (endtime - starttime) / CLOCKS_PER_SEC << endl;
  preprocessTimer += (double) (endtime - starttime) / CLOCKS_PER_SEC;

  starttime = clock();
  createElementList();
  endtime = clock();
  cout << SecondEleWidth_ << "Time generating element list: "
       << (double) (endtime - starttime) / CLOCKS_PER_SEC << endl;
  preprocessTimer += (double) (endtime - starttime) / CLOCKS_PER_SEC;
  
  starttime = clock();
  assignElePID();
  endtime = clock();
  cout << SecondEleWidth_ << "Time for assigning material IDs to elements: "
       << (double) (endtime - starttime) / CLOCKS_PER_SEC << endl;
  preprocessTimer += (double) (endtime - starttime) / CLOCKS_PER_SEC;

  starttime = clock();
  checkElementJac();
  endtime = clock();
  cout << SecondEleWidth_ << "Time for calculating jacobians: "
       << (double) (endtime - starttime) / CLOCKS_PER_SEC << endl;
  preprocessTimer += (double) (endtime - starttime) / CLOCKS_PER_SEC;

  
  starttime = clock();
  assignElBirthList();
  endtime = clock();
  cout << SecondEleWidth_ << "Time for assigning birth elements: "
       << (double) (endtime - starttime) / CLOCKS_PER_SEC << endl;
  preprocessTimer += (double) (endtime - starttime) / CLOCKS_PER_SEC;
  

  starttime = clock();
  createElElConn();
  endtime = clock();
  cout << SecondEleWidth_ << "Time for generating element neighbors: "
       << (double) (endtime - starttime) / CLOCKS_PER_SEC << endl;
  preprocessTimer += (double) (endtime - starttime) / CLOCKS_PER_SEC;

  starttime = clock();
  createConnSurf();
  endtime = clock();
  cout << SecondEleWidth_ << "Time for generating element surface connections: "
       << (double) (endtime - starttime) / CLOCKS_PER_SEC << endl;
  preprocessTimer += (double) (endtime - starttime) / CLOCKS_PER_SEC;

  starttime = clock();
  generateBins();
  endtime = clock();
  cout << SecondEleWidth_ << "Time for generating buckets for bucket search: "
       << (double) (endtime - starttime) / CLOCKS_PER_SEC << endl;
  preprocessTimer += (double) (endtime - starttime) / CLOCKS_PER_SEC;

  starttime = clock();
  assignProbePoints();
  endtime = clock();
  cout << SecondEleWidth_ << "Time for getting parent coordinates for probe: "
       << (double) (endtime - starttime) / CLOCKS_PER_SEC << endl;
  preprocessTimer += (double) (endtime - starttime) / CLOCKS_PER_SEC;

  cout <<  "\nTOTAL TIME FOR PREPROCESSING:\n";
  cout <<  SecondEleWidth_ << preprocessTimer << endl;
  cout <<  "===============================================================\n";  

  nn_ = node_global_to_local_.size();
  nel_ = element_global_to_local_.size();

  initializeActiveElements();

}//end initializeDomain

//////////////////////////////////////////////////////
//		isInElement()			    //
//////////////////////////////////////////////////////
double
DomainManager::isInElement(vector< vector <double> > &elem_nodal_coor,
                    double* point_coor, double* par_coor)
{
  const int maxNonlinearIter = 20;
  const double isInElemConverged = 1.0e-16;
  // Translate element so that (x,y,z) coordinates of the first node are (0,0,0)

  double x[] = {0.,
	      0.125*(elem_nodal_coor[1][0] - elem_nodal_coor[0][0]),
	      0.125*(elem_nodal_coor[2][0] - elem_nodal_coor[0][0]),
	      0.125*(elem_nodal_coor[3][0] - elem_nodal_coor[0][0]),
	      0.125*(elem_nodal_coor[4][0] - elem_nodal_coor[0][0]),
	      0.125*(elem_nodal_coor[5][0] - elem_nodal_coor[0][0]),
	      0.125*(elem_nodal_coor[6][0] - elem_nodal_coor[0][0]),
	      0.125*(elem_nodal_coor[7][0] - elem_nodal_coor[0][0]) };
  double y[] = {0.,
	      0.125*(elem_nodal_coor[1][1] - elem_nodal_coor[0][1]),
	      0.125*(elem_nodal_coor[2][1] - elem_nodal_coor[0][1]),
	      0.125*(elem_nodal_coor[3][1] - elem_nodal_coor[0][1]),
	      0.125*(elem_nodal_coor[4][1] - elem_nodal_coor[0][1]),
	      0.125*(elem_nodal_coor[5][1] - elem_nodal_coor[0][1]),
	      0.125*(elem_nodal_coor[6][1] - elem_nodal_coor[0][1]),
	      0.125*(elem_nodal_coor[7][1] - elem_nodal_coor[0][1]) };
  double z[] = {0.,
	      0.125*(elem_nodal_coor[1][2] - elem_nodal_coor[0][2]),
	      0.125*(elem_nodal_coor[2][2] - elem_nodal_coor[0][2]),
	      0.125*(elem_nodal_coor[3][2] - elem_nodal_coor[0][2]),
	      0.125*(elem_nodal_coor[4][2] - elem_nodal_coor[0][2]),
	      0.125*(elem_nodal_coor[5][2] - elem_nodal_coor[0][2]),
	      0.125*(elem_nodal_coor[6][2] - elem_nodal_coor[0][2]),
	      0.125*(elem_nodal_coor[7][2] - elem_nodal_coor[0][2]) };

  // (xp,yp,zp) is the point at which we're searching for (xi,eta,zeta)
  // (must translate this also)

  double xp = point_coor[0] - elem_nodal_coor[0][0];
  double yp = point_coor[1] - elem_nodal_coor[0][1];
  double zp = point_coor[2] - elem_nodal_coor[0][2];

  // Newton-Raphson iteration for (xi,eta,zeta)
  double j[9];
  double f[3];
  double shapefct[8];
  double xinew = 0.5;     // initial guess
  double etanew = 0.5;
  double zetanew = 0.5;
  double xicur = 0.5;
  double etacur = 0.5;
  double zetacur = 0.5;
  double xidiff[] = { 1.0, 1.0, 1.0 };
  int i = 0;

  // structured of jacobian: 
  // 0 - x/xi
  // 1 - x/eta
  // 2 - x/zeta
  // 3 - y/xi
  // 4 - y/eta
  // 5 - y/zeta
  // 6 - z/xi
  // 7 - z/eta
  // 8 - z/zeta
  do
  {
    j[0]=
      -(1.0-etacur)*(1.0-zetacur)*x[1]
      -(1.0+etacur)*(1.0-zetacur)*x[2]
      +(1.0+etacur)*(1.0-zetacur)*x[3]
      +(1.0-etacur)*(1.0+zetacur)*x[4]
      -(1.0-etacur)*(1.0+zetacur)*x[5]
      -(1.0+etacur)*(1.0+zetacur)*x[6]
      +(1.0+etacur)*(1.0+zetacur)*x[7];

    j[1]=
       (1.0+xicur)*(1.0-zetacur)*x[1]
      -(1.0+xicur)*(1.0-zetacur)*x[2]
      -(1.0-xicur)*(1.0-zetacur)*x[3]
      +(1.0-xicur)*(1.0+zetacur)*x[4]
      +(1.0+xicur)*(1.0+zetacur)*x[5]
      -(1.0+xicur)*(1.0+zetacur)*x[6]
      -(1.0-xicur)*(1.0+zetacur)*x[7];

    j[2]=
       (1.0-etacur)*(1.0+xicur)*x[1]
      +(1.0+etacur)*(1.0+xicur)*x[2]
      +(1.0+etacur)*(1.0-xicur)*x[3]
      -(1.0-etacur)*(1.0-xicur)*x[4]
      -(1.0-etacur)*(1.0+xicur)*x[5]
      -(1.0+etacur)*(1.0+xicur)*x[6]
      -(1.0+etacur)*(1.0-xicur)*x[7];

    j[3]=
      -(1.0-etacur)*(1.0-zetacur)*y[1]
      -(1.0+etacur)*(1.0-zetacur)*y[2]
      +(1.0+etacur)*(1.0-zetacur)*y[3]
      +(1.0-etacur)*(1.0+zetacur)*y[4]
      -(1.0-etacur)*(1.0+zetacur)*y[5]
      -(1.0+etacur)*(1.0+zetacur)*y[6]
      +(1.0+etacur)*(1.0+zetacur)*y[7];

    j[4]=
       (1.0+xicur)*(1.0-zetacur)*y[1]
      -(1.0+xicur)*(1.0-zetacur)*y[2]
      -(1.0-xicur)*(1.0-zetacur)*y[3]
      +(1.0-xicur)*(1.0+zetacur)*y[4]
      +(1.0+xicur)*(1.0+zetacur)*y[5]
      -(1.0+xicur)*(1.0+zetacur)*y[6]
      -(1.0-xicur)*(1.0+zetacur)*y[7];

    j[5]=
       (1.0-etacur)*(1.0+xicur)*y[1]
      +(1.0+etacur)*(1.0+xicur)*y[2]
      +(1.0+etacur)*(1.0-xicur)*y[3]
      -(1.0-etacur)*(1.0-xicur)*y[4]
      -(1.0-etacur)*(1.0+xicur)*y[5]
      -(1.0+etacur)*(1.0+xicur)*y[6]
      -(1.0+etacur)*(1.0-xicur)*y[7];

    j[6]=
      -(1.0-etacur)*(1.0-zetacur)*z[1]
      -(1.0+etacur)*(1.0-zetacur)*z[2]
      +(1.0+etacur)*(1.0-zetacur)*z[3]
      +(1.0-etacur)*(1.0+zetacur)*z[4]
      -(1.0-etacur)*(1.0+zetacur)*z[5]
      -(1.0+etacur)*(1.0+zetacur)*z[6]
      +(1.0+etacur)*(1.0+zetacur)*z[7];

    j[7]=
       (1.0+xicur)*(1.0-zetacur)*z[1]
      -(1.0+xicur)*(1.0-zetacur)*z[2]
      -(1.0-xicur)*(1.0-zetacur)*z[3]
      +(1.0-xicur)*(1.0+zetacur)*z[4]
      +(1.0+xicur)*(1.0+zetacur)*z[5]
      -(1.0+xicur)*(1.0+zetacur)*z[6]
      -(1.0-xicur)*(1.0+zetacur)*z[7];

    j[8]=
       (1.0-etacur)*(1.0+xicur)*z[1]
      +(1.0+etacur)*(1.0+xicur)*z[2]
      +(1.0+etacur)*(1.0-xicur)*z[3]
      -(1.0-etacur)*(1.0-xicur)*z[4]
      -(1.0-etacur)*(1.0+xicur)*z[5]
      -(1.0+etacur)*(1.0+xicur)*z[6]
      -(1.0+etacur)*(1.0-xicur)*z[7];

    double jdet=-(j[2]*j[4]*j[6])+j[1]*j[5]*j[6]+j[2]*j[3]*j[7]-
      j[0]*j[5]*j[7]-j[1]*j[3]*j[8]+j[0]*j[4]*j[8];

    if (!jdet) {
      i = maxNonlinearIter;
      break;
    }
    shapefct[0]=(1.0-etacur)*(1.0-xicur)*(1.0-zetacur);

    shapefct[1]=(1.0-etacur)*(1.0+xicur)*(1.0-zetacur);

    shapefct[2]=(1.0+etacur)*(1.0+xicur)*(1.0-zetacur);

    shapefct[3]=(1.0+etacur)*(1.0-xicur)*(1.0-zetacur);

    shapefct[4]=(1.0-etacur)*(1.0-xicur)*(1.0+zetacur);

    shapefct[5]=(1.0-etacur)*(1.0+xicur)*(1.0+zetacur);

    shapefct[6]=(1.0+etacur)*(1.0+xicur)*(1.0+zetacur);

    shapefct[7]=(1.0+etacur)*(1.0-xicur)*(1.0+zetacur);

    f[0]=xp-shapefct[1]*x[1]-shapefct[2]*x[2]-shapefct[3]*x[3]-shapefct[4]*x[4]-\
      shapefct[5]*x[5]-shapefct[6]*x[6]-shapefct[7]*x[7];

    f[1]=yp-shapefct[1]*y[1]-shapefct[2]*y[2]-shapefct[3]*y[3]-shapefct[4]*y[4]-\
      shapefct[5]*y[5]-shapefct[6]*y[6]-shapefct[7]*y[7];

    f[2]=zp-shapefct[1]*z[1]-shapefct[2]*z[2]-shapefct[3]*z[3]-shapefct[4]*z[4]-\
      shapefct[5]*z[5]-shapefct[6]*z[6]-shapefct[7]*z[7];

    xinew = (jdet*xicur+f[2]*(j[2]*j[4]-j[1]*j[5])-f[1]*j[2]*j[7]+f[0]*j[5]*j[7]+
	     f[1]*j[1]*j[8]-f[0]*j[4]*j[8])/jdet;

    etanew = (etacur*jdet+f[2]*(-(j[2]*j[3])+j[0]*j[5])+f[1]*j[2]*j[6]-f[0]*j[5]*j[6]-
	      f[1]*j[0]*j[8]+f[0]*j[3]*j[8])/jdet;

    zetanew = (jdet*zetacur+f[2]*(j[1]*j[3]-j[0]*j[4])-f[1]*j[1]*j[6]+
	       f[0]*j[4]*j[6]+f[1]*j[0]*j[7]-f[0]*j[3]*j[7])/jdet;

    xidiff[0] = xinew - xicur;
    xidiff[1] = etanew - etacur;
    xidiff[2] = zetanew - zetacur;
    xicur = xinew;
    etacur = etanew;
    zetacur = zetanew;

  }
  while ( !within_tolerance( vector_norm_sq(xidiff,3), isInElemConverged) && ++i < maxNonlinearIter);

  par_coor[0] = par_coor[1] = par_coor[2] = numeric_limits<double>::max();
  double dist = numeric_limits<double>::max();

  if (i <maxNonlinearIter) {
    par_coor[0] = xinew;
    par_coor[1] = etanew;
    par_coor[2] = zetanew;

    vector<double> xtmp(3);
    xtmp[0] = par_coor[0];
    xtmp[1] = par_coor[1];
    xtmp[2] = par_coor[2];
    dist = parametric_distance(xtmp);
  }
  return dist;

}//end isInElement()

//--------------------------------------------------------------------------
//-------- within_tolerance ------------------------------------------------
//--------------------------------------------------------------------------
bool 
DomainManager::within_tolerance( const double & val, const double & tol )
{
  return (fabs(val)<tol);
}

//--------------------------------------------------------------------------
//-------- vector_norm_sq --------------------------------------------------
//--------------------------------------------------------------------------
double 
DomainManager::vector_norm_sq( const double * vect, int len )
{
  double norm_sq = 0.0;
  for (int i=0; i<len; i++) {
    norm_sq += vect[i]*vect[i];
  }
  return norm_sq;
}

//--------------------------------------------------------------------------
//-------- parametric_distance ---------------------------------------------
//--------------------------------------------------------------------------
double 
DomainManager::parametric_distance(vector<double> x)
{
  vector<double> y(3);
  for (int i=0; i<3; ++i) {
    y[i] = fabs(x[i]);
  }

  double d = 0;
  for (int i=0; i<3; ++i) {
    if (d < y[i]) {
      d = y[i];
    }
  }
  return d;
}
