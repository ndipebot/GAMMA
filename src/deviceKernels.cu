
/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <deviceHeader.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

void createDataOnDeveice(HeatSolverManager *heatMgr_, DomainManager *domainMgr_) {
	int numEl = domainMgr_->nel_;
	int nn = domainMgr_->nn_;

	//Element data
	thrust::device_vector<int> eleNodes(8*numEl);
	thrust::device_vector<double> eleStiffnes(36*numEl);
	thrust::device_vector<double> eleMass(36*numEl);
	thrust::device_vector<double> eleNodeCoords(24*numEl);
	thrust::device_vector<double> eleMat(6*numEl);
	thrust::device_vector<double> volWeight(8*numEl);
	thrust::device_vector<double> eleBirthTimes(numEl);


	// set element values
	for(const auto & elem : domainMgr_->elementList_) {
		 int * nodes = elem->nID_;
		 int eID = &elem - &domainMgr_->elementList_[0]; // grab current element id


		 for(int i = 0; i < 8; ++i) {
			 //assign nodes to device vector in coalesced manner
			 eleNodes[numEl*i + eID] = nodes[i];
			 //grab coalesced nodal coordinates while you're at it
			 for(int j = 0; j < 3; ++j)
				 eleNodeCoords[3*numEl*i + j*numEl + eID] = domainMgr_->coordList_[nodes[i]*3+j];
		 }
		//material properties
	    eleMat[eID] = elem->rho_;
	    eleMat[numEl + eID] = elem->solidus_;
	    eleMat[2*numEl + eID] = elem->liquidus_;
	    eleMat[3*numEl + eID] = elem->latent_;
	    eleMat[4*numEl + eID] = elem->cp_;
	    eleMat[5*numEl + eID] = elem->cond_;

	    //element birth times
	    eleBirthTimes[eID] = elem->birthTime_;
	}




}
