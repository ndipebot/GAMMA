
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
#include <iostream>
#include <helper_cuda.h>
#include <algorithm>

__constant__ double parCoords[24];
__constant__ double parCoordsSurf[72];
__constant__ double parCoords2D[8];
__constant__ int mapIndex[64];

__global__ void updateMass(const double* __restrict__ thetaN, const double* __restrict__ eleMat, const double* __restrict__ eleNodeCoords,
		double* globMass, const int* __restrict__ eleNodes, const int* __restrict__ nUniId, const int numEl, const int nn, const int numElAct);

__global__ void initializeStiffness(const double* __restrict__ eleMat, const double* __restrict__ eleNodeCoords, double* eleStiffness,
		const int* __restrict__ eleNodes, const int numEl, const int nn);

__global__ void getInternalForce(double* globRHS, double* __restrict__ eleStiffness, double* __restrict__ thetaN, 
	const int* __restrict__ eleNodes, const int* __restrict__ nUniId, const int numEl, const int nn, const int numElAct);

__global__ void massReduce(double* __restrict__ globMass, const int nn, const int rhsCount);

__global__ void interForceReduce(double* __restrict__ globRHS, const int nn);

__global__ void applyFlux(const double* __restrict__ surfFlux, const int* __restrict__ surfPlane, const double* __restrict__ thetaN, const int* __restrict__ surfIndx,
	const double* __restrict__ surfNodeCoords, const double* __restrict__ eleNodeCoords, const int* __restrict__ surfNodes, const int* __restrict__ surfElemBirth,
	double* rhs, int numSurf, double ambient, double abszero, double tool0, double tool1, double tool2, int laserState, int numSurfAct, double sigma, int numEl, int nn);

__global__ void massFlux(double* rhs, const int nn, int rhsCount, double* globalRhs);

__global__ void advanceTime(double* thetaN, const double* __restrict__ globalMass, const double* __restrict__ rhs,const int* __restrict__ birthNodes, double dt, int numActNodes);

__global__ void prescribeDirichlet(double* thetaN, const int* __restrict__ fixedNodes, const double* __restrict__ fixedNodeVals, int numFixed);

void compareMass(elementData& elemData, vector<double>& Mvec) {
	int nn = Mvec.size();
	int base = 0;
	for (int i = 0; i < nn; i++) {
		if (abs(elemData.globMass[base + i] - Mvec[base + i]) > 0.000001) {
			std::cout << "Mismatch found on node: "<<i<<", GPU: "<< elemData.globMass[base + i] << ", CPU: "<< Mvec[base + i]<<std::endl;
		}
	}
	std::cout << "check passed!" << std::endl;
}

void compareStiff(elementData& elemData, vector<Element*>& elementList) {
	int numEl = elementList.size();
	for (const auto & elem : elementList) {
		int eID = &elem - &elementList[0];
		for (int i = 0; i < 36; ++i) {
			if (abs(elemData.eleStiffness[eID + i*numEl] - elem->stiffMatrix_[i]) > 0.0001) {
				std::cout << "Mismatch found on element: " << eID << ", GPU: " << elemData.eleStiffness[eID + i*numEl] << ", CPU: " << elem->stiffMatrix_[i] << std::endl;
			}
		}
	}
}

void compareIntForce(elementData& elemData, vector<double>& rhs) {
	int nn = rhs.size();
	int base = 0;
	for (int i = 0; i < nn; i++) {
		std::cout << ", GPU: " << elemData.globRHS[base + i]<< ", CPU: " << rhs[base + i] << std::endl;
		if (abs(elemData.globRHS[base + i] - rhs[base + i]) > 0.000001) {
			std::cout << "Mismatch found on node: " << i << ", GPU: " << elemData.globRHS[base + i] << ", CPU: " << rhs[base + i] << std::endl;
		}
	}
	std::cout << "check passed!" << std::endl;
}

void compareFlux(elementData& elemData, vector<double>& rhs) {
	int nn = rhs.size();
	int base = 0;
	for (int i = 0; i < nn; i++) {
		if (abs(elemData.globRHS_Surf[base + i] - rhs[base + i]) > 0.000001) {
			std::cout << "Mismatch found on node: " << i << ", GPU: " << elemData.globRHS_Surf[base + i] << ", CPU: " << rhs[base + i] << std::endl;
		}
	}
	std::cout << "check passed!" << std::endl;
}

void compareTemp(elementData& elemData, vector<double>& thetaN ) {
	int base = 0;
	for (int i = 0; i < elemData.nn; i++) {
		if (abs(elemData.thetaN[base + i] - thetaN[base + i]) > 0.000001) {
			std::cout << "Mismatch found on node: " << i << ", GPU: " << elemData.thetaN[base + i] << ", CPU: " << thetaN[base + i] << std::endl;
		}
	}
	std::cout << "check passed!" << std::endl;
}

void createDataOnDevice(DomainManager*& domainMgr, elementData& elemData, HeatSolverManager*& heatMgr) {
    //Start recording time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Start record
    cudaEventRecord(start, 0);

	/*
	* ELEMENTS
	*/

	int numEl = domainMgr->nel_;
	int nn = domainMgr->nn_;


	elemData.initTemp = heatMgr->initTheta_;
	elemData.nn = nn;
	elemData.numEl = numEl;
	elemData.dt = heatMgr->dt_;

	//Element data
	elemData.eleNodes.resize(8*numEl);
	elemData.eleNodeCoords.resize(24*numEl);
	elemData.eleMat.resize(6*numEl);
	elemData.eleBirthTimes.resize(numEl);
	elemData.thetaN.resize(nn, elemData.initTemp);

	elemData.eleStiffness.resize(numEl*36);


	elemData.nUniId.resize(numEl * 8);
	vector<int> nodeInd;
	nodeInd.resize(nn,0);
	int eleRowCnt = 0;

	// set element values
	for(const auto & elem : domainMgr->elementList_) {
		 int * nodes = elem->nID_;
		 int eID = &elem - &domainMgr->elementList_[0]; // grab current element id


		 for(int i = 0; i < 8; ++i) {
			 //assign nodes to device vector in coalesced manner
			 elemData.eleNodes[numEl*i + eID] = nodes[i];
			 elemData.nUniId[numEl*i + eID] = nodeInd[nodes[i]]++;
			 eleRowCnt = std::max(eleRowCnt, nodeInd[nodes[i]]);

			 //grab coalesced nodal coordinates while you're at it
			 for(int j = 0; j < 3; ++j)
				 elemData.eleNodeCoords[3*numEl*i + j*numEl + eID] = domainMgr->coordList_[nodes[i]*3+j];
		 }
		//material properties
		elemData.eleMat[eID] = elem->rho_;
		elemData.eleMat[numEl + eID] = elem->solidus_;
		elemData.eleMat[2*numEl + eID] = elem->liquidus_;
		elemData.eleMat[3*numEl + eID] = elem->latent_;
		elemData.eleMat[4*numEl + eID] = elem->cp_;
		elemData.eleMat[5*numEl + eID] = elem->cond_;

	    //element birth times
		elemData.eleBirthTimes[eID] = elem->birthTime_;
	}
	elemData.rhsCountEle = eleRowCnt;
	//-----------------------  END OF ELEMENT -----------------------------------------------///

	/*
	*  SURFACES
	*/
	// Surface ninjutsu
	vector<double> tempFlux;
	vector<int> tempNodes;
	vector<double> tempCoords;

	//grab static surface
	for (auto const surfID : heatMgr->heatBCManager_->activeSurfaces_) {
		Surface surf = heatMgr->heatBCManager_->staticSurfList_[surfID];
		elemData.boundSurfBirthTime.push_back(surf.birthTime_);
		elemData.boundSurfDeathTime.push_back(surf.deathTime_);
		elemData.surfPlane.push_back(surf.plane_);
		elemData.surfBirthElem.push_back(surf.birthElement_);

		for (int i = 0; i < 4; ++i)
			tempNodes.push_back(surf.surfaceNodes_[i]);

		for (int i = 0; i < 5; ++i)
			tempFlux.push_back(surf.flux[i]);

		for (int i = 0; i < 8; ++i)
			tempCoords.push_back(surf.mappedCoords[i]);
	}

	//grab birthSurfaces
	for (auto const surf : heatMgr->heatBCManager_->surfaceList_) {
		if (surf.isFlux_) {
			elemData.boundSurfBirthTime.push_back(surf.birthTime_);
			elemData.boundSurfDeathTime.push_back(surf.deathTime_);
			elemData.surfPlane.push_back(surf.plane_);
			elemData.surfBirthElem.push_back(surf.birthElement_);

			for (int i = 0; i < 4; ++i)
				tempNodes.push_back(surf.surfaceNodes_[i]);

			for (int i = 0; i < 5; ++i)
				tempFlux.push_back(surf.flux[i]);

			for (int i = 0; i < 8; ++i)
				tempCoords.push_back(surf.mappedCoords[i]);
		}
	}

	//get number of surfaces in total
	elemData.numSurf = elemData.boundSurfBirthTime.size();
	int numSurf = elemData.numSurf;
	elemData.surfNodes.resize(4 * numSurf, 0); // nodes
	elemData.surfIndx.resize(4*numSurf, 0);  // surface location
	elemData.surfNodeCoords.resize(8 * numSurf, 0);
	elemData.surfFlux.resize(5 * numSurf, 0); // flux contains [flux, flux variable], flux is 0 or 1

											  //count index nodes for rhs vector
	vector<int> nodeIndex(nn, 0);
	int maxCount = 0;

	//order surface nodes, nodal coordinates, and flux for coalesced accesses
	for (int i = 0; i < numSurf; ++i) {
		for (int j = 0; j < 4; ++j) {
			//nodes
			int nodeID = tempNodes[4 * i + j];
			elemData.surfNodes[j*numSurf + i] = nodeID;
			elemData.surfIndx[i*4 + j] = nodeIndex[nodeID]++;
			maxCount = std::max(maxCount, nodeIndex[nodeID]);
			//coordinates
			for (int k = 0; k < 2; ++k)
				elemData.surfNodeCoords[2 * numSurf*j + k*numSurf + i] = tempCoords[i * 8 + j*2 + k];
		}

		//flux
		for (int j = 0; j < 5; ++j)
			elemData.surfFlux[j*numSurf + i] = tempFlux[i * 5 + j];
	}

	elemData.rhsCount = maxCount;

	elemData.ambient = domainMgr->meshObj_->Rambient_;
	elemData.abszero = domainMgr->meshObj_->Rabszero_;
	elemData.sigma = domainMgr->meshObj_->Rboltz_;

	elemData.globMass.resize(nn * 12);
	elemData.globRHS.resize(nn * 12);
	elemData.globRHS_Surf.resize(nn * maxCount);


	//------------- Fixed Nodes --------------------------
	elemData.numFixed = heatMgr->heatBCManager_->fixedNodeIDs_.size();
	elemData.fixedNodes = heatMgr->heatBCManager_->fixedNodeIDs_.data();
	elemData.fixedValues = heatMgr->heatBCManager_->fixedNodeVals_.data();

	//------------------------birth Nodes
	for(int i = 0; i < domainMgr->activeNodes_.size(); ++i) {
		elemData.birthNodes.push_back(domainMgr->activeNodes_[i]);
		elemData.birthNodeTimes.push_back(0.0);
	}

	for(int i = 0; i < domainMgr->birthNodes_.size(); ++i) {
		elemData.birthNodes.push_back(domainMgr->birthNodes_[i]);
		elemData.birthNodeTimes.push_back(domainMgr->birthNodeTimes_[i]);
	}

	//----------------------------------------------------

	//Alocate and Copy
	AllocateDeviceData(elemData);
	CopyToDevice(elemData);

	//move parametric coordinates to local memory

	vector<double> coords(24);

	vector<double> coeff {-1.0, -1.0, -1.0,
							1.0, -1.0, -1.0,
							1.0,  1.0, -1.0,
							-1.0,  1.0, -1.0,
							-1.0, -1.0,  1.0,
							1.0, -1.0,  1.0,
							1.0,  1.0,  1.0,
							-1.0,  1.0,  1.0};

	  for (int j = 0; j < 8; j++)
	    for (int i = 0; i < 3; i++)
			coords[j * 3 + i] = coeff[j * 3 + i] * 0.5773502692;

      cudaMemcpyToSymbol(parCoords,coords.data(),24*sizeof(double));

	  //move surface parametric coordinates to local memory
	  vector<vector<double>> coeff2D{ { -1.0 / sqrt(3.0), -1.0 / sqrt(3.0) },
	  { -1.0 / sqrt(3.0),  1.0 / sqrt(3.0) },
	  { 1.0 / sqrt(3.0), -1.0 / sqrt(3.0) },
	  { 1.0 / sqrt(3.0),  1.0 / sqrt(3.0) } };

	  vector<double> coordsSurf(72);

	  for (int ip = 0; ip < 4; ++ip) {
		  coordsSurf[ip * 18] = coeff2D[ip][0];
		  coordsSurf[ip * 18 + 1] = coeff2D[ip][1];
		  coordsSurf[ip * 18 + 2] = 1.0;

		  coordsSurf[ip * 18 + 3] = coeff2D[ip][0];
		  coordsSurf[ip * 18 + 4] = coeff2D[ip][1];
		  coordsSurf[ip * 18 + 5] = -1.0;

		  coordsSurf[ip * 18 + 6] = coeff2D[ip][0];
		  coordsSurf[ip * 18 + 7] = -1.0;
		  coordsSurf[ip * 18 + 8] = coeff2D[ip][1];

		  coordsSurf[ip * 18 + 9] = coeff2D[ip][0];
		  coordsSurf[ip * 18 + 10] = 1.0;
		  coordsSurf[ip * 18 + 11] = coeff2D[ip][1];

		  coordsSurf[ip * 18 + 12] = -1.0;
		  coordsSurf[ip * 18 + 13] = coeff2D[ip][0];
		  coordsSurf[ip * 18 + 14] = coeff2D[ip][1];

		  coordsSurf[ip * 18 + 15] = 1.0;
		  coordsSurf[ip * 18 + 16] = coeff2D[ip][0];
		  coordsSurf[ip * 18 + 17] = coeff2D[ip][1];
	  }

	  cudaMemcpyToSymbol(parCoordsSurf, coordsSurf.data(), 72 * sizeof(double));

	  // 2D surface parametric coordinates to constant memory
	  vector<double> coords2D{ -0.5773502692, -0.5773502692,
		  -0.5773502692,  0.5773502692,
		  0.5773502692, -0.5773502692,
		  0.5773502692,  0.5773502692 };

	  cudaMemcpyToSymbol(parCoords2D, coords2D.data(), 8 * sizeof(double));

		vector<int> mapIndx{ 0,  1,  2,  3,  4,  5,  6,  7,
							1,  8,  9, 10, 11, 12, 13, 14 ,
							2,  9, 15, 16, 17, 18, 19, 20 ,
							3, 10, 16, 21, 22, 23, 24, 25 ,
							4, 11, 17, 22, 26, 27, 28, 29 ,
							5, 12, 18, 23, 27, 30, 31, 32 ,
							6, 13, 19, 24, 28, 31, 33, 34 ,
							7, 14, 20, 25, 29, 32, 34, 35 };

	  cudaMemcpyToSymbol(mapIndex, mapIndx.data(), 8 * sizeof(int));

      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      float elapsedTime;
      cudaEventElapsedTime(&elapsedTime, start, stop); // that's our time!
      // Clean up:
      cudaEventDestroy(start);
      cudaEventDestroy(stop);

	  std::cout << "Device data setup took " << (double)elapsedTime/1000 << " seconds" << std::endl;

}

void AllocateDeviceData(elementData& elem) {
	int nn = elem.nn;
	int numEl = elem.numEl;
	int numSurf = elem.numSurf;

	//Allocate device arrays
	checkCudaErrors(cudaMalloc((void**)&elem.dGlobMass, elem.rhsCountEle*nn*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&elem.dGlobRHS, 12 * nn * sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&elem.dEleStiffness, 36*numEl*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&elem.dEleNodes, 8*numEl*sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&elem.dEleNodeCoords, 24*numEl*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&elem.dEleMat, 6*numEl*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&elem.dEleBirthTimes, numEl*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&elem.dthetaN, nn*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&elem.dNUniId, 8*numEl * sizeof(int)));

	checkCudaErrors(cudaMalloc((void**)&elem.dSurfNodes, 4 * numSurf * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&elem.dSurfNodeCoords, 8 * numSurf * sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&elem.dSurfIndx, 4 * numSurf * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&elem.dSurfPlane, numSurf * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&elem.dSurfFlux, 5 * numSurf * sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&elem.dSurfBirthElem, numSurf * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&elem.dGlobRHS_Surf, nn*elem.rhsCount * sizeof(double)));

	checkCudaErrors(cudaMalloc((void**)&elem.dFixedNodes, elem.numFixed * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&elem.dFixedNodeVals, elem.numFixed * sizeof(double)));

	checkCudaErrors(cudaMalloc((void**)&elem.dBirthNodes, nn * sizeof(int)));


	cudaMemset(elem.dEleStiffness, 0, 36*numEl*sizeof(double));
}

void CopyToDevice(elementData& elem) {
	int nn = elem.nn;
	int numEl = elem.numEl;
	int numSurf = elem.numSurf;
	checkCudaErrors(cudaMemcpy(elem.dEleNodes,elem.eleNodes.data(),8*numEl*sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(elem.dEleNodeCoords,elem.eleNodeCoords.data(),24*numEl*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(elem.dEleMat,elem.eleMat.data(), 6*numEl*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(elem.dEleBirthTimes,elem.eleBirthTimes.data(),numEl*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(elem.dthetaN, elem.thetaN.data(), nn*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(elem.dNUniId, elem.nUniId.data(), 8*numEl * sizeof(int), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(elem.dSurfNodes, elem.surfNodes.data(), 4 * numSurf * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(elem.dSurfIndx, elem.surfIndx.data(), 4 * numSurf * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(elem.dSurfNodeCoords, elem.surfNodeCoords.data(), 8 * numSurf * sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(elem.dSurfPlane, elem.surfPlane.data(), numSurf * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(elem.dSurfFlux, elem.surfFlux.data(), 5 * numSurf * sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(elem.dSurfBirthElem, elem.surfBirthElem.data(), numSurf * sizeof(int), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(elem.dFixedNodes, elem.fixedNodes, elem.numFixed * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(elem.dFixedNodeVals, elem.fixedValues, elem.numFixed * sizeof(double), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(elem.dBirthNodes, elem.birthNodes.data(), nn * sizeof(int), cudaMemcpyHostToDevice));
}

void CopyToHost(elementData& elem) {
	int nn = elem.nn;
	//int numEl = elem.numEl;

	//checkCudaErrors(cudaMemcpy(elem.eleStiffness.data(), elem.dEleStiffness, 36*numEl*sizeof(double), cudaMemcpyDeviceToHost));
	//checkCudaErrors(cudaMemcpy(elem.globMass.data(), elem.dGlobMass, 12*nn*sizeof(double), cudaMemcpyDeviceToHost));
	//checkCudaErrors(cudaMemcpy(elem.globRHS.data(), elem.dGlobRHS, 12*nn * sizeof(double), cudaMemcpyDeviceToHost));
	//checkCudaErrors(cudaMemcpy(elem.globRHS_Surf.data(), elem.dGlobRHS_Surf, elem.rhsCount * nn * sizeof(double), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(elem.thetaN.data(), elem.dthetaN,  nn * sizeof(double), cudaMemcpyDeviceToHost));
}

void FreeDevice(elementData& elem) {
	checkCudaErrors(cudaFree(elem.dGlobMass));
	checkCudaErrors(cudaFree(elem.dEleStiffness));
	checkCudaErrors(cudaFree(elem.dEleNodes));
	checkCudaErrors(cudaFree(elem.dEleNodeCoords));
	checkCudaErrors(cudaFree(elem.dEleMat));
	checkCudaErrors(cudaFree(elem.dEleBirthTimes));
	checkCudaErrors(cudaFree(elem.dthetaN));
	checkCudaErrors(cudaFree(elem.dNUniId));
	checkCudaErrors(cudaFree(elem.dGlobRHS));

	checkCudaErrors(cudaFree(elem.dSurfNodes));
	checkCudaErrors(cudaFree(elem.dSurfIndx));
	checkCudaErrors(cudaFree(elem.dSurfNodeCoords));
	checkCudaErrors(cudaFree(elem.dSurfPlane));
	checkCudaErrors(cudaFree(elem.dSurfFlux));
	checkCudaErrors(cudaFree(elem.dSurfBirthElem));
	checkCudaErrors(cudaFree(elem.dGlobRHS_Surf));

	checkCudaErrors(cudaFree(elem.dFixedNodes));
	checkCudaErrors(cudaFree(elem.dFixedNodeVals));

	checkCudaErrors(cudaFree(elem.dBirthNodes));
}

void clearDeviceData(elementData& elem) {
	//Clear some arrays
	int nn = elem.nn;
	cudaMemset(elem.dGlobMass, 0, 12*nn*sizeof(double));
	cudaMemset(elem.dGlobRHS, 0, 12 * nn * sizeof(double));
	cudaMemset(elem.dGlobRHS_Surf, 0, elem.rhsCount*nn * sizeof(double));
}

void initializeStiffnessOnD(elementData& elemData) {
	cudaError_t cudaStatus;
	int gridSize = elemData.numEl/256 + 1;
	int blockSize = 256;

    //Start recording time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Start record
    cudaEventRecord(start, 0);

	initializeStiffness<<<gridSize, blockSize>>>(elemData.dEleMat, elemData.dEleNodeCoords, elemData.dEleStiffness,
				elemData.dEleNodes, elemData.numEl, elemData.nn);

	cudaStatus = cudaGetLastError();
	if(cudaStatus != cudaSuccess) {
		std::cout <<"initialiazeStiffness Kernel failed: "<<cudaGetErrorString(cudaStatus)<< std::endl;
		FreeDevice(elemData);
	}

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop); // that's our time!
    // Clean up:
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

	//std::cout << "initialize Stiffness took " << (double)elapsedTime/1000 << " seconds" << std::endl;
}

void updateMassOnD(elementData& elemData, DomainManager*& domainMgr) {
	cudaError_t cudaStatus;


	//find birth point
	auto up = std::upper_bound(domainMgr->elementList_.begin(), domainMgr->elementList_.end(),
		domainMgr->currTime_, [](const double bTime, Element* a) {return a->birthTime_ > bTime; });

	int birthElemPos = up - domainMgr->elementList_.begin();

	int gridSize = birthElemPos / 256 + 1;
	int blockSize = 256;

    //Start recording time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Start record
    cudaEventRecord(start, 0);

	updateMass<<<gridSize, blockSize>>>(elemData.dthetaN, elemData.dEleMat, elemData.dEleNodeCoords,
			elemData.dGlobMass, elemData.dEleNodes, elemData.dNUniId, elemData.numEl, elemData.nn, birthElemPos);

	cudaStatus = cudaGetLastError();
	if(cudaStatus != cudaSuccess) {
		std::cout <<"updateMass Kernel failed: "<<cudaGetErrorString(cudaStatus)<< std::endl;
		FreeDevice(elemData);
	}

	gridSize= elemData.nn / 256 + 1;
	massReduce <<<gridSize, blockSize >>> (elemData.dGlobMass, elemData.nn, elemData.rhsCountEle);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		cout << "massReduce Kernel failed: " << cudaGetErrorString(cudaStatus) << endl;
		FreeDevice(elemData);
	}


    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop); // that's our time!
    // Clean up:
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

	//std::cout << "update Mass took " << (double)elapsedTime/1000 << " seconds" << std::endl;

}

void updateIntForceOnD(elementData& elemData, DomainManager*& domainMgr) {
	cudaError_t cudaStatus;

	//find birth point
	auto up = std::upper_bound(domainMgr->elementList_.begin(), domainMgr->elementList_.end(),
		domainMgr->currTime_, [](const double bTime, Element* a) {return a->birthTime_ > bTime; });

	int birthElemPos = up - domainMgr->elementList_.begin();

	int gridSize = birthElemPos / 256 + 1;
	int blockSize = 256;

	//Start recording time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Start record
	cudaEventRecord(start, 0);

	getInternalForce <<<gridSize, blockSize >>> (elemData.dGlobRHS, elemData.dEleStiffness, elemData.dthetaN,
		elemData.dEleNodes, elemData.dNUniId, elemData.numEl, elemData.nn, birthElemPos);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		std::cout << "Internal-force Kernel failed: " << cudaGetErrorString(cudaStatus) << std::endl;
		FreeDevice(elemData);
	}

	gridSize = elemData.nn / 256 + 1;
	interForceReduce <<<gridSize, blockSize >>> (elemData.dGlobRHS, elemData.nn);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		cout << "interForceReduce Kernel failed: " << cudaGetErrorString(cudaStatus) << endl;
		FreeDevice(elemData);
	}


	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop); // that's our time!
													 // Clean up:
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//std::cout << "Internal force took " << (double)elapsedTime / 1000 << " seconds" << std::endl;

}

void updateFluxKernel(elementData& elemData, DomainManager*& domainMgr) {
	cudaError_t cudaStatus;

	//Start recording time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Start record
	cudaEventRecord(start, 0);

	/*
	*
	* Tool position calculation
	*
	*/

	// current point on toolpath
	auto up = std::upper_bound(domainMgr->tooltxyz_.begin(), domainMgr->tooltxyz_.end(), domainMgr->currTime_,
		[](const double Ctime, vector<double, allocator<double>> a) {return a[0] > Ctime; });

	if(up == domainMgr->tooltxyz_.end())
		up--;

	int toolpathIndex = up - domainMgr->tooltxyz_.begin();

	vector<double> & txyzN = domainMgr->tooltxyz_[toolpathIndex];
	vector<double> & txyzNminus = domainMgr->tooltxyz_[toolpathIndex - 1];

	double num = domainMgr->currTime_ - txyzNminus[0];
	double den = txyzN[0] - txyzNminus[0];
	double rat = num / den;
	elemData.tool0 = rat * (txyzN[1] - txyzNminus[1]) + txyzNminus[1];
	elemData.tool1 = rat * (txyzN[2] - txyzNminus[2]) + txyzNminus[2];
	elemData.tool2 = rat * (txyzN[3] - txyzNminus[3]) + txyzNminus[3];
	elemData.laserState = domainMgr->laserOn_[toolpathIndex];
	// ---------------------------------------- END OF TOOL CALCULATION ------------------------------//


	//find birth point
	auto up2 = std::upper_bound(elemData.boundSurfBirthTime.begin(), elemData.boundSurfBirthTime.end(),
		domainMgr->currTime_, [](double a, double bTime) {return a < bTime; });

	int birthSurfPos = up2 - elemData.boundSurfBirthTime.begin();



	int gridSize = birthSurfPos / 256 + 1;
	int blockSize = 256;


	applyFlux <<<gridSize, blockSize >>>(elemData.dSurfFlux, elemData.dSurfPlane, elemData.dthetaN, elemData.dSurfIndx,elemData.dSurfNodeCoords, elemData.dEleNodeCoords,
		elemData.dSurfNodes, elemData.dSurfBirthElem, elemData.dGlobRHS_Surf, elemData.numSurf, elemData.ambient, elemData.abszero, elemData.tool0, elemData.tool1, elemData.tool2,
		elemData.laserState, birthSurfPos, elemData.sigma, elemData.numEl, elemData.nn);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		std::cout << "Apply Flux Kernel failed: " << cudaGetErrorString(cudaStatus) << std::endl;
		FreeDevice(elemData);
	}

	gridSize = elemData.nn / 256 + 1;
	massFlux <<<gridSize, blockSize >>>(elemData.dGlobRHS_Surf, elemData.nn, elemData.rhsCount, elemData.dGlobRHS);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		std::cout << "Mass Flux Kernel failed: " << cudaGetErrorString(cudaStatus) << std::endl;
		FreeDevice(elemData);
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop); // that's our time!
													 // Clean up:
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//std::cout << "Apply Flux Kernel took " << (double)elapsedTime / 1000 << " seconds" << std::endl;


}

void advanceTimeKernel(elementData& elemData, DomainManager*& domainMgr) {
	cudaError_t cudaStatus;

	//Start recording time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Start record
	cudaEventRecord(start, 0);

	//get active node count
	auto up = std::upper_bound(elemData.birthNodeTimes.begin(), elemData.birthNodeTimes.end(), domainMgr->currTime_,
			[](double a, double bTime) {return a < bTime; });

	int nodeCount = up - elemData.birthNodeTimes.begin();


	int gridSize = nodeCount / 256 + 1;
	int blockSize = 256;

	advanceTime <<<gridSize, blockSize>>>(elemData.dthetaN, elemData.dGlobMass, elemData.dGlobRHS, elemData.dBirthNodes ,elemData.dt,nodeCount);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		std::cout << "Apply Flux Kernel failed: " << cudaGetErrorString(cudaStatus) << std::endl;
		FreeDevice(elemData);
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop); // that's our time!
													 // Clean up:
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//std::cout << "Advance Time Kernel took " << (double)elapsedTime / 1000 << " seconds" << std::endl;

}

void dirichletBCKernel(elementData& elemData) {
	cudaError_t cudaStatus;

	//Start recording time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Start record
	cudaEventRecord(start, 0);

	int gridSize = elemData.numFixed / 256 + 1;
	int blockSize = 256;

	prescribeDirichlet <<<gridSize, blockSize>>> (elemData.dthetaN, elemData.dFixedNodes, elemData.dFixedNodeVals, elemData.numFixed);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		std::cout << "Apply Flux Kernel failed: " << cudaGetErrorString(cudaStatus) << std::endl;
		FreeDevice(elemData);
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop); // that's our time!
													 // Clean up:
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//std::cout << "Prescribe Dirichlet  Kernel took " << (double)elapsedTime / 1000 << " seconds" << std::endl;
}

__global__ void updateMass(const double* __restrict__ thetaN, const double* __restrict__ eleMat, const double* __restrict__ eleNodeCoords,
		double* globMass, const int* __restrict__ eleNodes, const int* __restrict__ nUniId, const int numEl, const int nn, const int numElAct) {

	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	__shared__ int parCoordsS[24];

	if(idx < 24)
		parCoordsS[idx] = parCoords[idx];

	__syncthreads();

	if(idx < numElAct) {

		  // elemental material properties
		double liquidus = eleMat[idx + 2*numEl];
		double solidus = eleMat[idx + numEl];
		double latent = eleMat[idx + 3*numEl];
		double cp =  eleMat[idx + 4*numEl];
		double rho = eleMat[idx];

		double coords[24];

		//shape function
		double N[8];
		double chsi;
		double eta;
		double zeta;

		double deriv[24];
		double Jac[9];

		double thetaIp = 0.0;
		double cpPoint = 0;
		double detJac = 0;

		for(int ip = 0; ip < 8; ++ip) {

			//compute shape function
			chsi = parCoordsS[ip*3 + 0];
			eta = parCoordsS[ip*3 + 1];
			zeta = parCoordsS[ip*3 + 2];
			N[0] = 0.125*(1.0 - chsi)*(1.0 - eta)*(1.0 - zeta);
			N[3] = 0.125*(1.0 - chsi)*(1.0 + eta)*(1.0 - zeta);
			N[2] = 0.125*(1.0 + chsi)*(1.0 + eta)*(1.0 - zeta);
			N[1] = 0.125*(1.0 + chsi)*(1.0 - eta)*(1.0 - zeta);
			N[4] = 0.125*(1.0 - chsi)*(1.0 - eta)*(1.0 + zeta);
			N[7] = 0.125*(1.0 - chsi)*(1.0 + eta)*(1.0 + zeta);
			N[6] = 0.125*(1.0 + chsi)*(1.0 + eta)*(1.0 + zeta);
			N[5] = 0.125*(1.0 + chsi)*(1.0 - eta)*(1.0 + zeta);

			//Calculate temperature at integration points
			for(int i = 0; i < 8; ++i) {
				int ig = eleNodes[numEl*i + idx];
				thetaIp += N[i]*thetaN[ig];
			}

			//compute cp
			cpPoint = (thetaIp <= solidus && thetaIp >= liquidus) ? (cp + latent / ( liquidus - solidus )) : cp;

			//compute derivative of shape functions
			// with respect to chsi
			deriv[0] = -0.1250 * (1 - eta) * (1 - zeta);
			deriv[2] =  0.1250 * (1 + eta) * (1 - zeta);
			deriv[4] = -0.1250 * (1 - eta) * (1 + zeta);
			deriv[6] =  0.1250 * (1 + eta) * (1 + zeta);
			for (int i = 0; i < 4; i++)
				deriv[i * 2 + 1] = -deriv[i * 2];

			// with respect to eta
			deriv[0 + 8] = -0.1250 * (1 - chsi) * (1 - zeta);
			deriv[3 + 8] = -deriv[8];
			deriv[1 + 8] = -0.1250 * (1 + chsi) * (1 - zeta);
			deriv[2 + 8] = -deriv[9];
			deriv[4 + 8] = -0.1250 * (1 - chsi) * (1 + zeta);
			deriv[7 + 8] = -deriv[12];
			deriv[5 + 8] = -0.1250 * (1 + chsi) * (1 + zeta);
			deriv[6 + 8] = -deriv[13];

			// with respect to zeta
			deriv[4 + 16] = 0.1250 * (1 - chsi) * (1 - eta);
			deriv[5 + 16] = 0.1250 * (1 + chsi) * (1 - eta);
			deriv[6 + 16] = 0.1250 * (1 + chsi) * (1 + eta);
			deriv[7 + 16] = 0.1250 * (1 - chsi) * (1 + eta);
			for (int i = 0; i < 4; i++)
				deriv[i + 16] = -deriv[i + 20];

			// get coordinates
			for(int i = 0; i < 8; i++) {
				coords[3*i] = eleNodeCoords[3*numEl*i  + idx];
				coords[3*i + 1] = eleNodeCoords[3*numEl*i + numEl + idx];
				coords[3*i + 2] = eleNodeCoords[3*numEl*i + 2*numEl + idx];
			 }

			for (int k = 0; k < 9; k++)
				Jac[k] = 0;

			// Compute Jacobian
			for (int k = 0; k < 3; k++)
				for (int j = 0; j < 3; j++)
					for (int i = 0; i < 8; i++)
						Jac[k* 3 + j] += deriv[k * 8 + i] * coords[i * 3 + j];


			//determinant of Jacobian
			detJac = Jac[0] * Jac[4] * Jac[8] + Jac[1] * Jac[5] * Jac[6] +
					Jac[3] * Jac[7] * Jac[2] - Jac[2] * Jac[4] * Jac[6] -
					Jac[0] * Jac[5] * Jac[7] - Jac[1] * Jac[3] * Jac[8];

			//Calculate mass matrix
			for (int i = 0; i < 8; i++) {
				int ig = eleNodes[numEl*i + idx];
				for (int j = 0; j < 8; j++)
					globMass[nUniId[i*numEl+idx]*nn + ig] += N[i] * rho * cpPoint * N[j] * detJac;
			}

		}
	}

}

__global__ void initializeStiffness(const double* __restrict__ eleMat, const double* __restrict__ eleNodeCoords,
		double* eleStiffness, const int* __restrict__ eleNodes, const int numEl, const int nn) {

	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	__shared__ int parCoordsS[24];

	if(idx < 24)
		parCoordsS[idx] = parCoords[idx];

	__syncthreads();

	if(idx < numEl) {

		double coords[24];

		//shape function

		double chsi;
		double eta;
		double zeta;

		double iJac[9];
		double Jac[9];
		double gradN[24];
		double deriv[24];
		double swap;


		double detJac = 0;

		double kappa = eleMat[5*numEl + idx];

		for(int ip = 0; ip < 8; ++ip) {

			for(int i = 0; i < 24; i++)
				gradN[i] = 0;

			for(int i = 0; i < 9; i++)
				Jac[i] = 0;

			//compute shape function
			chsi = parCoordsS[ip*3 + 0];
			eta = parCoordsS[ip*3 + 1];
			zeta = parCoordsS[ip*3 + 2];

			//compute derivative of shape functions

			// with respect to chsi
			deriv[0] = -0.1250 * (1 - eta) * (1 - zeta);
			deriv[2] =  0.1250 * (1 + eta) * (1 - zeta);
			deriv[4] = -0.1250 * (1 - eta) * (1 + zeta);
			deriv[6] =  0.1250 * (1 + eta) * (1 + zeta);
			for (int i = 0; i < 4; i++)
				deriv[i * 2 + 1] = -deriv[i * 2];

			// with respect to eta
			deriv[0 + 8] = -0.1250 * (1 - chsi) * (1 - zeta);
			deriv[3 + 8] = -deriv[8];
			deriv[1 + 8] = -0.1250 * (1 + chsi) * (1 - zeta);
			deriv[2 + 8] = -deriv[9];
			deriv[4 + 8] = -0.1250 * (1 - chsi) * (1 + zeta);
			deriv[7 + 8] = -deriv[12];
			deriv[5 + 8] = -0.1250 * (1 + chsi) * (1 + zeta);
			deriv[6 + 8] = -deriv[13];

			// with respect to zeta
			deriv[4 + 16] = 0.1250 * (1 - chsi) * (1 - eta);
			deriv[5 + 16] = 0.1250 * (1 + chsi) * (1 - eta);
			deriv[6 + 16] = 0.1250 * (1 + chsi) * (1 + eta);
			deriv[7 + 16] = 0.1250 * (1 - chsi) * (1 + eta);
			for (int i = 0; i < 4; i++)
				deriv[i + 16] = -deriv[i + 20];

			// get coordinates
			for(int i = 0; i < 8; i++) {
				coords[3*i] = eleNodeCoords[3*numEl*i  + idx];
				coords[3*i + 1] = eleNodeCoords[3*numEl*i + numEl + idx];
				coords[3*i + 2] = eleNodeCoords[3*numEl*i + 2*numEl + idx];
			 }


			// Compute Jacobian
			for (int k = 0; k < 3; k++)
				for (int j = 0; j < 3; j++)
					for (int i = 0; i < 8; i++)
						Jac[k* 3 + j] += deriv[k * 8 + i] * coords[i * 3 + j];

			//determinant of Jacobian
			detJac = Jac[0] * Jac[4] * Jac[8] + Jac[1] * Jac[5] * Jac[6] +
					Jac[3] * Jac[7] * Jac[2] - Jac[2] * Jac[4] * Jac[6] -
					Jac[0] * Jac[5] * Jac[7] - Jac[1] * Jac[3] * Jac[8];


			//compute inverse of Jacobian

			iJac[0] = (1/detJac) * (Jac[4] * Jac[8] - Jac[5] * Jac[7]);
			iJac[1] = (-1/detJac) * (Jac[3] * Jac[8] - Jac[5] * Jac[6]);
			iJac[2] = (1/detJac) * (Jac[3] * Jac[7] - Jac[4] * Jac[6]);

			iJac[3] = (-1/detJac) * (Jac[1] * Jac[8] - Jac[2] * Jac[7]);
			iJac[4] = (1/detJac) * (Jac[0] * Jac[8] - Jac[2] * Jac[6]);
			iJac[5] = (-1/detJac) * (Jac[0] * Jac[7] - Jac[1] * Jac[6]);

			iJac[6] = (1/detJac) * (Jac[1] * Jac[5] - Jac[2] * Jac[4]);
			iJac[7] = (-1/detJac) * (Jac[0] * Jac[5] - Jac[2] * Jac[3]);
			iJac[8] = (1/detJac) * (Jac[0] * Jac[4] - Jac[1] * Jac[3]);

			swap = iJac[1];
			iJac[1] = iJac[3];
			iJac[3] = swap;

			swap = iJac[2];
			iJac[2] = iJac[6];
			iJac[6] = swap;

			swap = iJac[5];
			iJac[5] = iJac[7];
			iJac[7] = swap;

			//compute derivative of shape function w.r.t real coordinates


			for (int k = 0; k < 8; k++)
				for (int j = 0; j < 3; j++)
				  for (int i = 0; i < 3; i++)
					gradN[j*8 + k] += iJac[j*3 + i] * deriv[i*8 + k];


			//Calculate element stiffness matrix
			int count = 0;
			for (int i=0; i<8; i++)
			  for (int j = i; j < 8; j++) {
				for (int k = 0; k < 3; k++)
					eleStiffness[count*numEl + idx] += gradN[k * 8 + i] * kappa * gradN[k * 8 + j] * detJac;

				count++;
			  }
		}
	}
}

__global__ void massReduce(double* __restrict__ globMass, const int nn, const int rhsCount) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < nn)
		for (int elInd = 1; elInd < rhsCount; elInd++)
			globMass[idx] += globMass[idx + nn*elInd];

}

__global__ void getInternalForce(double* globRHS, double* __restrict__ eleStiffness, double* __restrict__ thetaN, 
	const int* __restrict__ eleNodes, const int* __restrict__ nUniId, const int numEl, const int nn, const int numElAct) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	__shared__ int mapIndexS[64];

	if(idx < 64)
		mapIndexS[idx] = mapIndex[idx];

	__syncthreads();

	if(idx < numElAct) {
		for (int row = 0; row < 8; row++)
		{
			int ig = eleNodes[numEl*row + idx];
			for (int col = 0; col < 8; col++)
			{
				int stiffInd = mapIndexS[row*8 + col];
				globRHS[nUniId[row*numEl + idx] * nn + ig] -= eleStiffness[idx + stiffInd*numEl] * thetaN[eleNodes[idx + col*numEl]];
			}
		}
	}
}

__global__ void interForceReduce(double* __restrict__ globRHS, const int nn) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < nn)
		for (int elInd = 1; elInd < 12; elInd++)
			globRHS[idx] += globRHS[idx + nn*elInd];
}

__global__ void applyFlux(const double* __restrict__ surfFlux, const int* __restrict__ surfPlane, const double* __restrict__ thetaN, const int* __restrict__ surfIndx,
	const double* __restrict__ surfNodeCoords, const double* __restrict__ eleNodeCoords, const int* __restrict__ surfNodes, const int* __restrict__ surfElemBirth,
	double* rhs, int numSurf, double ambient, double abszero, double tool0, double tool1, double tool2, int laserState, int numSurfAct, double sigma, int numEl, int nn) {

	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	//move to shared memory for speed
	__shared__ int parCoordsSurfS[72];
	__shared__ int parCoords2DS[8];

	if(idx < 8)
		parCoords2DS[idx] = parCoords2D[idx];

	if(idx < 72)
		parCoordsSurfS[idx] = parCoordsSurf[idx];

	__syncthreads();

	if (idx < numSurfAct) {
		//shape function
		double N[8];
		double chsi;
		double eta;
		double zeta;
		int plane;

		//global location of integration points
		double xip;
		double yip;
		double zip;
		double r2;

		//Convection, Radiation, and Moving Flux
		double thetaIp;
		double qconv;
		double ambient4;
		double thetaIp4;
		double qrad;
		double qmov;


		const double conv1 = surfFlux[idx];
		const double conv2 = surfFlux[idx + numSurf];
		const double rad1 = surfFlux[idx + 2 * numSurf];
		const double rad2 = surfFlux[idx + 3 * numSurf];
		const double heat = surfFlux[idx + 4 * numSurf];
		const double rb2 = 1.21;
		const double Qin = 892.5;
		const double pi = 3.14159265358979;

		double Jac[4];
		int element;
		double coords[8];
		double detJac;


		for (int ip = 0; ip < 4; ++ip) {

			//compute shape function for surface

			//get surface plane
			plane = surfPlane[idx];

			//integration points
			chsi = parCoordsSurfS[ip * 18 + plane * 3];
			eta = parCoordsSurfS[ip * 18 + plane * 3 + 1];
			zeta = parCoordsSurfS[ip * 18 + plane * 3 + 2];

			//shape function in parametric coordinates
			N[0] = 0.125*(1.0 - chsi)*(1.0 - eta)*(1.0 - zeta);
			N[3] = 0.125*(1.0 - chsi)*(1.0 + eta)*(1.0 - zeta);
			N[2] = 0.125*(1.0 + chsi)*(1.0 + eta)*(1.0 - zeta);
			N[1] = 0.125*(1.0 + chsi)*(1.0 - eta)*(1.0 - zeta);
			N[4] = 0.125*(1.0 - chsi)*(1.0 - eta)*(1.0 + zeta);
			N[7] = 0.125*(1.0 - chsi)*(1.0 + eta)*(1.0 + zeta);
			N[6] = 0.125*(1.0 + chsi)*(1.0 + eta)*(1.0 + zeta);
			N[5] = 0.125*(1.0 + chsi)*(1.0 - eta)*(1.0 + zeta);

			xip = 0;
			yip = 0;
			zip = 0;

			//global location of integration points
			element = surfElemBirth[idx];
			for (int i = 0; i < 8; ++i) {
				xip += N[i] * eleNodeCoords[3 * numEl*i + element];
				yip += N[i] * eleNodeCoords[3 * numEl*i + numEl + element];
				zip += N[i] * eleNodeCoords[3 * numEl*i + 2 * numEl + element];

				coords[i] = 0;
			}

			//get mapped coordinates,

			//reuse Nodal variable to store GradN
			//2D shape functions
			chsi = parCoords2DS[ip * 2];
			eta = parCoords2DS[ip * 2 + 1];

			N[0] = 0.25 * (eta - 1.0);   /// remember this is gradN not N, need to save up on registers
			N[1] = 0.25 * (1.0 - eta);
			N[2] = 0.25 * (1.0 + eta);
			N[3] = -0.25 * (1.0 + eta);
			N[4] = 0.25 * (chsi - 1.0);
			N[5] = -0.25 * (1.0 + chsi);
			N[6] = 0.25 * (1.0 + chsi);
			N[7] = 0.25 * (1.0 - chsi);

			//Calculate Jacobian
			for (int i = 0; i < 4; ++i)
				Jac[i] = 0.0;

			for (int i = 0; i < 4; ++i) {
				coords[i * 2] = surfNodeCoords[2 * numSurf*i + idx];
				coords[i * 2 + 1] = surfNodeCoords[2 * numSurf*i + numSurf + idx];
			}

			for (int i = 0; i < 2; i++)
				for (int j = 0; j < 4; j++)
					for (int k = 0; k < 2; k++)
						Jac[i * 2 + k] += N[i * 4 + j] * coords[j * 2 + k];

			detJac = Jac[0] * Jac[3] - Jac[2] * Jac[1];

			// Use N again to calculate the 2D shape function for rhs

			N[0] = 0.25 * (1 - chsi) * (1 - eta);
			N[1] = 0.25 * (1 + chsi) * (1 - eta);
			N[2] = 0.25 * (1 + chsi) * (1 + eta);
			N[3] = 0.25 * (1 - chsi) * (1 + eta);

			thetaIp = 0.0;
			//Calculate
			for (int i = 0; i < 4; i++) {
				int ig = surfNodes[numSurf*i + idx];
				thetaIp += N[i] * thetaN[ig];
			}

			//convection
			qconv = -conv2 * (thetaIp - ambient);

			//radiation
			ambient4 = (ambient - abszero)*(ambient - abszero)*(ambient - abszero)*(ambient - abszero);
			thetaIp4 = (thetaIp - abszero)*(thetaIp - abszero)*(thetaIp - abszero)*(thetaIp - abszero);

			qrad = -sigma * rad2 * (thetaIp4 - ambient4);


			//moving flux from laser
			r2 = ((xip - tool0) * (xip - tool0) +
				(yip - tool1) * (yip - tool1) +
				(zip - tool2) * (zip - tool2));

			const double val = 3.0 * Qin / (pi * rb2) * exp(-3.0 * r2 / rb2);

			qmov = (laserState == 1) ? val : 0.0;

			for (int i = 0; i < 4; ++i) {
				int ig = surfNodes[numSurf*i + idx];
				int ir = surfIndx[idx*4 + i];
				rhs[ig + ir*nn] += N[i] * detJac * (conv1*qconv + rad1*qrad + heat*qmov);
			}
		}
	}
}

__global__ void massFlux(double* rhs, const int nn, int rhsCount, double* globalRhs) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < nn)
		for (int elInd = 0; elInd < rhsCount; elInd++)
			globalRhs[idx] += rhs[idx + nn*elInd];
}

__global__ void advanceTime(double* thetaN, const double* __restrict__ globalMass, const double* __restrict__ rhs, const int* __restrict__ birthNodes,double dt, int numActNodes) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < numActNodes) {
		int ig = birthNodes[idx];
		thetaN[ig] += dt * (rhs[ig])/globalMass[ig];
	}
}

__global__ void prescribeDirichlet(double* thetaN, const int* __restrict__ fixedNodes, const double* __restrict__ fixedNodeVals, int numFixed) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < numFixed) {
		int node = fixedNodes[idx];
		thetaN[node] = fixedNodeVals[idx];
	}
}
