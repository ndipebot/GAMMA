
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


__constant__ double parCoords[24];

__global__ void updateMass(const double* __restrict__ thetaN, const double* __restrict__ eleMat, const double* __restrict__ eleNodeCoords,
		double* globMass, const int* __restrict__ eleNodes, const int numEl, const int nn);

__global__ void initializeStiffness(const double* __restrict__ eleMat, const double* __restrict__ eleNodeCoords, double* eleStiffness,
		const int* __restrict__ eleNodes, const int numEl, const int nn);

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void createDataOnDeveice(DomainManager*& domainMgr, elementData& elemData, double initTemp) {
    //Start recording time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Start record
    cudaEventRecord(start, 0);


	int numEl = domainMgr->nel_;
	int nn = domainMgr->nn_;

	elemData.nn = nn;
	elemData.numEl = numEl;

	//Element data
	elemData.eleNodes.resize(8*numEl);
	elemData.eleNodeCoords.resize(24*numEl,0);
	elemData.eleMat.resize(6*numEl);
	elemData.eleBirthTimes.resize(numEl);
	elemData.thetaN.resize(nn);

	// set element values
	for(const auto & elem : domainMgr->elementList_) {
		 int * nodes = elem->nID_;
		 int eID = &elem - &domainMgr->elementList_[0]; // grab current element id


		 for(int i = 0; i < 8; ++i) {
			 //assign nodes to device vector in coalesced manner
			 elemData.eleNodes[numEl*i + eID] = nodes[i];
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

	//Device Vectors
	elemData.globMass.resize(8*nn,0);
	//thrust::fill(elemData.globMass.begin(), elemData.globMass.end(), 0);
	elemData.eleStiffness.resize(36*numEl,0);
	//thrust::fill(elemData.eleStiffness.begin(), elemData.eleStiffness.end(), 0);


	elemData.dEleStiffness = thrust::raw_pointer_cast(elemData.eleStiffness.data());
	elemData.dGlobMass = thrust::raw_pointer_cast(elemData.globMass.data());

	elemData.eleNodes_d = elemData.eleNodes;
	elemData.dEleNodes = thrust::raw_pointer_cast(elemData.eleNodes_d.data());


	elemData.eleNodeCoords_d = elemData.eleNodeCoords;
	elemData.dEleNodeCoords = thrust::raw_pointer_cast(elemData.eleNodeCoords_d.data());

	elemData.eleMat_d = elemData.eleMat;
	elemData.dEleMat = thrust::raw_pointer_cast(elemData.eleMat_d.data());

	elemData.eleBirthTimes_d = elemData.eleBirthTimes;
	elemData.dEleBirthTimes = thrust::raw_pointer_cast(elemData.eleBirthTimes_d.data());

	elemData.thetaN_d.resize(nn,initTemp);
	elemData.dthetaN = thrust::raw_pointer_cast(elemData.thetaN_d.data());

	//move parametric coordinates to local memory

	thrust::host_vector<float> coords(24);

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
	      coords[i*3 + j] = coeff[i*3 + j] * 0.5773502692;

      cudaMemcpyToSymbol(parCoords,coords.data(),24*sizeof(double));

      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      float elapsedTime;
      cudaEventElapsedTime(&elapsedTime, start, stop); // that's our time!
      // Clean up:
      cudaEventDestroy(start);
      cudaEventDestroy(stop);

      cout << "Device data setup took " << (double)elapsedTime/1000 << " seconds" << endl;

}

void initializeStiffnessOnD(elementData& elemData) {
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

	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop); // that's our time!
    // Clean up:
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cout << "initialize Stiffness took " << (double)elapsedTime/1000 << " seconds" << endl;
}

void updateMassOnD(elementData& elemData) {
	int gridSize = elemData.numEl/256 + 1;
	int blockSize = 256;

    //Start recording time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Start record
    cudaEventRecord(start, 0);

	updateMass<<<gridSize, blockSize>>>(elemData.dthetaN, elemData.dEleMat, elemData.dEleNodeCoords,
			elemData.dGlobMass, elemData.dEleNodes, elemData.numEl, elemData.nn);

	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop); // that's our time!
    // Clean up:
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cout << "update Mass took " << (double)elapsedTime/1000 << " seconds" << endl;

}

__global__ void updateMass(const double* __restrict__ thetaN, const double* __restrict__ eleMat, const double* __restrict__ eleNodeCoords,
		double* globMass, const int* __restrict__ eleNodes, const int numEl, const int nn) {

	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < numEl) {

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
			chsi = parCoords[ip*3 + 0];
			eta = parCoords[ip*3 + 1];
			zeta = parCoords[ip*3 + 2];
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
					globMass[i*nn + ig] += N[i] * rho * cpPoint * N[j] * detJac;
			}

		}
	}

}

__global__ void initializeStiffness(const double* __restrict__ eleMat, const double* __restrict__ eleNodeCoords,
		double* eleStiffness, const int* __restrict__ eleNodes, const int numEl, const int nn) {

	int idx = blockIdx.x*blockDim.x + threadIdx.x;

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

		#pragma unroll 1
		for(int ip = 0; ip < 8; ++ip) {

			for(int i = 0; i < 24; i++) {
				gradN[i] = 0;
				deriv[i] = 0;
			}

			for(int i = 0; i < 9; i++) {
				Jac[i] = 0;
				iJac[i] = 0;
			}

			//compute shape function
			chsi = parCoords[ip*3 + 0];
			eta = parCoords[ip*3 + 1];
			zeta = parCoords[ip*3 + 2];

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

			iJac[0] = (1/detJac) * Jac[4] * Jac[8] - Jac[5] * Jac[7];
			iJac[1] = (-1/detJac) * Jac[3] * Jac[8] - Jac[5] * Jac[6];
			iJac[2] = (1/detJac) * Jac[3] * Jac[7] - Jac[4] * Jac[6];

			iJac[3] = (-1/detJac) * Jac[1] * Jac[8] - Jac[2] * Jac[7];
			iJac[4] = (1/detJac) * Jac[0] * Jac[8] - Jac[2] * Jac[6];
			iJac[5] = (-1/detJac) * Jac[0] * Jac[7] - Jac[1] * Jac[6];

			iJac[6] = (1/detJac) * Jac[1] * Jac[5] - Jac[2] * Jac[4];
			iJac[7] = (-1/detJac) * Jac[0] * Jac[5] - Jac[2] * Jac[3];
			iJac[8] = (1/detJac) * Jac[0] * Jac[4] - Jac[1] * Jac[3];

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







