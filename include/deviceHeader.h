/*
 * deviceHeader.h
 *
 *  Created on: Oct 3, 2017
 *      Author: leonardo
 */

#ifndef DEVICEHEADER_H_
#define DEVICEHEADER_H_

#include <DomainManager.h>
#include <HeatSolverManager.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

struct elementData {
	int* dEleNodes ;
	double* dEleStiffness;
	double* dGlobMass;
	double* dEleNodeCoords;
	double* dEleMat;
	double* dEleBirthTimes;
	double* dthetaN;
	int numEl;
	int nn;

	thrust::host_vector<int> eleNodes;
	thrust::host_vector<double> eleNodeCoords;
	thrust::host_vector<double> eleMat;
	thrust::host_vector<double> eleBirthTimes;
	thrust::host_vector<double> thetaN;

	thrust::device_vector<int> eleNodes_d;
	thrust::device_vector<double> eleNodeCoords_d;
	thrust::device_vector<double> eleMat_d;
	thrust::device_vector<double> eleBirthTimes_d;
	thrust::device_vector<double> thetaN_d;

	thrust::device_vector<double> eleStiffness;
	thrust::device_vector<double> globMass;
};

void createDataOnDeveice(DomainManager*& domainMgr, elementData& elemData, double initTemp);

void initializeStiffnessOnD(elementData& elemData);

void updateMassOnD(elementData& elemData);

void udpateMatK();

void updateCapK();

void getInternalForceK();


#endif /* DEVICEHEADER_H_ */
