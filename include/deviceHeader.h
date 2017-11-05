/*
 * deviceHeader.h
 *
 *  Created on: Oct 3, 2017
 *      Author: leonardo
 */

#ifndef DEVICEHEADER_H_
#define DEVICEHEADER_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <DomainManager.h>
#include <HeatSolverManager.h>



struct elementData {
	int* dEleNodes ;
	float* dEleStiffness;
	float* dGlobMass;
	float* dEleNodeCoords;
	float* dEleMat;
	float* dEleBirthTimes;
	float* dthetaN;
	int* dNUniId;
	float* dGlobRHS;
	float* dGlobRHS_Surf;
	int* dSurfIndx;
	int* dFixedNodes;
	float* dFixedNodeVals;
	int* dBirthNodes;


	float dt;
	int numFixed;

	int* dSurfNodes;
	float* dSurfNodeCoords;
	int* dSurfPlane;
	float* dSurfFlux;
	int* dSurfBirthElem;
	float* dSurfDeathTime;

	int* fixedNodes;
	vector<float> fixedValues;

	int numEl;
	int nn;
	float initTemp;
	int numSurf;
	float tool0;
	float tool1;
	float tool2;
	int laserState;
	int rhsCount;
	float ambient;
	float abszero;
	float sigma;
	int rhsCountEle;
	float rBeam;
	float Qin;

	vector<int> eleNodes;
	vector<float> eleNodeCoords;
	vector<float> eleMat;
	vector<float> eleBirthTimes;
	vector<float> thetaN;
	vector<float> eleStiffness;
	vector<float> globMass;
	vector<float> globRHS;
	vector<float> globRHS_Surf;
	vector<int> nUniId;
	vector<int>	surfIndx;

	vector<float> boundSurfBirthTime;
	vector<float> boundSurfDeathTime;
	vector<int> surfNodes;
	vector<float> surfNodeCoords;
	vector<int> surfPlane;
	vector<float> surfFlux;
	vector<int> surfBirthElem;

	vector<int> birthNodes;
	vector<float> birthNodeTimes;

};

void AllocateDeviceData(elementData& elem);

void CopyToDevice(elementData& elem);

void FreeDevice(elementData& elem);

void CopyToHost(elementData& elem);

void createDataOnDevice(DomainManager*& domainMgr, elementData& elemData, HeatSolverManager*& heatMgr);

void initializeStiffnessOnD(elementData& elemData);

void updateMassOnD(elementData& elemData, DomainManager*& domainMgr);

void updateIntForceOnD(elementData& elemData, DomainManager*& domainMgr);

void updateFluxKernel(elementData& elemData, DomainManager*& domainMgr);

void compareMass(elementData& elemData, vector<float>& Mvec);

void compareStiff(elementData& elemData, vector<Element*>& elementList);

void compareIntForce(elementData& elemData, vector<float>& rhs);

void compareFlux(elementData& elemData, vector<float>& rhs);

void dirichletBCKernel(elementData& elemData);

void advanceTimeKernel(elementData& elemData, DomainManager*& domainMgr);

void compareTemp(elementData& elemData, vector<float>& thetaN );

void clearDeviceData(elementData& elem);


#endif /* DEVICEHEADER_H_ */
