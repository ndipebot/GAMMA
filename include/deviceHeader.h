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
	double* dEleStiffness;
	double* dGlobMass;
	double* dEleNodeCoords;
	double* dEleMat;
	double* dEleBirthTimes;
	double* dthetaN;
	int* dNUniId;
	double* dGlobRHS;
	double* dGlobRHS_Surf;
	int* dSurfIndx;
	int* dFixedNodes;
	double* dFixedNodeVals;
	int* dBirthNodes;


	double dt;
	int numFixed;

	int* dSurfNodes;
	double* dSurfNodeCoords;
	int* dSurfPlane;
	double* dSurfFlux;
	int* dSurfBirthElem;
	int* fixedNodes;
	double* fixedValues;

	int numEl;
	int nn;
	double initTemp;
	int numSurf;
	double tool0;
	double tool1;
	double tool2;
	int laserState;
	int rhsCount;
	double ambient;
	double abszero;
	double sigma;
	int rhsCountEle;

	vector<int> eleNodes;
	vector<double> eleNodeCoords;
	vector<double> eleMat;
	vector<double> eleBirthTimes;
	vector<double> thetaN;
	vector<double> eleStiffness;
	vector<double> globMass;
	vector<double> globRHS;
	vector<double> globRHS_Surf;
	vector<int> nUniId;
	vector<int>	surfIndx;

	vector<double> boundSurfBirthTime;
	vector<double> boundSurfDeathTime;
	vector<int> surfNodes;
	vector<double> surfNodeCoords;
	vector<int> surfPlane;
	vector<double> surfFlux;
	vector<int> surfBirthElem;

	vector<int> birthNodes;
	vector<double> birthNodeTimes;

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

void compareMass(elementData& elemData, vector<double> Mvec);

void compareStiff(elementData& elemData, vector<Element*> elementList);

void compareIntForce(elementData& elemData, vector<double> rhs);

void compareFlux(elementData& elemData, vector<double> rhs);

void dirichletBCKernel(elementData& elemData);

void advanceTimeKernel(elementData& elemData, DomainManager*& domainMgr);

void compareTemp(elementData& elemData, vector<double>& thetaN );

void clearDeviceData(elementData& elem);


#endif /* DEVICEHEADER_H_ */
