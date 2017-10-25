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

	vector<int> eleNodes;
	vector<double> eleNodeCoords;
	vector<double> eleMat;
	vector<double> eleBirthTimes;
	vector<double> thetaN;
	vector<double> eleStiffness;
	vector<double> globMass;

	vector<double> boundSurfBirthTime;
	vector<double> boundSurfDeathTime;
	vector<int> surfNodes;
	vector<double> surfNodeCoords;
	vector<int> surfPlane;
	vector<double> surfFlux;
	vector<int> surfBirthElem;

	int* dSurfNodes;
	double* dSurfNodeCoords;
	int* dSurfPlane;
	double* dSurfFlux;
	int* dSurfBirthElem;

	double* rhs;

};

void AllocateDeviceData(elementData& elem);

void CopyToDevice(elementData& elem);

void FreeDevice(elementData& elem);

void CopyToHost(elementData& elem);

void createDataOnDevice(DomainManager*& domainMgr, elementData& elemData, HeatSolverManager*& heatMgr);

void initializeStiffnessOnD(elementData& elemData);

void updateMassOnD(elementData& elemData, DomainManager*& domainMgr);

void compareMass(elementData& elemData, vector<double> Mvec);

void compareStiff(elementData& elemData, vector<Element*> elementList);


#endif /* DEVICEHEADER_H_ */
