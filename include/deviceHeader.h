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

	vector<int> eleNodes;
	vector<double> eleNodeCoords;
	vector<double> eleMat;
	vector<double> eleBirthTimes;
	vector<double> thetaN;
	vector<double> eleStiffness;
	vector<double> globMass;
};

void AllocateDeviceData(elementData& elem);

void CopyToDevice(elementData& elem);

void FreeDevice(elementData& elem);

void CopyToHost(elementData& elem);

void createDataOnDeveice(DomainManager*& domainMgr, elementData& elemData, double initTemp);

void initializeStiffnessOnD(elementData& elemData);

void updateMassOnD(elementData& elemData);

void udpateMatK();

void updateCapK();

void getInternalForceK();


#endif /* DEVICEHEADER_H_ */
