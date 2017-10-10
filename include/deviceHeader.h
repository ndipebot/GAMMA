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

void createDataOnDeveice(HeatSolverManager *heatMgr_, DomainManager *domainMgr_);

void udpateMatK();

void updateCapK();

void getInternalForceK();



#endif /* DEVICEHEADER_H_ */
