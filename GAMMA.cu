#include <vector>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <math.h>
#include <iomanip>
#include <stdlib.h>
#include "time.h"
#include <algorithm>
#include <iterator>
#include <fstream>
#include <numeric>

// user defined headers
#include <Mesh.h>
#include <Element.h>
#include <dynaInput.h>
#include <Surface.h>
#include <BCManager.h>
#include <HeatSolverManager.h>
#include <vtuWriter.h>
#include <FluxManager.h>
#include <ConvManager.h>
#include <RadManager.h>
#include <DomainManager.h>
#include <MaterialManager.h>
#include <ThermalIsoManager.h>
#include <vtuBinWriter.h>
#include <deviceHeader.h>

using namespace std;

int main(int arg, char *argv[])
{

  if (arg != 2)
  {
    cout << "Missing parameter: Mesh file name" << endl;
    exit(1);
    return 0;
  }

  //for clocking 
  double starttime, endtime, preprocessTimer;
  int numSteps, numOut, numOutSteps;
  string meshName = argv[1];
  Mesh meshObj(meshName);

  // Vtk indentation rule
  const std::string SecondEleWidth_ = "  ";
  const std::string ThirdEleWidth_  = "    ";
  const std::string FourthEleWidth_ = "      ";
  const std::string FifthEleWidth_  = "        ";
  const std::string SixthEleWidth_  = "          ";
  preprocessTimer =0.0;

  // get mesh 
  meshObj.getDomainInfo();

  // get user parameters
  meshObj.assignParameters();

  // Title output
  cout << "|==================================================================|\n";
  cout << "|       Northwestern University Finite Element Software            |\n";
  cout << "|                      G.A.M.M.A Software                          |\n";
  cout << "| Generalized Analysis of Multiscale and Multiphysics Applications |\n";
  cout << "|                                                                  |\n";
  cout << "|                        Developers:                               |\n";
  cout << "|                        Stephen Lin                               |\n";
  cout << "|                        Jacob Smith                               |\n";
  cout << "|                      Kevontrez Jones                             |\n";
  cout << "|==================================================================|\n";
  cout << "\n\n";


  cout << "===============================================================\n";
  cout << "\tUser Input Information\n\n";
  cout << SecondEleWidth_ << "Reading in Mesh File: " << argv[1] << endl;
  cout << SecondEleWidth_ << "Writing out to output database name: " << meshObj.outFileName_ << endl;
  cout << SecondEleWidth_ << "Number of probe points selected: " << meshObj.probeNames_.size() << endl;
  if (meshObj.isLENS_)
  {
    cout << SecondEleWidth_ << "Toolpath file name: " << meshObj.toolFileName_ << endl;
  }
  if (meshObj.calcEnergy_)
  {  
    cout << SecondEleWidth_ << "Energy evaluator file name: " << meshObj.energyFileName_ << endl;
  }
  cout << SecondEleWidth_ << "Input parameter list: " << endl;
  for (map<string,double>::iterator it = meshObj.paramValues_.begin();
       it != meshObj.paramValues_.end(); it++)
  {
    string paramName = it->first;
    double value = it->second;
    cout << ThirdEleWidth_ << paramName << "  \t:  " << value << endl;
  }//end for(it)
  cout << "\n\n";
  
  // Set up domain (connectivity etc.)
  DomainManager *domainMgr = new DomainManager(&meshObj);
  domainMgr->initializeDomain();

  cout <<  "\n";
  cout <<  "\n";

  ///////////////////////////////////////////////////////
  // 		Begin simulation setup		       //
  ///////////////////////////////////////////////////////
  cout <<  "===============================================================\n";  
  cout <<  "TIMING FOR SIMULATION SETUP\n\n";
  preprocessTimer = 0.0;

  // Create solver managers here
  starttime = clock();
  HeatSolverManager *heatMgr = new HeatSolverManager(domainMgr, &meshObj); 
  
  heatMgr->initializeSystem();
  endtime = clock();
  preprocessTimer += (double) (endtime - starttime) / CLOCKS_PER_SEC;  
  cout <<  "\nTOTAL TIME FOR SIMULATION SETUP\n";
  cout << SecondEleWidth_ << preprocessTimer << endl;
  cout <<  "===============================================================\n";  


  // Calculate outputs/approximate # of timesteps
  numSteps = (int)ceil(meshObj.finalTime_/heatMgr->dt_);
  numOut = (int)ceil(meshObj.finalTime_/meshObj.outTime_);
  numOutSteps = (int)ceil(meshObj.outTime_/heatMgr->dt_);
  
  cout << "\n\n";
  cout <<  "===============================================================\n";
  cout <<  "Mesh and Timestep Statistics:\n";
  cout << "\n";
  cout << SecondEleWidth_ << "Number of Elements: " << domainMgr->nel_ << endl;
  cout << SecondEleWidth_ << "Number of Nodes: " << domainMgr->nn_ << endl;
//  cout << SecondEleWidth_ << "Number of Surfaces: " << surfaceList_.size() + staticSurfList_.size() <<endl;
  cout << SecondEleWidth_ << "Calculated minimum time step size: " << heatMgr->dt_ << endl;
  cout << SecondEleWidth_ << "Approximate # of timesteps: " << numSteps << endl;
  cout << SecondEleWidth_ << "Approximate # of outputs: " << numOut << endl;
  cout << SecondEleWidth_ << "Approximate # of steps between outputs: " << numOutSteps << endl;
  cout << "\n";
  cout <<  "End Mesh and Timestep Statistics:\n";
  cout <<  "===============================================================\n";
  cout << "\n\n";

  ///////////////////////////////////////////////////////
  // 		Time stepping 			       //
  ///////////////////////////////////////////////////////
  double dumpTime = 0.0;
  double simStart = clock();
  domainMgr->currTime_ = 0.0;
  double outTrack = 0.0;
  double simTime = meshObj.finalTime_;
  int outCt = 0;
  string extName = ".vtu";
  string outFile = meshObj.outFileName_ + to_string(outCt) + extName;
  
  // set up output manager
  vtuBinWriter * vtuMgr = new vtuBinWriter(domainMgr, heatMgr, outFile);
  vtuMgr->execute();

  //Copy element Data to GPU
  elementData elemData;
  createDataOnDevice(domainMgr, elemData, heatMgr);


  // time integrator
  while (domainMgr->currTime_ <= simTime)
  {
    // update time counters
    outTrack += heatMgr->dt_;

    //domainMgr->currTime_ += heatMgr->dt_;
	domainMgr->currTime_ = 0.0;

    heatMgr->pre_work();
 
    heatMgr->updateCap();

	heatMgr->heatBCManager_->applyFluxes();

	////////////////////////////////////

	//initializeStiffnessOnD(elemData);
	//updateMassOnD(elemData, domainMgr);
	//updateIntForceOnD(elemData, domainMgr);
	updateFluxKernel(elemData, domainMgr);

	CopyToHost(elemData);

	//compareMass(elemData, heatMgr->Mvec_);
	//compareStiff(elemData, domainMgr->elementList_);
	//compareIntForce(elemData, heatMgr->rhs_);
	compareFlux(elemData, heatMgr->rhs_);

	FreeDevice(elemData);

	/////////////////////////////////////

    heatMgr->integrateForce();

    heatMgr->advance_time_step();
  
    heatMgr->post_work();

    // File Manager
    if (outTrack >= meshObj.outTime_)
    {
      outCt++;
      outFile = meshObj.outFileName_ + to_string(outCt) + extName;
      starttime = clock();
      vtuMgr->execute();
      endtime = clock();
      outTrack = 0.0;
      cout << "===============================================================\n";
      cout << left << 
              "   Output at time: " << setw(43) << domainMgr->currTime_ 
                                    << "|" << endl;
      cout << left << 
              "   Percentage done: " << setw(42) << (domainMgr->currTime_/simTime) * 100.0 
                                     << "|" << endl;
      cout << left << 
              "   Timer for outputting files (ascii) " << setw(24) 
	   << (double) (endtime - starttime) / CLOCKS_PER_SEC 
                                     << "|" << endl;
      dumpTime += (double) (endtime - starttime) / CLOCKS_PER_SEC;
      cout << "===============================================================\n";
    }

    // no longer beginning of simulation
    if (domainMgr->isInit_)
    {
      domainMgr->isInit_ = false;
    }
  }//end for(t)
  outCt++;
  vtuMgr->execute();
  double simEnd  = clock();

  cout << "\n\n";
  cout <<  "===============================================================\n";
  cout <<  "Simulation Finished: cpu time statistics:\n";
  cout << "\n";
  cout << SecondEleWidth_ << "Final simulation time: " << domainMgr->currTime_ << endl;
  cout << SecondEleWidth_ << "Total compute time: "
       << (double) (simEnd - simStart) / CLOCKS_PER_SEC << endl;
  cout << SecondEleWidth_ << "Total time for dumping data: "
       << dumpTime << endl;
  cout << SecondEleWidth_ << "Total time for probing data: "
       << heatMgr->probeTime_ << endl;
  cout << "\n";
  cout <<  "End cpu time statistics:\n";
  cout <<  "===============================================================\n";
}
