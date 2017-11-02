/*
 * Mesh.cpp
 *
 *  Created on: Oct 6, 2014
 *      Author: leonardo
 */
#include <vector>
#include <iterator>
#include <sstream>
#include <algorithm>
#include <string>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include "Mesh.h"
#include <iomanip>



Mesh::Mesh(string fileName) {
	//Contstructor
	file.open(fileName.c_str());
}

Mesh::~Mesh() {
	// Destructor
	file.close();
}


void Mesh::openNew(string fileName) {
	file.close();
	file.open(fileName.c_str());
}

/*
 * Get inputs
 */
void Mesh::getDomainInfo()
{
  string line;
  int option = 0;
  int setID, PID;
  bool Init;

  if(file.is_open()) {
    while(getline(file, line)) {
      if(line.at(0) == '$')
        continue;
      if(line == "*NODE")
      {
        option = 0;
        continue;
      }
      else if(line == "*ELEMENT_SOLID")
      {
	option = 1;
	continue;
      }
      else if(line == "*SET_NODE_LIST")
      {
        Init = true;
	option = 2;
	continue;
      }
      else if(line == "*LOAD_NODE_SET")
      {
	option = 3;
	continue;
      }
      else if(line == "*MAT_THERMAL_ISOTROPIC")
      {
        Init = true;
	option = 4;
	continue;
      }
      else if(line == "*CONTROL_TIMESTEP")
      {
	option = 5;
	continue;
      }
      else if(line == "*DEFINE_CURVE")
      {
	option = 6;
	continue;
      }
      else if(line == "*INITIAL_TEMPERATURE_SET")
      {
	option = 7;
	continue;
      }
      else if(line == "*CONTROL_TERMINATION")
      {
	option = 8;
	continue;
      }
      else if(line == "*DATABASE_NODOUT")
      {
	option = 9;
	continue;
      }
      else if(line == "*KEYWORD_ID")
      {
	option = 10;
	continue;
      }
      else if(line == "*PARAMETER")
      {
	option = 11;
	continue;
      }
      else if(line == "*TOOL_FILE")
      {
        isLENS_ = true;
	option = 12;
	continue;
      }
      else if(line == "*PROBE")
      {
	option = 13;
	continue;
      }
      else if(line == "*ENERGY_OUT")
      {
	option = 14;
	continue;
      }
      else if(line == "*MAT_THERMAL_USER_DEFINED")
      {
        Init = true;
	option = 15;
	continue;
      }
      else if(line == "*GAUSS_LASER")
      {
	option = 16;
	continue;
      }
      else if(line.at(0) == '*') 
      {
	option = 1000;
	continue;
      }

      if(option==0) {
	istringstream lines(line);
	vector<double> coords((istream_iterator<double>(lines)), istream_iterator<double>());
	int key = (int) coords[0];
	vector<double> value(coords.begin()+1, coords.begin()+4);
	NODES_[key] = value;

	continue;
      }// option 0
      else if (option==1) {
        istringstream lines(line);
	vector<int> coords((istream_iterator<double>(lines)), istream_iterator<double>());
	int ele = coords[0];
	int pid = coords[1];
	vector<int> conn(coords.begin()+2, coords.end());

	if(PIDS_.count(pid))
	  PIDS_[pid].push_back(ele);
	else {
	  vector<int> elements;
	  elements.push_back(ele);
	  PIDS_[pid] = elements;
        }
	ELEM_[ele] = conn;
	continue;
      }// option 1
      else if (option == 2)
      {
	  istringstream lines(line);
	  vector<int> coords((istream_iterator<double>(lines)), istream_iterator<double>());
          if (coords.size() == 1 && Init)
          {
            Init = false;
            setID = (int)coords[0];
          }
          else
          {
	    vector<int> nodeRow(coords.begin(), coords.end());
            for (int ii = 0; ii < nodeRow.size(); ii++)
            {
              nodeSets_[setID].push_back(nodeRow[ii]);
            }
          }
      }// option 2
      else if (option == 3)
      {
	istringstream lines(line);
	vector<double> coords((istream_iterator<double>(lines)), istream_iterator<double>());
        int setID = (int)coords[0];
        int bcID = (int)coords[1];
        double bcVal = (double)coords[2];
        loadSets_[setID].push_back(bcID);
        loadSetVals_[setID].push_back(bcVal);
      }// option 3
      else if (option == 4)
      {
	istringstream lines(line);
	vector<double> coords((istream_iterator<double>(lines)), istream_iterator<double>());
        if(Init)
        {
          Init = false;
          PID = (int)coords[0];
          PID_to_MAT_Type_[PID] = 1; // 1 = isotropic materials
	  for (int ii = 1; ii < coords.size(); ii++)
	  {
            double mat = (double)coords[ii];
	    PID_to_MAT_[PID].push_back(mat);
	  }//end for(ii)
        }
        else
        {
	  for (int ii = 0; ii < coords.size(); ii++)
	  {
	    PID_to_MAT_[PID].push_back(coords[ii]);
	  }//end for(ii)
        }
      }// option 4
      else if (option == 5)
      {
	istringstream lines(line);
	vector<double> coords((istream_iterator<double>(lines)), istream_iterator<double>());
        inputDt_ = coords[0];
      }// option 5
      else if (option == 6)
      {
	istringstream lines(line);
	vector<double> coords((istream_iterator<double>(lines)), istream_iterator<double>());
        double time = coords[0];
        int elemID = (int)coords[1];
        if (coords.size() == 2)
        {
	  birthID_.push_back(elemID);
   	  birthTime_.push_back(time);
        }
      }// option 6
      else if (option == 7)
      {
	istringstream lines(line);
	vector<double> coords((istream_iterator<double>(lines)), istream_iterator<double>());
        int setID = (int)coords[0];
        double initTemp = (double)coords[1];
        initialCondition_[setID] = initTemp;
      }// option 7
      else if (option == 8)
      {
	istringstream lines(line);
	vector<double> coords((istream_iterator<double>(lines)), istream_iterator<double>());
        finalTime_ = (double)coords[0];
      }// option 8
      else if (option == 9)
      {
	istringstream lines(line);
	vector<double> coords((istream_iterator<double>(lines)), istream_iterator<double>());
        outTime_ = (double)coords[0];
      }// option 9
      else if (option == 10)
      {
	istringstream lines(line);
	vector<string> coords((istream_iterator<string>(lines)), istream_iterator<string>());
        outFileName_ = coords[0];
      }// option 10
      else if (option == 11)
      {
	istringstream lines(line);
	vector<string> coords((istream_iterator<string>(lines)), istream_iterator<string>());
        string paramName = coords[0]; 
        double value = stod(coords[1]);
        paramValues_[paramName] = value;
      }// option 11
      else if (option == 12)
      {
	istringstream lines(line);
	vector<string> coords((istream_iterator<string>(lines)), istream_iterator<string>());
        toolFileName_ = coords[0]; 
      }// option 12
      else if (option == 13)
      {
	istringstream lines(line);
	vector<string> coords((istream_iterator<string>(lines)), istream_iterator<string>());
	double *probeCoords = new double[3];
        probeNames_.push_back( coords[0] + probeExtension_ ); 
        probeCoords[0] = stod(coords[1]);
        probeCoords[1] = stod(coords[2]);
        probeCoords[2] = stod(coords[3]);
        probePos_.push_back(probeCoords);
        
      }// option 13

      else if (option == 14)
      {
	istringstream lines(line);
	vector<string> coords((istream_iterator<string>(lines)), istream_iterator<string>());
        energyFileName_ = coords[0] + probeExtension_;
        calcEnergy_ = true;
      }// option 14
      else if (option == 15)	// *MAT_THERMAL_USER_DEFINED
      {
	istringstream lines(line);
	vector<double> coords((istream_iterator<double>(lines)), istream_iterator<double>());
        if(Init)
        {
          Init = false;
          PID = (int)coords[0];
          PID_to_MAT_Type_[PID] = 2; // 1 = user-defined materials
	  for (int ii = 1; ii < coords.size(); ii++)
	  {
            double mat = (double)coords[ii];
	    PID_to_MAT_[PID].push_back(mat);
	  }//end for(ii)
        }
      }// option 15
      else if (option == 16)	// *GAUSS_LASER
      {
	istringstream lines(line);
	vector<double> coords((istream_iterator<double>(lines)), istream_iterator<double>());
        Qin_ = (double)coords[0];
        rBeam_ = (double)coords[1];
        Qeff_ = (double)coords[2];
      }// option 16

    } //end option conditional loops
  } 
}


string Mesh::getParam() {
	string line;
	while(true){
		getline(file, line);
		if(line.at(0) == '$')
			continue;
		else
			break;
	}
	return line;
}

//////////////////////////////////////////////////////
//		assignParameters		    //
//////////////////////////////////////////////////////
void 
Mesh::assignParameters ()
{
  // Assign parameter values
  for (map<string,double>::iterator it = paramValues_.begin();
       it != paramValues_.end(); it++)
  {
    string paramName = it->first;
    double value = it->second;
    if (paramName == "Rambient")
    {
      Rambient_ = value;
    }
    else if (paramName == "Rboltz")
    {
      Rboltz_ = value;
    }
    else if (paramName == "Rabszero")
    {
      Rabszero_ = value;
    }

  }//end for(it)
}//assignParamters

