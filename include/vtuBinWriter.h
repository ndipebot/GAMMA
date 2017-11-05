#ifndef vtuBinWriter_h
#define vtuBinWriter_h

#include <vector>
#include <math.h>
#include <map>
#include <stdlib.h>
#include <string>
#include "Element.h"
#include "Surface.h"
#include <DomainManager.h>
#include <HeatSolverManager.h>
#include <deviceHeader.h>

class DomainManager;
class HeatSolverManger;

class vtuBinWriter
{
public:
  vtuBinWriter(DomainManager *domainMgr, 
               HeatSolverManager *heatMgr,
	           elementData& elem,
               string &FileOut);

  virtual ~vtuBinWriter() {}

  // member variables
  DomainManager *domainMgr_;
  HeatSolverManager *heatMgr_;
  elementData *elem_;
  string &FileOut_;
  ofstream outputFile_;
  int offSetCtr_, nCells_, nPoints_, byteCtr_;

  // methods
  void execute();

  void writeVTU_pvtu (vector< string> octreeFileNames, string pvtuFileName);

  void writeVTU_coordinates ();

  void writeVTU_coordsHeader ();

  void writeVTU_coordsEnd ();

  void writeVTU_connectivity ();

  void writeVTU_offsets (int offset);

  void writeVTU_cellTypes ();

  void writeVTU_pointData (vector<float> &pointVec );

  void writeVTU_connectHeader();

  void writeVTU_connectEnd();

  void writeVTU_cellHeader();

  void writeVTU_cellEnd();

  void writeVTU_offsetHeader();

  void writeVTU_offsetEnd();

  void writeVTU_cellTypeHeader( );

  void writeVTU_cellTypeEnd( );

  void writeVTU_header ();

  void writeVTU_end ( );

  void writeVTU_pointDataHeader( );

  void writeVTU_pointDataEnd( );

  void writeVTU_pointOutHeader( );

  void writeVTU_pointOutEnd( );

  void writeVTU_appendHeader();

  void writeVTU_appendEnd();

  void writeVTU_cellDataHeader();
  
  void writeVTU_cellOutHeader();
  
  void writeVTU_cellDataEnd();

  void writeVTU_cellData();

};

#endif
