#include <math.h>
#include <map>
#include <stdlib.h>
#include "Element.h"
#include "Surface.h"

void writeVTU_File (vector <Element*> elementList, vector <double> coordList,
                    vector <int> &activeElements, vector <double> thetaNp1, string FileOut);

void writeVTU_pvtu (vector< string> octreeFileNames, string pvtuFileName);

void writeVTU_coordinates (vector<double> coordList, ofstream &outputFile, int nn);

void writeVTU_coordsHeader (ofstream &outputFile);

void writeVTU_coordsEnd (ofstream &outputFile);

void writeVTU_connectivity (vector<Element*> elementList, vector<int> &activeElements,
                            ofstream &outputFile);

void writeVTU_offsets (int nCells, ofstream &outputFile, int offset);

void writeVTU_cellTypes (int nCells, ofstream &outputFile);

void writeVTU_pointData (vector<double> pointVec, ofstream &outputFile);

void writeVTU_connectHeader(ofstream &outputFile);

void writeVTU_connectEnd(ofstream &outputFile);

void writeVTU_cellHeader(ofstream &outputFile);

void writeVTU_cellEnd(ofstream &outputFile);

void writeVTU_offsetHeader(ofstream &outputFile);

void writeVTU_offsetEnd(ofstream &outputFile);

void writeVTU_cellTypeHeader(ofstream &outputFile);

void writeVTU_cellTypeEnd(ofstream &outputFile);

void writeVTU_header (int nCells, int nPoints, ofstream &outputFile);

void writeVTU_end (ofstream &outputFile);

void writeVTU_pvtuHeader (ofstream &outputFile);

void writeVTU_pvtuMeshDataType (ofstream &outputFile);

void writeVTU_pvtuSources (vector<string> octreeFileNames, ofstream &outputFile);

void writeVTU_pvtuEnd (ofstream &outputFile);

void writeVTU_cellDataHeader(ofstream &outputFile);

void writeVTU_cellDataEnd(ofstream &outputFile);

void writeVTU_cellOutHeader(ofstream &outputFile);

void writeVTU_cellOutEnd(ofstream &outputFile);

void writeVTU_pointDataHeader(ofstream &outputFile);

void writeVTU_pointDataEnd(ofstream &outputFile);

void writeVTU_pointOutHeader(ofstream &outputFile);

void writeVTU_pointOutEnd(ofstream &outputFile);
