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

#include "vtuWriter.h"
#include "Element.h"
#include "Surface.h"

///////////////////////////////////////////////////////////////////////
//			writeVTU_File				    //
//////////////////////////////////////////////////////////////////////
void 
writeVTU_File (vector <Element*> elementList, vector<double> coordList,
               vector <int> &activeElements, vector <double> thetaNp1, string FileOut)
{
  ofstream outputFile(FileOut);

  int nCells = activeElements.size();
  int nPoints = thetaNp1.size();
  writeVTU_header(nCells, nPoints, outputFile);

  writeVTU_coordsHeader(outputFile);

  writeVTU_coordinates(coordList, outputFile, nPoints);

  writeVTU_coordsEnd(outputFile);

  writeVTU_pointDataHeader(outputFile);
  
  writeVTU_pointOutHeader(outputFile);

  writeVTU_pointData(thetaNp1, outputFile);
  
  writeVTU_pointOutEnd(outputFile);

  writeVTU_pointDataEnd(outputFile);

  writeVTU_cellHeader(outputFile);

  writeVTU_connectHeader(outputFile);

  writeVTU_connectivity(elementList, activeElements, outputFile );

  writeVTU_connectEnd(outputFile);

  writeVTU_offsetHeader(outputFile);

  writeVTU_offsets (nCells, outputFile, 0);

  writeVTU_offsetEnd(outputFile);

  writeVTU_cellTypeHeader(outputFile);

  writeVTU_cellTypes(nCells, outputFile);

  writeVTU_cellTypeEnd(outputFile);

  writeVTU_cellEnd(outputFile);

  writeVTU_end (outputFile);
  
  outputFile.close();
}//end writeVTU_File

//////////////////////////////////////////////////////////////////////
//			writeVTU_pvtu   			    //
//////////////////////////////////////////////////////////////////////
void
writeVTU_pvtu(vector<string> octreeFileNames, string pvtuFileName)
{
  ofstream outputFile(pvtuFileName);
 
  writeVTU_pvtuHeader(outputFile);

  writeVTU_pvtuMeshDataType(outputFile);

  writeVTU_pvtuSources(octreeFileNames,outputFile);

  writeVTU_pvtuEnd(outputFile);

  outputFile.close();

}//end writeVTU_pvtu


//////////////////////////////////////////////////////////////////////
//			writeVTU_coordinates			    //
//////////////////////////////////////////////////////////////////////
void 
writeVTU_coordinates (vector<double> coordList, ofstream &outputFile, int nn)
{

  string SecondEleWidth_   = "  ";
  string ThirdEleWidth_    = "    ";
  string FourthEleWidth_   = "      ";
  string FifthEleWidth_    = "        ";
  string SixthEleWidth_    = "          ";

  for (int ii = 0; ii < nn; ii++)
  {
    double xcurr = coordList[ii*3 + 0];
    double ycurr = coordList[ii*3 + 1];
    double zcurr = coordList[ii*3 + 2];
    outputFile << SixthEleWidth_ << xcurr << " " << ycurr << " " << zcurr;
    outputFile << "\n";
  }//end for(ii)

}//end writeVTU_coordinates

//////////////////////////////////////////////////////////////////////
//			writeVTU_coordsHeader			    //
//////////////////////////////////////////////////////////////////////
void 
writeVTU_coordsHeader (ofstream &outputFile)
{
  string SecondEleWidth_   = "  ";
  string ThirdEleWidth_    = "    ";
  string FourthEleWidth_   = "      ";
  string FifthEleWidth_    = "        ";
  string SixthEleWidth_    = "          ";

  outputFile << FourthEleWidth_ <<"<Points>\n";

  outputFile << FifthEleWidth_ <<"<DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
}//end writeVTU_coordsHeader

//////////////////////////////////////////////////////////////////////
//			writeVTU_coordsEnd			    //
//////////////////////////////////////////////////////////////////////
void 
writeVTU_coordsEnd (ofstream &outputFile)
{
  string SecondEleWidth_   = "  ";
  string ThirdEleWidth_    = "    ";
  string FourthEleWidth_   = "      ";
  string FifthEleWidth_    = "        ";
  string SixthEleWidth_    = "          ";

  outputFile << FifthEleWidth_ <<"</DataArray>\n";
  outputFile << FourthEleWidth_ << "</Points>\n\n";

}//end writeVTU_coordsEnd

//////////////////////////////////////////////////////////////////////
//			writeVTU_connectivity			    //
//////////////////////////////////////////////////////////////////////
void 
writeVTU_connectivity (vector<Element*> elementList, vector<int> &activeElements,
                       ofstream &outputFile)
{
  string SecondEleWidth_ = "  ";
  string ThirdEleWidth_  = "    ";
  string FourthEleWidth_ = "      ";
  string FifthEleWidth_  = "        ";
  string SixthEleWidth_  = "          ";

  //write out connectivity data
  int n1, n2, n3, n4, n5, n6, n7, n8;
  for (int ie = 0; ie < activeElements.size(); ie++)
  {
    int eID = activeElements[ie];
    Element * element = elementList[eID];
    int *localNodes = element->nID_;

    n1 = localNodes[0];
    n2 = localNodes[1];
    n3 = localNodes[2];
    n4 = localNodes[3];
    n5 = localNodes[4];
    n6 = localNodes[5];
    n7 = localNodes[6];
    n8 = localNodes[7];

    outputFile << SixthEleWidth_ << n1 << " " << 
			     n2 << " " <<
			     n3 << " " <<
			     n4 << " " <<
			     n5 << " " <<
			     n6 << " " <<
			     n7 << " " <<
			     n8;
    outputFile << "\n";

  }//end for(ie)

}//end writeVTU_connectivity

//////////////////////////////////////////////////////////////////////
//			writeVTU_offsets			    //
//////////////////////////////////////////////////////////////////////
void 
writeVTU_offsets (int nCells, ofstream &outputFile, int offset)
{
  string SecondEleWidth_   = "  ";
  string ThirdEleWidth_    = "    ";
  string FourthEleWidth_   = "      ";
  string FifthEleWidth_    = "        ";
  string SixthEleWidth_    = "          ";

  //write out connectivity offsets
  for (int ii = 0; ii < nCells; ii++)
  {
    int offCount = (ii+1)*8;
    outputFile << SixthEleWidth_ << offCount + offset;
    outputFile << "\n";
  }//end for(ii)

}//end writeVTU_offsets

//////////////////////////////////////////////////////////////////////
//			writeVTU_cellTypes			    //
//////////////////////////////////////////////////////////////////////
void 
writeVTU_cellTypes (int nCells, ofstream &outputFile)
{
  string SecondEleWidth_   = "  ";
  string ThirdEleWidth_    = "    ";
  string FourthEleWidth_   = "      ";
  string FifthEleWidth_    = "        ";
  string SixthEleWidth_    = "          ";

  //write out element type (8 node brick)
  for (int ii = 0; ii < nCells; ii++)
  {
    outputFile << SixthEleWidth_ << 12;
    outputFile << "\n";
  }//end for(ii)

}//end writeVTU_cellTypes

//////////////////////////////////////////////////////////////////////
//			writeVTU_pointData			    //
//////////////////////////////////////////////////////////////////////
void
writeVTU_pointData(vector<double> pointVec, ofstream &outputFile)
{
  string SecondEleWidth_   = "  ";
  string ThirdEleWidth_    = "    ";
  string FourthEleWidth_   = "      ";
  string FifthEleWidth_    = "        ";
  string SixthEleWidth_    = "          ";

  //write out point data
  for (int ii = 0; ii < pointVec.size(); ii++)
  {
    outputFile << SixthEleWidth_ << pointVec[ii];
    outputFile << "\n";
  }
}

//////////////////////////////////////////////////////////////////////
//			writeVTU_connectHeader			    //
//////////////////////////////////////////////////////////////////////
void 
writeVTU_connectHeader (ofstream &outputFile)
{
  string SecondEleWidth_   = "  ";
  string ThirdEleWidth_    = "    ";
  string FourthEleWidth_   = "      ";
  string FifthEleWidth_    = "        ";
  string SixthEleWidth_    = "          ";

  outputFile << FifthEleWidth_ <<"<DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
}//end writeVTU_connectHeader

//////////////////////////////////////////////////////////////////////
//			writeVTU_connectEnd			    //
//////////////////////////////////////////////////////////////////////
void 
writeVTU_connectEnd (ofstream &outputFile)
{
  string SecondEleWidth_   = "  ";
  string ThirdEleWidth_    = "    ";
  string FourthEleWidth_   = "      ";
  string FifthEleWidth_    = "        ";
  string SixthEleWidth_    = "          ";

  outputFile << FifthEleWidth_ <<"</DataArray>\n";
}//end writeVTU_connectEnd

//////////////////////////////////////////////////////////////////////
//			writeVTU_cellHeader			    //
//////////////////////////////////////////////////////////////////////
void 
writeVTU_cellHeader (ofstream &outputFile)
{
  string SecondEleWidth_   = "  ";
  string ThirdEleWidth_    = "    ";
  string FourthEleWidth_   = "      ";
  string FifthEleWidth_    = "        ";
  string SixthEleWidth_    = "          ";

  outputFile << FourthEleWidth_ <<"<Cells>\n";
}//end writeVTU_cellHeader

//////////////////////////////////////////////////////////////////////
//			writeVTU_cellEnd			    //
//////////////////////////////////////////////////////////////////////
void 
writeVTU_cellEnd (ofstream &outputFile)
{
  string SecondEleWidth_   = "  ";
  string ThirdEleWidth_    = "    ";
  string FourthEleWidth_   = "      ";
  string FifthEleWidth_    = "        ";
  string SixthEleWidth_    = "          ";

  outputFile << FourthEleWidth_ <<"</Cells>\n";
}//end writeVTU_cellEnd

//////////////////////////////////////////////////////////////////////
//			writeVTU_offsetHeader			    //
//////////////////////////////////////////////////////////////////////
void 
writeVTU_offsetHeader (ofstream &outputFile)
{
  string SecondEleWidth_   = "  ";
  string ThirdEleWidth_    = "    ";
  string FourthEleWidth_   = "      ";
  string FifthEleWidth_    = "        ";
  string SixthEleWidth_    = "          ";

  outputFile << FifthEleWidth_ <<"<DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
}//end writeVTU_offsetHeader

//////////////////////////////////////////////////////////////////////
//			writeVTU_offsetEnd			    //
//////////////////////////////////////////////////////////////////////
void 
writeVTU_offsetEnd (ofstream &outputFile)
{
  string SecondEleWidth_   = "  ";
  string ThirdEleWidth_    = "    ";
  string FourthEleWidth_   = "      ";
  string FifthEleWidth_    = "        ";
  string SixthEleWidth_    = "          ";

  outputFile << FifthEleWidth_ <<"</DataArray>\n";
}//end writeVTU_offsetEnd

//////////////////////////////////////////////////////////////////////
//			writeVTU_cellTypeHeader			    //
//////////////////////////////////////////////////////////////////////
void 
writeVTU_cellTypeHeader (ofstream &outputFile)
{
  string SecondEleWidth_   = "  ";
  string ThirdEleWidth_    = "    ";
  string FourthEleWidth_   = "      ";
  string FifthEleWidth_    = "        ";
  string SixthEleWidth_    = "          ";
  outputFile << FifthEleWidth_ <<"<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
}//end writeVTU_cellTypeHeader

//////////////////////////////////////////////////////////////////////
//			writeVTU_cellTypeEnd			    //
//////////////////////////////////////////////////////////////////////
void 
writeVTU_cellTypeEnd (ofstream &outputFile)
{
  string SecondEleWidth_   = "  ";
  string ThirdEleWidth_    = "    ";
  string FourthEleWidth_   = "      ";
  string FifthEleWidth_    = "        ";
  string SixthEleWidth_    = "          ";

  outputFile << FifthEleWidth_ <<"</DataArray>\n";
}//end writeVTU_cellTypeEnd

//////////////////////////////////////////////////////////////////////
//			writeVTU_header  			    //
//////////////////////////////////////////////////////////////////////
void 
writeVTU_header (int nCells, int nPoints, ofstream &outputFile)
{
  string SecondEleWidth_ = "  ";
  string ThirdEleWidth_  = "    ";
  string FourthEleWidth_ = "      ";
  string FifthEleWidth_  = "        ";
  string SixthEleWidth_  = "          ";

  outputFile << "<?xml version=\"1.0\"?>" << endl;
  outputFile << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";

  outputFile << SecondEleWidth_ << "<UnstructuredGrid>\n";
  outputFile << ThirdEleWidth_ << "<Piece NumberOfPoints=\""
                               << nPoints << "\""
                               << " NumberOfCells=\"" 
                               << nCells << "\">\n";
}

//////////////////////////////////////////////////////////////////////
//			writeVTU_end    			    //
//////////////////////////////////////////////////////////////////////
void 
writeVTU_end (ofstream &outputFile)
{
  string SecondEleWidth_ = "  ";
  string ThirdEleWidth_  = "    ";
  string FourthEleWidth_ = "      ";
  string FifthEleWidth_  = "        ";
  string SixthEleWidth_  = "          ";
  outputFile << ThirdEleWidth_ << "</Piece>\n";
  outputFile << SecondEleWidth_ << "</UnstructuredGrid>\n";
  outputFile << "</VTKFile>\n";
}

//////////////////////////////////////////////////////////////////////
//			writeVTU_pvtuHeader  			    //
//////////////////////////////////////////////////////////////////////
void 
writeVTU_pvtuHeader (ofstream &outputFile)
{
  string SecondEleWidth_ = "  ";
  string ThirdEleWidth_  = "    ";
  string FourthEleWidth_ = "      ";
  string FifthEleWidth_  = "        ";
  string SixthEleWidth_  = "          ";

  outputFile << "<?xml version=\"1.0\"?>" << endl;
  outputFile << "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";

  outputFile << SecondEleWidth_ << "<PUnstructuredGrid GhostLevel=\"0\">\n";
}

//////////////////////////////////////////////////////////////////////
//			writeVTU_pvtuMeshDataType		    //
//////////////////////////////////////////////////////////////////////
void 
writeVTU_pvtuMeshDataType (ofstream &outputFile)
{
  string SecondEleWidth_ = "  ";
  string ThirdEleWidth_  = "    ";
  string FourthEleWidth_ = "      ";
  string FifthEleWidth_  = "        ";
  string SixthEleWidth_  = "          ";
   
  // Data type for coordinates
  outputFile << ThirdEleWidth_ << "<PPoints>\n";
  outputFile << FourthEleWidth_ << "<PDataArray type=\"Float64\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\"/>\n";
  outputFile << ThirdEleWidth_ << "</PPoints>\n";

  // Data type for cells
  /*
  outputFile << ThirdEleWidth_ << "<PCells>\n";
  outputFile << FourthEleWidth_ << "<PDataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
  outputFile << FourthEleWidth_ << "<PDataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
  outputFile << FourthEleWidth_ << "<PDataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
  outputFile << ThirdEleWidth_ << "</PCells>\n";*/
}

//////////////////////////////////////////////////////////////////////
//			writeVTU_pvtuMeshDataType		    //
//////////////////////////////////////////////////////////////////////
void writeVTU_pvtuSources (vector<string> octreeFileNames, ofstream &outputFile)
{
  string SecondEleWidth_ = "  ";
  string ThirdEleWidth_  = "    ";
  string FourthEleWidth_ = "      ";
  string FifthEleWidth_  = "        ";
  string SixthEleWidth_  = "          ";
  for (int ii = 0; ii < octreeFileNames.size(); ii++)
  {
    outputFile << ThirdEleWidth_ << "<Piece Source=\""<< octreeFileNames[ii] << "\"/>\n";

  }
}//end writeVTU_Sources

//////////////////////////////////////////////////////////////////////
//			writeVTU_pvtuEnd		    	    //
//////////////////////////////////////////////////////////////////////
void writeVTU_pvtuEnd (ofstream &outputFile)
{
  string SecondEleWidth_ = "  ";
  string ThirdEleWidth_  = "    ";
  string FourthEleWidth_ = "      ";
  string FifthEleWidth_  = "        ";
  string SixthEleWidth_  = "          ";

  outputFile << SecondEleWidth_ << "</PUnstructuredGrid>\n";
  outputFile << "</VTKFile>";
}

//////////////////////////////////////////////////////////////////////
//			writeVTU_cellDataHeader		    	    //
//////////////////////////////////////////////////////////////////////
void writeVTU_cellDataHeader (ofstream &outputFile)
{
  string SecondEleWidth_ = "  ";
  string ThirdEleWidth_  = "    ";
  string FourthEleWidth_ = "      ";
  string FifthEleWidth_  = "        ";
  string SixthEleWidth_  = "          ";

  outputFile << FourthEleWidth_ <<"<CellData>\n";
}

//////////////////////////////////////////////////////////////////////
//			writeVTU_cellDataEnd		   	    //
//////////////////////////////////////////////////////////////////////
void writeVTU_cellDataEnd (ofstream &outputFile)
{
  string SecondEleWidth_ = "  ";
  string ThirdEleWidth_  = "    ";
  string FourthEleWidth_ = "      ";
  string FifthEleWidth_  = "        ";
  string SixthEleWidth_  = "          ";

  outputFile << FourthEleWidth_ <<"</CellData>\n";
}

//////////////////////////////////////////////////////////////////////
//			writeVTU_cellOutHeader		    	    //	
//////////////////////////////////////////////////////////////////////
void writeVTU_cellOutHeader (ofstream &outputFile)
{
  string SecondEleWidth_ = "  ";
  string ThirdEleWidth_  = "    ";
  string FourthEleWidth_ = "      ";
  string FifthEleWidth_  = "        ";
  string SixthEleWidth_  = "          ";

  outputFile << FifthEleWidth_ <<"<DataArray type=\"UInt8\" Name=\"ID\" format=\"ascii\">\n";
}

//////////////////////////////////////////////////////////////////////
//			writeVTU_cellOutEnd		    	    //	
//////////////////////////////////////////////////////////////////////
void writeVTU_cellOutEnd (ofstream &outputFile)
{
  string SecondEleWidth_ = "  ";
  string ThirdEleWidth_  = "    ";
  string FourthEleWidth_ = "      ";
  string FifthEleWidth_  = "        ";
  string SixthEleWidth_  = "          ";

  outputFile << FifthEleWidth_ <<"</DataArray>\n";
}

//////////////////////////////////////////////////////////////////////
//			writeVTU_pointDataHeader		    //
//////////////////////////////////////////////////////////////////////
void writeVTU_pointDataHeader (ofstream &outputFile)
{
  string SecondEleWidth_ = "  ";
  string ThirdEleWidth_  = "    ";
  string FourthEleWidth_ = "      ";
  string FifthEleWidth_  = "        ";
  string SixthEleWidth_  = "          ";

  outputFile << FourthEleWidth_ <<"<PointData>\n";
}

//////////////////////////////////////////////////////////////////////
//			writeVTU_pointDataEnd		   	    //
//////////////////////////////////////////////////////////////////////
void writeVTU_pointDataEnd (ofstream &outputFile)
{
  string SecondEleWidth_ = "  ";
  string ThirdEleWidth_  = "    ";
  string FourthEleWidth_ = "      ";
  string FifthEleWidth_  = "        ";
  string SixthEleWidth_  = "          ";

  outputFile << FourthEleWidth_ <<"</PointData>\n";
}

//////////////////////////////////////////////////////////////////////
//			writeVTU_pointOutHeader		    	    //	
//////////////////////////////////////////////////////////////////////
void writeVTU_pointOutHeader (ofstream &outputFile)
{
  string SecondEleWidth_ = "  ";
  string ThirdEleWidth_  = "    ";
  string FourthEleWidth_ = "      ";
  string FifthEleWidth_  = "        ";
  string SixthEleWidth_  = "          ";

  outputFile << FifthEleWidth_ <<"<DataArray type=\"Float64\" Name=\"temp\" format=\"ascii\">\n";
}

//////////////////////////////////////////////////////////////////////
//			writeVTU_pointOutEnd		    	    //	
//////////////////////////////////////////////////////////////////////
void writeVTU_pointOutEnd (ofstream &outputFile)
{
  string SecondEleWidth_ = "  ";
  string ThirdEleWidth_  = "    ";
  string FourthEleWidth_ = "      ";
  string FifthEleWidth_  = "        ";
  string SixthEleWidth_  = "          ";

  outputFile << FifthEleWidth_ <<"</DataArray>\n";
}//end writeVTU_pointOutEnd
