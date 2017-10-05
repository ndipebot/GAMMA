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

#include <vtuBinWriter.h>
#include <DomainManager.h>
#include <HeatSolverManager.h>
#include <Element.h>
#include <Surface.h>

//////////////////////////////////////////////////////
//		Constructor			    //
//////////////////////////////////////////////////////
vtuBinWriter::vtuBinWriter(DomainManager *domainMgr,
                           HeatSolverManager *heatMgr,
                           string &FileOut)
                           : domainMgr_(domainMgr),
                             heatMgr_(heatMgr),
            	 	     FileOut_(FileOut)
{
}
///////////////////////////////////////////////////////////////////////
//			writeVTU_File				    //
//////////////////////////////////////////////////////////////////////
void 
vtuBinWriter::execute ()
{
  offSetCtr_ = 0;
  byteCtr_ = 0;
  outputFile_.open(FileOut_, ios::out | ios::binary);
  nCells_ = domainMgr_->activeElements_.size();
  nPoints_ = heatMgr_->thetaN_.size();
  writeVTU_header();

  writeVTU_coordsHeader();

  writeVTU_coordsEnd();

  writeVTU_pointDataHeader();
  
  writeVTU_pointOutHeader();

  writeVTU_pointDataEnd();

  writeVTU_cellHeader();

  writeVTU_connectHeader();

  writeVTU_offsetHeader();

  writeVTU_cellTypeHeader();

  writeVTU_cellEnd();

  // cell data (NEED TO MAKE THIS MORE GENERALIZED )

  writeVTU_cellDataHeader();
  
  writeVTU_cellOutHeader();
 
  writeVTU_cellDataEnd();

  writeVTU_end ();

  writeVTU_appendHeader();

  // Append data...

  writeVTU_coordinates();

  writeVTU_pointData(heatMgr_->thetaN_);

  writeVTU_connectivity();

  writeVTU_offsets (0);

  writeVTU_cellTypes();

  writeVTU_cellData();

  // end Append data ...

  writeVTU_appendEnd();

  
  outputFile_.close();
}//end writeVTU_File

//////////////////////////////////////////////////////////////////////
//			writeVTU_coordinates			    //
//////////////////////////////////////////////////////////////////////
void 
vtuBinWriter::writeVTU_coordinates ()
{

  byteCtr_ = sizeof(double) * nPoints_ * 3;
  outputFile_.write((char*) &byteCtr_, sizeof(int));
  for (int ii = 0; ii < nPoints_; ii++)
  {
    double xcurr = domainMgr_->coordList_[ii*3 + 0];
    double ycurr = domainMgr_->coordList_[ii*3 + 1];
    double zcurr = domainMgr_->coordList_[ii*3 + 2];
    outputFile_.write((char*) &xcurr, sizeof(double));
    outputFile_.write((char*) &ycurr, sizeof(double));
    outputFile_.write((char*) &zcurr, sizeof(double));
  }//end for(ii)

}//end writeVTU_coordinates

//////////////////////////////////////////////////////////////////////
//			writeVTU_coordsHeader			    //
//////////////////////////////////////////////////////////////////////
void 
vtuBinWriter::writeVTU_coordsHeader ()
{
  outputFile_ << "<Points> ";

  outputFile_ << "<DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"appended\" offset = \"" << offSetCtr_ << "\" /> ";

  offSetCtr_ += sizeof(double) * nPoints_ * 3 + sizeof(offSetCtr_);
}//end writeVTU_coordsHeader

//////////////////////////////////////////////////////////////////////
//			writeVTU_coordsEnd			    //
//////////////////////////////////////////////////////////////////////
void 
vtuBinWriter::writeVTU_coordsEnd ()
{

  outputFile_  << "</Points> ";

}//end writeVTU_coordsEnd

//////////////////////////////////////////////////////////////////////
//			writeVTU_connectivity			    //
//////////////////////////////////////////////////////////////////////
void 
vtuBinWriter::writeVTU_connectivity ()
{
  //write out connectivity data
  int n1, n2, n3, n4, n5, n6, n7, n8;
  byteCtr_ = sizeof(int) * nCells_ * 8;
  outputFile_.write((char*) &byteCtr_, sizeof(int));
  for (int ie = 0; ie < domainMgr_->activeElements_.size(); ie++)
  {
    int eID = domainMgr_->activeElements_[ie];
    Element * element = domainMgr_->elementList_[eID];
    int *localNodes = element->nID_;

    n1 = localNodes[0];
    n2 = localNodes[1];
    n3 = localNodes[2];
    n4 = localNodes[3];
    n5 = localNodes[4];
    n6 = localNodes[5];
    n7 = localNodes[6];
    n8 = localNodes[7];

    outputFile_.write((char*) &n1, sizeof(int));
    outputFile_.write((char*) &n2, sizeof(int));
    outputFile_.write((char*) &n3, sizeof(int));
    outputFile_.write((char*) &n4, sizeof(int));
    outputFile_.write((char*) &n5, sizeof(int));
    outputFile_.write((char*) &n6, sizeof(int));
    outputFile_.write((char*) &n7, sizeof(int));
    outputFile_.write((char*) &n8, sizeof(int));


  }//end for(ie)

}//end writeVTU_connectivity

//////////////////////////////////////////////////////////////////////
//			writeVTU_offsets			    //
//////////////////////////////////////////////////////////////////////
void 
vtuBinWriter::writeVTU_offsets (int offset)
{

  //write out connectivity offsets
  byteCtr_ = sizeof(int) * nCells_;
  outputFile_.write((char*) &byteCtr_, sizeof(int));
  for (int ii = 0; ii < nCells_; ii++)
  {
    int offCount = (ii+1)*8;
    outputFile_.write((char*) &offCount, sizeof(int));
  }//end for(ii)

}//end writeVTU_offsets

//////////////////////////////////////////////////////////////////////
//			writeVTU_cellTypes			    //
//////////////////////////////////////////////////////////////////////
void 
vtuBinWriter::writeVTU_cellTypes ()
{
  //write out element type (8 node brick)
  byteCtr_ = sizeof(int) * nCells_;
  outputFile_.write((char*) &byteCtr_, sizeof(int));
  int cellType = 12;
  for (int ii = 0; ii < nCells_; ii++)
  {
    outputFile_.write((char*) &cellType, sizeof(int));
  }//end for(ii)

}//end writeVTU_cellTypes

//////////////////////////////////////////////////////////////////////
//			writeVTU_pointData			    //
//////////////////////////////////////////////////////////////////////
void
vtuBinWriter::writeVTU_pointData(vector<double> &pointVec)
{
  // write out point data
  byteCtr_ = sizeof(double) * nPoints_;
  outputFile_.write((char*) &byteCtr_, sizeof(int));
  
  for (int ii = 0; ii < pointVec.size(); ii++)
  {
    outputFile_.write((char*) &pointVec[ii], sizeof(double));
  }
}

//////////////////////////////////////////////////////////////////////
//			writeVTU_connectHeader			    //
//////////////////////////////////////////////////////////////////////
void 
vtuBinWriter::writeVTU_connectHeader ()
{
  outputFile_ << "<DataArray type=\"Int32\" Name=\"connectivity\" format=\"appended\" offset=\"" << offSetCtr_ << "\" /> ";
  offSetCtr_ += sizeof(int) * nCells_ * 8 + sizeof(offSetCtr_);
}//end writeVTU_connectHeader

//////////////////////////////////////////////////////////////////////
//			writeVTU_connectEnd			    //
//////////////////////////////////////////////////////////////////////
void 
vtuBinWriter::writeVTU_connectEnd ()
{
  string SecondEleWidth_   = "  ";
  string ThirdEleWidth_    = "    ";
  string FourthEleWidth_   = "      ";
  string FifthEleWidth_    = "        ";
  string SixthEleWidth_    = "          ";

  outputFile_ << FifthEleWidth_ <<"</DataArray>\n";
}//end writeVTU_connectEnd

//////////////////////////////////////////////////////////////////////
//			writeVTU_cellHeader			    //
//////////////////////////////////////////////////////////////////////
void 
vtuBinWriter::writeVTU_cellHeader ()
{
  outputFile_ << "<Cells> ";
}//end writeVTU_cellHeader

//////////////////////////////////////////////////////////////////////
//			writeVTU_cellEnd			    //
//////////////////////////////////////////////////////////////////////
void 
vtuBinWriter::writeVTU_cellEnd ()
{
  outputFile_ << "</Cells> ";
}//end writeVTU_cellEnd

//////////////////////////////////////////////////////////////////////
//			writeVTU_offsetHeader			    //
//////////////////////////////////////////////////////////////////////
void 
vtuBinWriter::writeVTU_offsetHeader ()
{

  outputFile_ << "<DataArray type=\"Int32\" Name=\"offsets\" format=\"appended\" offset=\"" << offSetCtr_ << "\" /> ";
  offSetCtr_ += sizeof(int) * nCells_ + sizeof(offSetCtr_);
}//end writeVTU_offsetHeader

//////////////////////////////////////////////////////////////////////
//			writeVTU_offsetEnd			    //
//////////////////////////////////////////////////////////////////////
void 
vtuBinWriter::writeVTU_offsetEnd ()
{
  string SecondEleWidth_   = "  ";
  string ThirdEleWidth_    = "    ";
  string FourthEleWidth_   = "      ";
  string FifthEleWidth_    = "        ";
  string SixthEleWidth_    = "          ";

  outputFile_ << FifthEleWidth_ <<"</DataArray>\n";
}//end writeVTU_offsetEnd

//////////////////////////////////////////////////////////////////////
//			writeVTU_cellTypeHeader			    //
//////////////////////////////////////////////////////////////////////
void 
vtuBinWriter::writeVTU_cellTypeHeader ()
{
  outputFile_ << "<DataArray type=\"Int32\" Name=\"types\" format=\"appended\" offset=\"" << offSetCtr_ << "\" /> ";

  offSetCtr_ += sizeof(int) * nCells_ + sizeof(offSetCtr_);
}//end writeVTU_cellTypeHeader

//////////////////////////////////////////////////////////////////////
//			writeVTU_cellTypeEnd			    //
//////////////////////////////////////////////////////////////////////
void 
vtuBinWriter::writeVTU_cellTypeEnd ()
{
  string SecondEleWidth_   = "  ";
  string ThirdEleWidth_    = "    ";
  string FourthEleWidth_   = "      ";
  string FifthEleWidth_    = "        ";
  string SixthEleWidth_    = "          ";

  outputFile_ << FifthEleWidth_ <<"</DataArray>\n";
}//end writeVTU_cellTypeEnd

//////////////////////////////////////////////////////////////////////
//			writeVTU_header  			    //
//////////////////////////////////////////////////////////////////////
void 
vtuBinWriter::writeVTU_header ()
{
  string SecondEleWidth_ = "  ";
  string ThirdEleWidth_  = "    ";
  string FourthEleWidth_ = "      ";
  string FifthEleWidth_  = "        ";
  string SixthEleWidth_  = "          ";

  outputFile_ << "<?xml version=\"1.0\"?> ";
  outputFile_ << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\"> ";

  outputFile_ << "<UnstructuredGrid> ";
  outputFile_ << "<Piece NumberOfPoints=\""
                               << nPoints_ << "\""
                               << " NumberOfCells=\"" 
                               << nCells_ << "\"> ";
}

//////////////////////////////////////////////////////////////////////
//			writeVTU_end    			    //
//////////////////////////////////////////////////////////////////////
void 
vtuBinWriter::writeVTU_end ()
{
  string SecondEleWidth_ = "  ";
  string ThirdEleWidth_  = "    ";
  string FourthEleWidth_ = "      ";
  string FifthEleWidth_  = "        ";
  string SixthEleWidth_  = "          ";
  outputFile_ << "</Piece> ";
  outputFile_ << "</UnstructuredGrid> ";
//  outputFile_ << "</VTKFile>\n";
}

//////////////////////////////////////////////////////////////////////
//			writeVTU_pointDataHeader		    //
//////////////////////////////////////////////////////////////////////
void 
vtuBinWriter::writeVTU_pointDataHeader ()
{
  outputFile_ <<"<PointData> ";
}

//////////////////////////////////////////////////////////////////////
//			writeVTU_pointDataEnd		   	    //
//////////////////////////////////////////////////////////////////////
void 
vtuBinWriter::writeVTU_pointDataEnd ()
{
  outputFile_ <<"</PointData> ";
}

//////////////////////////////////////////////////////////////////////
//			writeVTU_pointOutHeader		    	    //	
//////////////////////////////////////////////////////////////////////
void 
vtuBinWriter::writeVTU_pointOutHeader ()
{
  outputFile_ << "<DataArray type=\"Float64\" Name=\"Temperature\" format=\"appended\" offset=\""
              << offSetCtr_ << "\" />  ";
  offSetCtr_ += sizeof(double) * nPoints_ + sizeof(offSetCtr_);

}

//////////////////////////////////////////////////////////////////////
//			writeVTU_pointOutEnd		    	    //	
//////////////////////////////////////////////////////////////////////
void 
vtuBinWriter::writeVTU_pointOutEnd ()
{
  string SecondEleWidth_ = "  ";
  string ThirdEleWidth_  = "    ";
  string FourthEleWidth_ = "      ";
  string FifthEleWidth_  = "        ";
  string SixthEleWidth_  = "          ";

  outputFile_ << FifthEleWidth_ <<"</DataArray>\n";
}//end writeVTU_pointOutEnd

//////////////////////////////////////////////////////////////////////
//			writeVTU_appendHeader		    	    //	
//////////////////////////////////////////////////////////////////////
void 
vtuBinWriter::writeVTU_appendHeader ()
{
  outputFile_ <<"<AppendedData encoding=\"raw\">_";
}//end writeVTU_appendHeader

//////////////////////////////////////////////////////////////////////
//			writeVTU_appendEnd		    	    //	
//////////////////////////////////////////////////////////////////////
void 
vtuBinWriter::writeVTU_appendEnd ()
{
  outputFile_ << "\n";
  outputFile_ << "</AppendedData> </VTKFile>";
}//end writeVTU_appendEnd

//////////////////////////////////////////////////////////////////////
//			writeVTU_cellDataHeader 		    //
//////////////////////////////////////////////////////////////////////
void 
vtuBinWriter::writeVTU_cellDataHeader ()
{
  outputFile_ <<"<CellData> ";
}

//////////////////////////////////////////////////////////////////////
//			writeVTU_cellOutHeader		    	    //	
//////////////////////////////////////////////////////////////////////
void 
vtuBinWriter::writeVTU_cellOutHeader ()
{
  outputFile_ << "<DataArray type=\"Float64\" Name=\"solidFrac\" format=\"appended\" offset=\""
              << offSetCtr_ << "\" />  ";
  offSetCtr_ += sizeof(double) * nCells_ + sizeof(offSetCtr_);

}

//////////////////////////////////////////////////////////////////////
//			writeVTU_cellDataEnd		   	    //
//////////////////////////////////////////////////////////////////////
void 
vtuBinWriter::writeVTU_cellDataEnd ()
{
  outputFile_ <<"</CellData> ";
}

//////////////////////////////////////////////////////////////////////
//			writeVTU_cellData		   	    //
//////////////////////////////////////////////////////////////////////
void 
vtuBinWriter::writeVTU_cellData ()
{
  // write out point data
  byteCtr_ = sizeof(double) * nCells_;
  outputFile_.write((char*) &byteCtr_, sizeof(int));
  
  for (int ie = 0; ie < domainMgr_->activeElements_.size(); ie++)
  {
    int eID = domainMgr_->activeElements_[ie];
    Element * element = domainMgr_->elementList_[eID];
    double * volWeights = element->volWeight_;
    double * consolidFrac = element->solidRate_;

    double consolidFracSum = 0.0;
    double volSum = 0.0;
    for (int ip = 0; ip < 8; ip++)
    {
      consolidFracSum += consolidFrac[ip] * volWeights[ip];
      volSum += volWeights[ip];
    }//end for(ip)  
    double result = consolidFracSum/volSum;
    outputFile_.write((char*) &result, sizeof(double));

  }//end for(ie)
}
