#include <Probe.h>
#include <math.h>

//==========================================================================
// Class Definition
//==========================================================================
// Holder for member variables and methods for each voxel
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
Probe::Probe(
 float *physCoords)
 : physCoords_(physCoords),
   hasElement_(false)
{
  //nothing to do
}
