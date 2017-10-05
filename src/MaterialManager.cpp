#include <vector>
#include <iterator>
#include <sstream>
#include <algorithm>
#include <string>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <iomanip>

#include <BCManager.h>
#include <MaterialManager.h>

////////////////////////////////////////
//		Constructor           //
////////////////////////////////////////
MaterialManager::MaterialManager()
                : updateMass_(false)
{
}

////////////////////////////////////////
//		Destructor           //
////////////////////////////////////////
MaterialManager::~MaterialManager()
{
}
