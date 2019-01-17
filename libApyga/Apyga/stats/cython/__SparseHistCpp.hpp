
#ifndef APYGA_STATS_SPARSEHISTCPP
#define APYGA_STATS_SPARSEHISTCPP

#include <vector>
#include "SparseHist.hpp"

typedef std::vector<double> VectValue ;
typedef std::vector<int> VectIndex ;
typedef std::vector<std::vector<double>> DataType ;
typedef Acyga::stats::SparseHist<VectValue,VectIndex,DataType> SparseHistCpp ;


#endif
