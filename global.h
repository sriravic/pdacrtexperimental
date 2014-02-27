/**
Divide and Conquer Ray Tracing - Extracting Parallelism from it.
Author : Srinath Ravichandran
*/

#ifndef __GLOBAL_H__
#define __GLOBAL_H__

#pragma once
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <fstream>
#include <cfloat>
#include <exception>
#include <cmath>
//#include <tuple>
#include <utility>
#include <queue>
#include <stack>
#include <cassert>
#include <omp.h>
#include <stdint.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/sort.h>
#include <thrust/for_each.h>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <thrust/for_each.h>
#include <thrust/transform.h>
#include <thrust/version.h>

#include "cutil_math.h"

#define SAFE_RELEASE(A) if(A) {delete[] A;}
#define CUDA_SAFE_RELEASE(A) if(A) { checkCuda(cudaFree(A)); }


#endif