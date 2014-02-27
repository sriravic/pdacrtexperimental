#ifndef __CUDA_APP_H__
#define __CUDA_APP_H__

#pragma once
#include <global.h>

// utility classes to do some tests and properly set which cuda version to use
class Cuda
{
public:
	Cuda() {}
	void initCuda();
	static void printCudaDevices();
	
};



#endif