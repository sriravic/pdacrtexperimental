#include <cuda/cuda_app.h>

void Cuda::printCudaDevices() {
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	printf("=================== Device Details===============\n");
	printf("Device Name                    : %s\n", prop.name);
	printf("Device Multiprocessor count    : %d\n", prop.multiProcessorCount);
	printf("Device Per Block Shared Memory : %d kb\n", prop.sharedMemPerBlock);
	printf("Max Blocks                     : %d\n", prop.maxGridSize);
	printf("Max Threads Per Block          : %d\n", prop.maxThreadsPerBlock);
	printf("Major version                  : %d\n", prop.major);
	printf("Minor version                  : %d\n", prop.minor);
	printf("==================================================\n");
}

void Cuda::initCuda() {
	printf("Initializing Cuda\n");
	//printCudaDevices();
}