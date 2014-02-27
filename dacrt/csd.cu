#include <global.h>
#include <dacrt/csd.h>
#include <util/util.h>
#include <util/cutimer.h>

// given a sequence of key and values, we can do a very fast compressSortDecompress to get the final required array
// From Garanzha's work.

struct HeadFunctor
{
	__device__ __host__ bool operator() (int stencil_value) { return stencil_value == 1; }
};


__global__ void createHeadFlags(uint* values, uint* head_flags, uint num_elements) {
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	uint my_val;
	uint my_prev_val;
	if(tidx < num_elements) {
		my_val = values[tidx];
		if(tidx == 0) my_prev_val = UINT_MAX;			// this will never be my prev val. So we can expect a 1 in the head array
		else my_prev_val = values[tidx-1];
		head_flags[tidx] = (my_val != my_prev_val);
	}
}

__global__ void computeChunkSize(int* chunk_val, int* chunk_size, uint num_elements, uint num_chunks) {
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	int my_val;
	int my_next_val;
	if(tidx < num_chunks) {
		my_val = chunk_val[tidx];
		if(tidx == (num_chunks-1)) my_next_val = num_elements;
		else my_next_val = chunk_val[tidx+1];
		chunk_size[tidx] = my_next_val - my_val;
	}
}

__global__ void writeChunkBaseFlags(int* skeleton, int* head_flags2, int* chunk_size_scan, int* chunk_base, uint num_chunks) {
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	if(tidx < num_chunks) {
		skeleton[chunk_size_scan[tidx]] = chunk_base[tidx];
		head_flags2[chunk_size_scan[tidx]] = 1;
	}
}

// keys will be the sorted keys.!
__global__ void writeVals(int* keys, uint* input_vals, uint* output_vals, uint num_elements) {
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	if(tidx < num_elements) {
		output_vals[tidx] = input_vals[keys[tidx]];
	}
}

// this kernel will create the array with correct values
__global__ void createChunkBase(uint* head_flags, int* chunk_base, int* scan_head_flags, int num_elements) {
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	if(tidx < num_elements) {
		// scan_head_flags gives us the location to write the value
		// head_flags gives us a reason to write to.(if 1)
		if(head_flags[tidx] == 1) {
			chunk_base[scan_head_flags[tidx]] = tidx;
		}	
	}
}


// Note: The vals are the ones with which we want to sort.! In trifiltering and rayfiltering, the status is the vals here.
//       keys are the rayid/triangle id themselves.
void compressSortDecompress(int* keys, uint* vals, uint num_elements) {
	
#ifdef _DEBUG
	int* debugkeys1 = new int[num_elements];
	uint* debugvals1 = new uint[num_elements];
	checkCuda(cudaMemcpy(debugkeys1, keys, sizeof(int) * num_elements, cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(debugvals1, vals, sizeof(uint) * num_elements, cudaMemcpyDeviceToHost));
#endif


	//#1. Compress step

	//#1.1 first we create head flags
	uint* head_flags      = NULL;
	int* scan_head_flags = NULL;
	checkCuda(cudaMalloc((void**)&head_flags, sizeof(int) * num_elements));	
	checkCuda(cudaMalloc((void**)&scan_head_flags, sizeof(int) * num_elements));
	int NUM_THREADS = 512;
	int NUM_BLOCKS = (num_elements / NUM_THREADS) + (num_elements % NUM_THREADS != 0);
	Timer tim1("Timer1");
	tim1.start();
	createHeadFlags<<<NUM_BLOCKS, NUM_THREADS>>>(vals, head_flags, num_elements);
	tim1.stop();

#ifdef _DEBUG
	uint* debugheadflags1 = new uint[num_elements];
	checkCuda(cudaMemcpy(debugheadflags1, head_flags, sizeof(int) * num_elements, cudaMemcpyDeviceToHost));
#endif

	//#1.2 then we do a scan on the array (NOTE -> we dont need this.!)
	thrust::exclusive_scan(thrust::device_ptr<uint>(head_flags), thrust::device_ptr<uint>(head_flags) + num_elements, thrust::device_ptr<int>(scan_head_flags));

#ifdef _DEBUG
	int* debug9  = new int[num_elements];
	checkCuda(cudaMemcpy(debug9, scan_head_flags, sizeof(int) * num_elements, cudaMemcpyDeviceToHost));
#endif

	//#1.3 Now we do a compression step
	/// NOTE: the chunk base helps us at arriving at sizes respectively. ! We use the threadidx values to do it.
	//        But skeleton array needs the original id's starting at each chunk. So we use the dacrt_chunk_base for that.
	uint num_chunks       = thrust::reduce(thrust::device_ptr<uint>(head_flags), thrust::device_ptr<uint>(head_flags) + num_elements);
	int* chunk_base       = NULL;
	int* dacrt_chunk_base = NULL;			// we need this chunk base to fill the skeleton array and not the chunk base
	uint* chunk_val       = NULL;
	int* chunk_size       = NULL;
	checkCuda(cudaMalloc((void**)&chunk_base, sizeof(int) * num_chunks));
	checkCuda(cudaMalloc((void**)&chunk_val, sizeof(uint) * num_chunks));
	checkCuda(cudaMalloc((void**)&chunk_size, sizeof(int) * num_chunks));
	checkCuda(cudaMalloc((void**)&dacrt_chunk_base, sizeof(int) * num_chunks));
	thrust::copy_if(thrust::device_ptr<uint>(vals), thrust::device_ptr<uint>(vals) + num_elements, thrust::device_ptr<uint>(head_flags), thrust::device_ptr<uint>(chunk_val), HeadFunctor());
	thrust::copy_if(thrust::device_ptr<int>(keys), thrust::device_ptr<int>(keys) + num_elements, thrust::device_ptr<uint>(head_flags), thrust::device_ptr<int>(dacrt_chunk_base), HeadFunctor());

#ifdef _DEBUG
	uint* debug2 = new uint[num_chunks];
	int* debug8  = new int[num_chunks];
	checkCuda(cudaMemcpy(debug2, chunk_val, sizeof(uint) * num_chunks, cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(debug8, dacrt_chunk_base, sizeof(int) * num_chunks, cudaMemcpyDeviceToHost));
#endif


	NUM_BLOCKS = (num_elements / NUM_THREADS) + (num_elements % NUM_THREADS != 0);
	// call create chunkbase
	Timer tim5("Timer 5");
	tim5.start();
	createChunkBase<<<NUM_BLOCKS, NUM_THREADS>>>(head_flags, chunk_base, scan_head_flags, num_elements);
	tim5.stop();

#ifdef _DEBUG
	int* debug1  = new int[num_chunks];
	checkCuda(cudaMemcpy(debug1, chunk_base, sizeof(int) * num_chunks, cudaMemcpyDeviceToHost));
#endif

	NUM_BLOCKS = (num_chunks / NUM_THREADS) + (num_chunks % NUM_THREADS != 0);
	Timer tim2("Timer 2");
	tim2.start();
	computeChunkSize<<<NUM_BLOCKS, NUM_THREADS>>>(chunk_base, chunk_size, num_elements, num_chunks);
	tim2.stop();

#ifdef _DEBUG
	int debug_num = thrust::reduce(thrust::device_ptr<int>(chunk_size), thrust::device_ptr<int>(chunk_size) + num_chunks);
	int* debug3  = new int[num_chunks];
	checkCuda(cudaMemcpy(debug3, chunk_size, sizeof(int) * num_chunks, cudaMemcpyDeviceToHost));
#endif
	
	int place_holder = 0;
	// sort step
	// call thrust
	thrust::sort_by_key(thrust::device_ptr<uint>(chunk_val), thrust::device_ptr<uint>(chunk_val) + num_chunks, 
		thrust::make_zip_iterator(thrust::make_tuple(thrust::device_ptr<int>(chunk_base), thrust::device_ptr<int>(chunk_size), thrust::device_ptr<int>(dacrt_chunk_base))));

#ifdef _DEBUG
	int* debug5  = new int[num_chunks];
	uint* debug6 = new uint[num_chunks];
	int* debug7  = new int[num_chunks];
	checkCuda(cudaMemcpy(debug5, chunk_base, sizeof(int) * num_chunks, cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(debug6, chunk_val, sizeof(uint) * num_chunks, cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(debug7, chunk_size, sizeof(int) * num_chunks, cudaMemcpyDeviceToHost));
#endif


	// decompress step
	// #3.1 We do an exlusive scan on the chunk_sizes
	int* chunk_size_scan = NULL;
	checkCuda(cudaMalloc((void**)&chunk_size_scan, sizeof(int) * num_chunks));
	thrust::exclusive_scan(thrust::device_ptr<int>(chunk_size), thrust::device_ptr<int>(chunk_size) + num_chunks, thrust::device_ptr<int>(chunk_size_scan));

#ifdef _DEBUG
	int* debug4 = new int[num_chunks];
	checkCuda(cudaMemcpy(debug4, chunk_size_scan, sizeof(int) * num_chunks, cudaMemcpyDeviceToHost));
#endif

	int* skeleton = NULL;
	int* skeleton2 = NULL;
	int* head_flags2 = NULL;
	checkCuda(cudaMalloc((void**)&skeleton, sizeof(int) * num_elements));
	checkCuda(cudaMalloc((void**)&skeleton2, sizeof(int) * num_elements));
	checkCuda(cudaMalloc((void**)&head_flags2, sizeof(int) * num_elements));
	thrust::fill(thrust::device_ptr<int>(skeleton), thrust::device_ptr<int>(skeleton) + num_elements, 1);
	thrust::fill(thrust::device_ptr<int>(skeleton2), thrust::device_ptr<int>(skeleton2) + num_elements, 1);
	thrust::fill(thrust::device_ptr<int>(head_flags2), thrust::device_ptr<int>(head_flags2) + num_elements, 0);
	Timer tim3("Timer3");
	tim3.start();
	writeChunkBaseFlags<<<NUM_BLOCKS, NUM_THREADS>>>(skeleton, head_flags2, chunk_size_scan, dacrt_chunk_base, num_chunks);
	tim3.stop();

	Timer tim6("Timer5");
	tim6.start();
	writeChunkBaseFlags<<<NUM_BLOCKS, NUM_THREADS>>>(skeleton2, head_flags2, chunk_size_scan, chunk_base, num_chunks);
	tim6.stop();

	// #3.2 We do an inclusive segmented scan on skeleton
	// we need a segment list
	int* segmented_scan   = NULL;			// this will contain the final result. We have to swap this with the keys in the end 
	int* segmented_scan2  = NULL;
	uint* output_vals     = NULL;
	int* scan_head_flags2 = NULL;
	checkCuda(cudaMalloc((void**)&scan_head_flags2, sizeof(int) * num_elements));
	checkCuda(cudaMalloc((void**)&segmented_scan, sizeof(int) * num_elements));
	checkCuda(cudaMalloc((void**)&segmented_scan2, sizeof(int) * num_elements));
	checkCuda(cudaMalloc((void**)&output_vals, sizeof(uint) * num_elements));
	thrust::inclusive_scan(thrust::device_ptr<int>(head_flags2), thrust::device_ptr<int>(head_flags2) + num_elements, thrust::device_ptr<int>(scan_head_flags2));
	
#ifdef _DEBUG
	int* debugheadflags = new int[num_elements];
	int* debugheadsegfs = new int[num_elements];
	checkCuda(cudaMemcpy(debugheadflags, head_flags2, sizeof(int) * num_elements, cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(debugheadsegfs, scan_head_flags2, sizeof(int) * num_elements, cudaMemcpyDeviceToHost));
	
#endif


	// do an inclusive segmented scan of skeleton, using scan_head_flag2 for segment ids and put them into segmented_scan result
	thrust::inclusive_scan_by_key(thrust::device_ptr<int>(scan_head_flags2), thrust::device_ptr<int>(scan_head_flags2) + num_elements, thrust::device_ptr<int>(skeleton),
		thrust::device_ptr<int>(segmented_scan));
	
	thrust::inclusive_scan_by_key(thrust::device_ptr<int>(scan_head_flags2), thrust::device_ptr<int>(scan_head_flags2) + num_elements, thrust::device_ptr<int>(skeleton2),
		thrust::device_ptr<int>(segmented_scan2));

	// we now have to output the correct vals for each of the sorted elements.!
	Timer tim4("Timer4");
	NUM_BLOCKS = (num_elements / NUM_THREADS) + (num_elements % NUM_THREADS != 0);
	tim4.start();
	writeVals<<<NUM_BLOCKS, NUM_THREADS>>>(segmented_scan2, vals, output_vals, num_elements);
	tim4.stop();

#ifdef _DEBUG
	int*  debugkeys2 = new int[num_elements];
	uint* debugvals2 = new uint[num_elements];
	checkCuda(cudaMemcpy(debugkeys2, segmented_scan, sizeof(int) * num_elements, cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(debugvals2, output_vals, sizeof(uint) * num_elements, cudaMemcpyDeviceToHost));
	
	SAFE_RELEASE(debug1);
	SAFE_RELEASE(debug2);
	SAFE_RELEASE(debug3);
	SAFE_RELEASE(debug4);
	SAFE_RELEASE(debug5);
	SAFE_RELEASE(debug6);
	SAFE_RELEASE(debug7);
	SAFE_RELEASE(debug8);
	SAFE_RELEASE(debugkeys1);
	SAFE_RELEASE(debugvals1);
	SAFE_RELEASE(debugheadflags1);
	SAFE_RELEASE(debugheadflags);
	SAFE_RELEASE(debugheadsegfs);
	SAFE_RELEASE(debugkeys2);
	SAFE_RELEASE(debugvals2);
#endif

	// copy the segmented_scan and output vals to input_keys and input_vals
	checkCuda(cudaMemcpy(keys, segmented_scan, sizeof(int) * num_elements, cudaMemcpyDeviceToDevice));
	checkCuda(cudaMemcpy(vals, output_vals, sizeof(uint) * num_elements, cudaMemcpyDeviceToDevice));

	// free stuff
	CUDA_SAFE_RELEASE(head_flags);
	CUDA_SAFE_RELEASE(scan_head_flags);
	CUDA_SAFE_RELEASE(chunk_base);
	CUDA_SAFE_RELEASE(chunk_val);
	CUDA_SAFE_RELEASE(chunk_size);
	CUDA_SAFE_RELEASE(chunk_size_scan);
	CUDA_SAFE_RELEASE(skeleton);
	CUDA_SAFE_RELEASE(skeleton2);
	CUDA_SAFE_RELEASE(head_flags2);
	CUDA_SAFE_RELEASE(scan_head_flags2);
	CUDA_SAFE_RELEASE(segmented_scan);
	CUDA_SAFE_RELEASE(segmented_scan2);
	CUDA_SAFE_RELEASE(output_vals);
	CUDA_SAFE_RELEASE(dacrt_chunk_base);
}