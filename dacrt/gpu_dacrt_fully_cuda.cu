#include <dacrt/dacrt.h>
#include <util/cutimer.h>
#include <util/util.h>

std::ofstream memlog("memlog.txt");

extern "C" __global__ void modifiedSegmentedBruteForce(RayArray rays, TriangleArray triangles, int* buffered_ray_ids, int ray_buffer_occupied, int* buffered_tri_ids, int tri_buffer_occupied,
													   int* ray_segment_sizes, int* tri_segment_sizes, int* ray_segment_start, int* tri_segment_start, 
													   int* segment_no, int* blockStart, float* maxts, int* hitids);

extern "C" __global__ void parallelTriFilter(float3* v0, float3* v1, float3* v2, AABB* boxes, int* tri_ids, uint* keys, uint* segment_no, uint* segment_block_no, unsigned* segment_sizes, uint* trioffsets, int* tsegment_filter_status, unsigned int* split_axis, float* split_pos, uint total_elements, uint depth, unsigned int total_valid_blocks);
extern "C" __global__ void parallelRayFilter(float3* origin, float3* direction, AABB* boxes, int* ray_ids, uint* keys, uint* segment_no, uint* segment_block_no, uint* segment_sizes, uint* rayoffsets, int* rsegment_filter_status, uint total_elements, uint depth, unsigned int total_valid_blocks);
extern "C" void completeBruteForceModified(ParallelPackModified& pack, TriangleArray& d_triangles, RayArray& d_rays, DacrtRunTimeParameters& rtparams, Counters& ctr);
extern "C" void memoryusage(size_t memory_in_bytes);

// cuda parallel level. Fully uses device memory only. Exact equivalent of the level structure above this.
struct CuLevel
{
	int depth;
	AABB*   node_aabbs;
	uint2*  tpivots;
	uint2*  rpivots;
	uint*   tsegment_sizes;
	uint*   rsegment_sizes;
	int*    tri_idx;
	int*    ray_idx;
	uint    num_nodes;
	uint    num_tris;
	uint    num_rays;
	CuLevel() {
		depth = -1;
		node_aabbs = NULL;
		tpivots = rpivots = NULL;
		tsegment_sizes = rsegment_sizes = NULL;
		tri_idx = ray_idx = NULL;
		num_nodes = num_tris = num_rays = 0;
	}
	~CuLevel() {
		CUDA_SAFE_RELEASE(node_aabbs);
		CUDA_SAFE_RELEASE(tpivots);
		CUDA_SAFE_RELEASE(rpivots);
		CUDA_SAFE_RELEASE(tsegment_sizes);
		CUDA_SAFE_RELEASE(rsegment_sizes);
		CUDA_SAFE_RELEASE(tri_idx);
		CUDA_SAFE_RELEASE(ray_idx);
	}
	void allocateMemory(int _num_nodes, int _num_tris, int _num_rays) {
		checkCuda(cudaMalloc((void**)&node_aabbs, sizeof(AABB) * _num_nodes));
		checkCuda(cudaMalloc((void**)&tpivots, sizeof(uint2) * _num_nodes));
		checkCuda(cudaMalloc((void**)&rpivots, sizeof(uint2) * _num_nodes));
		checkCuda(cudaMalloc((void**)&tsegment_sizes, sizeof(uint) * _num_nodes));
		checkCuda(cudaMalloc((void**)&rsegment_sizes, sizeof(uint) * _num_nodes));
		checkCuda(cudaMalloc((void**)&tri_idx, sizeof(int) * _num_tris));
		checkCuda(cudaMalloc((void**)&ray_idx, sizeof(int) * _num_rays));
		num_nodes = _num_nodes;
		num_tris  = _num_tris;
		num_rays  = _num_rays;
	}
};

// functor to check if segment is not brute force segment
struct IsNotBruteForce
{
	__host__ __device__ bool operator() (const uint val) {
		return val != 0;
	}
};

struct IsBruteForce
{
	template<typename Tuple>
	__host__ __device__ void operator() (Tuple T) {
		// we pass in d_ptr sizes, d_seg_flags and the new bf_flags
		uint seg_flag = thrust::get<0>(T);
		uint tri_size = thrust::get<1>(T);
		uint ray_size = thrust::get<2>(T);
		uint res      = 0;
		if(seg_flag == 0 && (tri_size < 256 || ray_size < 256) && (tri_size > 0 && ray_size > 0)) {
			res = 1;
		}
		thrust::get<3>(T) = res;
	}
};

// some other kernels added to increase parallelism
__global__ void calculateBlocks(uint* d_tsegment_sizes, uint* d_rsegment_sizes, uint* d_tblocks, uint* d_rblocks, 
	uint NUM_THREADS_PER_BLOCK, uint num_segments, uint depth) {
		// NOTE: num_thread_per_segment is the number of threads per block that will be used to do triangle and ray filtering.
		//       Its value can change depending upon architecture.
		int tidx = threadIdx.x + blockIdx.x * blockDim.x;
		if(tidx < num_segments) {
			uint tsize			= d_tsegment_sizes[tidx];
			size_t tNUM_BLOCKS	= (tsize / NUM_THREADS_PER_BLOCK) + (tsize % NUM_THREADS_PER_BLOCK != 0);
			uint rsize			=	d_rsegment_sizes[tidx];
			size_t rNUM_BLOCKS	= (rsize / NUM_THREADS_PER_BLOCK) + (rsize % NUM_THREADS_PER_BLOCK != 0);
			// write result
			d_tblocks[tidx] = tNUM_BLOCKS;
			d_rblocks[tidx] = rNUM_BLOCKS;
		}
}

// This kernel computes the keyblocks and keyblockstart values for each segment
__global__ void calculateKeyBlocks(uint* d_tsegment_sizes, uint* d_rsegment_sizes, uint* d_tblocks, uint* d_rblocks, uint* d_tri_offset, uint* d_ray_offset,
	uint* d_tkeyblocks, uint* d_tkeyblockStart, uint* d_rkeyblocks, uint* d_rkeyblockStart, uint num_segments, uint depth) {

		// get offset for each segment using blockId 
		int tid = threadIdx.x;		// we launch one block of threads for each segment. That should keep occupancy high
		uint tblockSize = d_tblocks[blockIdx.x];
		uint rblockSize = d_rblocks[blockIdx.x];
		uint toffset    = d_tri_offset[blockIdx.x];
		uint roffset	= d_ray_offset[blockIdx.x];
		// we have to do tblockSize elements
		uint tpasses = (tblockSize / blockDim.x) + (tblockSize % blockDim.x != 0);
		uint rpasses = (rblockSize / blockDim.x) + (rblockSize % blockDim.x != 0);
		for(size_t i = 0; i < tpasses; i++) {
			if((i*blockDim.x + tid) < tblockSize) {
				d_tkeyblocks[i*blockDim.x + tid + toffset]	   = blockIdx.x;
				d_tkeyblockStart[i*blockDim.x + tid + toffset] = i*blockDim.x + tid;
			}
		}

		for(size_t i = 0; i < rpasses; i++) {
			if((i*blockDim.x + tid) < rblockSize) {
				d_rkeyblocks[i*blockDim.x + tid + roffset]	   = blockIdx.x;
				d_rkeyblockStart[i*blockDim.x + tid + roffset] = i*blockDim.x + tid;
			}
		}
}

// this method can calculate the ray and triangle sizes of all the segments parallely
// NOTE: Thanks to Parikshit Sagurikar for the wonderful idea.!
__global__ void calculateRayTriSizes(uint* d_trikeys, uint* d_trivalues, uint* d_raykeys, uint* d_rayvalues,
	uint* d_trisizes, uint* d_raysizes, unsigned num_tkeys, unsigned int num_rkeys, uint depth) {
		int tidx = threadIdx.x + blockIdx.x * blockDim.x;
		/*
		Idea: The idea is to use the key to locate where to scatter the correct value. Since we know that each key is one-to-one mapped with its
			  value, then all I have to is divide key by value to determine the count and put it in the index pointed by (key-4) due to the offset
			  by 2 bits for status.
		*/
		uint tricnt = 0;
		uint raycnt = 0;
		if(tidx < num_tkeys) {
			uint tkey = d_trikeys[tidx];
			uint tvalue = d_trivalues[tidx];
			if(tkey != 0) {
				tricnt = tvalue / tkey;			// note we end up with a divide by zero error?
				d_trisizes[tkey-4] = tricnt;
			}
			
		}
		if(tidx < num_rkeys) {
			uint rkey = d_raykeys[tidx];
			uint rvalue = d_rayvalues[tidx];
			if(rkey != 0) {
				raycnt = rvalue / rkey;
				d_raysizes[rkey-4] = raycnt;
			}
		}
}

// this kernel takes the output of the raytrisizes kernel to determine what each segments ray and triangle sizes i.e: left and right
// We can combine mark brute force here itself.!
/// NOTE: Unoptimized : Each thread is responsible for loading all the three/four values and putting it in
//		  its corresponding location
__global__ void calculateChildSegmentSizes(uint* d_trisizes, uint* d_raysizes, uint2* d_tpivots, uint2* d_rpivots,
	uint* d_tchild_segment_sizes, uint* d_rchild_segment_sizes, uint2* d_tchild_pivots, uint2* d_rchild_pivots,
	uint* segment_flags, uint num_segments, uint depth) {

		int tidx = threadIdx.x + blockIdx.x * blockDim.x;
		
		if(tidx < num_segments) {
			size_t tleft						 = d_trisizes[tidx * 4];
			size_t tboth						 = d_trisizes[tidx * 4 + 1];
			size_t tright						 = d_trisizes[tidx * 4 + 2];
			size_t next_tleft					 = tleft + tboth;
			size_t next_tright					 = tright + tboth;
			uint2 tparent_pivot					 = d_tpivots[tidx];
			d_tchild_pivots[tidx * 2]			 = make_uint2(tparent_pivot.x, tparent_pivot.x + tleft + tboth);
			d_tchild_pivots[tidx * 2 + 1]		 = make_uint2(tparent_pivot.x + tleft, tparent_pivot.y);
			d_tchild_segment_sizes[tidx * 2]	 = next_tleft;
			d_tchild_segment_sizes[tidx * 2 + 1] = next_tright;
			// for rays now
			size_t rleft						 = d_raysizes[tidx * 4];
			size_t rboth						 = d_raysizes[tidx * 4 + 1];
			size_t rright						 = d_raysizes[tidx * 4 + 2];
			size_t rnone						 = d_raysizes[tidx * 4 + 3];
			size_t next_rleft					 = rleft + rboth;
			size_t next_rright					 = rright + rboth;
			uint2 rparent_pivot					 = d_rpivots[tidx];
			d_rchild_pivots[tidx * 2]			 = make_uint2(rparent_pivot.x, rparent_pivot.x + rleft + rboth);
			d_rchild_pivots[tidx * 2 + 1]		 = make_uint2(rparent_pivot.x + rleft, rparent_pivot.y - rnone);
			d_rchild_segment_sizes[tidx * 2]	 = next_rleft;
			d_rchild_segment_sizes[tidx * 2 + 1] = next_rright;
			// now mark brute force
			segment_flags[tidx * 2]				 = (next_tleft > 256 && next_rleft > 256);
			segment_flags[tidx * 2 + 1]			 = (next_tright > 256 && next_rright > 256);
		}
}

// we have to somehow scatter all data using segment_id which is a one to one mapping
// global offset - used for the brute force scatter.
__global__ void triGatherData(int* d_tri_parent_idx, uint2* t_child_pivots, unsigned int* segment_no, unsigned int* segment_block_no, unsigned int* segment_sizes, 
	unsigned int* trioffsets, int* d_tri_child_idx, size_t global_offset, unsigned int total_elements, unsigned int depth) {

		unsigned int segment				= segment_no[blockIdx.x];
		unsigned int block_within_segment	= segment_block_no[blockIdx.x];
		unsigned int num_elements			= segment_sizes[segment];
		unsigned int offset					= trioffsets[segment];
		unsigned int tid_within_segment		= block_within_segment * 256 + threadIdx.x;
		unsigned int tid					= global_offset + offset + tid_within_segment;
		uint2 pivots						= t_child_pivots[segment];
		if(tid_within_segment < num_elements) {
			// find what my thread id is wrto the pivot ranges. Convert range to global range. Locate element and put it tid
			int id = tid_within_segment + pivots.x;
			d_tri_child_idx[tid] = d_tri_parent_idx[id];			// put the result.
		}
}

__global__ void rayGatherData(int* d_ray_parent_idx, uint2* r_child_pivots, unsigned int* segment_no, unsigned int* segment_block_no, unsigned int* segment_sizes, 
	unsigned int* rayoffsets, int* d_ray_child_idx, size_t global_offset, unsigned int total_elements, unsigned int depth) {
		unsigned int segment				= segment_no[blockIdx.x];
		unsigned int block_within_segment	= segment_block_no[blockIdx.x];
		unsigned int num_elements			= segment_sizes[segment];
		unsigned int offset					= rayoffsets[segment];
		unsigned int tid_within_segment		= block_within_segment * 256 + threadIdx.x;
		unsigned int tid					= global_offset + offset + tid_within_segment;
		uint2 pivots						= r_child_pivots[segment];
		if(tid_within_segment < num_elements) {
			// find what my thread id is wrto the pivot ranges. Convert range to global range. Locate element and put it tid
			int id = tid_within_segment + pivots.x;
			d_ray_child_idx[tid] = d_ray_parent_idx[id];			// put the result.
		}
}

// each thread computes two child boxes for its parent box at its idx and then outputs if corresponding (2*i) and (2*i+1) is 1 and if so copies the result to the output
// NOTE: writes will be uncoalasced i guess. We can improve by using shared memory and stuff.! But thats all later.!! Final optimization.!
__global__ void computeNextLevelAabbs(AABB* parent_boxes, AABB* child_boxes, uint* d_segment_flags, uint* d_next_level_location, uint num_parents, uint num_children, uint depth) {
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	if(tidx < num_parents) {
		// first compute boxes
		AABB left, right;
		AABB parent = parent_boxes[tidx];
		float split_pos;
		splitSpatialMedian(parent, left, right, split_pos);
		if(d_segment_flags[2 * tidx] == 1) {
			child_boxes[d_next_level_location[2*tidx]] = left;
		}
		if(d_segment_flags[2 * tidx + 1] == 1) {
			child_boxes[d_next_level_location[2*tidx + 1]] = right;
		}
	}
}

// this kernel can be used for both ray and triangle pivot reindexing purposes
__global__ void reindexPivots(uint* d_next_level_tsizes, uint* d_next_level_tscan, uint2* d_next_level_tpivots, 
	uint* d_next_level_rsizes, uint* d_next_level_rscan, uint2* d_next_level_rpivots,
	size_t num_next_level_segments, uint depth) {
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	if(tidx < num_next_level_segments) {
		uint my_tstart = d_next_level_tscan[tidx];
		uint my_rstart = d_next_level_rscan[tidx];
		uint my_tend   = 0;
		uint my_rend   = 0;	
		if(tidx < num_next_level_segments - 1) {
			my_tend = d_next_level_tscan[tidx + 1];
			my_rend = d_next_level_rscan[tidx + 1];
		} else {
			my_tend = d_next_level_tsizes[tidx] + d_next_level_tscan[tidx];
			my_rend = d_next_level_rsizes[tidx] + d_next_level_rscan[tidx];
		}
		// output them
		d_next_level_tpivots[tidx] = make_uint2(my_tstart, my_tend);
		d_next_level_rpivots[tidx] = make_uint2(my_rstart, my_rend);
	}
}

// return how much memory was allocated during this call. 
// We eventually free up space allocated within this call, but still we return to indicate how much memory we actually allocate.
size_t bruteforceScatter(int* tri_idx_array, int* ray_idx_array, size_t num_tris, size_t num_rays, uint2* d_child_tpivots, uint2* d_child_rpivots,
	uint* d_child_tsizes, uint* d_child_rsizes, uint* d_segment_flags, size_t num_child_segments, ParallelPackModified* pack, 
	DacrtRunTimeParameters& rtparams, Counters& ctr, uint depth) {

		// we first do a segment analysis on the segment flags to check for brute force and if sizes are all greater than zero.
		// we dont copy any zero size segments.
		uint* d_bf_segment_flags = NULL;
		double start, end;
		size_t allocated_memory = 0;
		checkCuda(cudaMalloc((void**)&d_bf_segment_flags, sizeof(uint) * num_child_segments));
		allocated_memory += sizeof(uint) * num_child_segments;
		Timer misctimer1("misc timer 1");
		misctimer1.start();
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(thrust::device_ptr<uint>(d_segment_flags), 
																	  thrust::device_ptr<uint>(d_child_tsizes),
																	  thrust::device_ptr<uint>(d_child_rsizes), 
																	  thrust::device_ptr<uint>(d_bf_segment_flags))),
						 thrust::make_zip_iterator(thrust::make_tuple(thrust::device_ptr<uint>(d_segment_flags) + num_child_segments,
																	  thrust::device_ptr<uint>(d_child_tsizes) + num_child_segments,
																	  thrust::device_ptr<uint>(d_child_rsizes) + num_child_segments,
																	  thrust::device_ptr<uint>(d_bf_segment_flags) + num_child_segments)),
						 IsBruteForce()
						 );

		// then we proceed ahead with the similar gather methods and add the global offset from previous brute force segments
		// compute the number of segments, number of triangles
		size_t num_bf_segments = thrust::reduce(thrust::device_ptr<uint>(d_bf_segment_flags), thrust::device_ptr<uint>(d_bf_segment_flags) + num_child_segments);
		misctimer1.stop();
		ctr.misc_time += misctimer1.get_ms();
		uint* next_bf_tsizes   = NULL;
		uint* next_bf_rsizes   = NULL;
		uint2* next_bf_tpivots = NULL;
		uint2* next_bf_rpivots = NULL;

#ifdef _DEBUG
		uint* debug_flags = new uint[num_child_segments];
		uint2* temp_tpivots = new uint2[num_child_segments];
		uint2* temp_rpivots = new uint2[num_child_segments];
		uint* temp_tsizes = new uint[num_child_segments];
		uint* temp_rsizes = new uint[num_child_segments];
		checkCuda(cudaMemcpy(debug_flags, d_bf_segment_flags, sizeof(uint) * num_child_segments, cudaMemcpyDeviceToHost));
		checkCuda(cudaMemcpy(temp_tpivots, d_child_tpivots, sizeof(uint2) * num_child_segments, cudaMemcpyDeviceToHost));
		checkCuda(cudaMemcpy(temp_rpivots, d_child_rpivots, sizeof(uint2) * num_child_segments, cudaMemcpyDeviceToHost));
		checkCuda(cudaMemcpy(temp_tsizes, d_child_tsizes, sizeof(uint) * num_child_segments, cudaMemcpyDeviceToHost));
		checkCuda(cudaMemcpy(temp_rsizes, d_child_rsizes, sizeof(uint) * num_child_segments, cudaMemcpyDeviceToHost));
#endif
		
		// do a copy if
		// NOTE: pivots should be the original pivot ranges and not reindex pivots.!
		if(num_bf_segments > 0) {

			checkCuda(cudaMalloc((void**)&next_bf_tsizes, sizeof(uint) * num_bf_segments));
			checkCuda(cudaMalloc((void**)&next_bf_rsizes, sizeof(uint) * num_bf_segments));
			checkCuda(cudaMalloc((void**)&next_bf_tpivots, sizeof(uint2) * num_bf_segments));
			checkCuda(cudaMalloc((void**)&next_bf_rpivots, sizeof(uint2) * num_bf_segments));
			allocated_memory += num_bf_segments * (sizeof(uint) * 2 + sizeof(uint2) * 2);
			//Timer misctimer2("memcpy timer 1");
			//misctimer2.start();
			start = omp_get_wtime();
			thrust::copy_if(thrust::device_ptr<uint>(d_child_tsizes), thrust::device_ptr<uint>(d_child_tsizes) + num_child_segments, 
				thrust::device_ptr<uint>(d_bf_segment_flags), thrust::device_ptr<uint>(next_bf_tsizes), IsNotBruteForce());		// IsNotBruteForce is a more like a flag checker only
			thrust::copy_if(thrust::device_ptr<uint>(d_child_rsizes), thrust::device_ptr<uint>(d_child_rsizes) + num_child_segments,
				thrust::device_ptr<uint>(d_bf_segment_flags), thrust::device_ptr<uint>(next_bf_rsizes), IsNotBruteForce());
			thrust::copy_if(thrust::device_ptr<uint2>(d_child_tpivots), thrust::device_ptr<uint2>(d_child_tpivots) + num_child_segments,
				thrust::device_ptr<uint>(d_bf_segment_flags), thrust::device_ptr<uint2>(next_bf_tpivots), IsNotBruteForce());
			thrust::copy_if(thrust::device_ptr<uint2>(d_child_rpivots), thrust::device_ptr<uint2>(d_child_rpivots) + num_child_segments,
				thrust::device_ptr<uint>(d_bf_segment_flags), thrust::device_ptr<uint2>(next_bf_rpivots), IsNotBruteForce());
			end = omp_get_wtime();
			ctr.misc_time += ((end-start) * 1000.0f);
			//misctimer2.stop();
			//ctr.misc_time += misctimer2.get_ms();

#ifdef _DEBUG
			uint* debug_tsizes = new uint[num_bf_segments];
			uint* debug_rsizes = new uint[num_bf_segments];
			uint2* debug_tpivots = new uint2[num_bf_segments];
			uint2* debug_rpivots = new uint2[num_bf_segments];
			checkCuda(cudaMemcpy(debug_tsizes, next_bf_tsizes, sizeof(uint) * num_bf_segments, cudaMemcpyDeviceToHost));
			checkCuda(cudaMemcpy(debug_rsizes, next_bf_rsizes, sizeof(uint) * num_bf_segments, cudaMemcpyDeviceToHost));
			checkCuda(cudaMemcpy(debug_tpivots, next_bf_tpivots, sizeof(uint2) * num_bf_segments, cudaMemcpyDeviceToHost));
			checkCuda(cudaMemcpy(debug_rpivots, next_bf_rpivots, sizeof(uint2) * num_bf_segments, cudaMemcpyDeviceToHost));
#endif
			
			start = omp_get_wtime();
			size_t num_bf_tris = thrust::reduce(thrust::device_ptr<uint>(next_bf_tsizes), thrust::device_ptr<uint>(next_bf_tsizes) + num_bf_segments);
			size_t num_bf_rays = thrust::reduce(thrust::device_ptr<uint>(next_bf_rsizes), thrust::device_ptr<uint>(next_bf_rsizes) + num_bf_segments);
			end = omp_get_wtime();
			ctr.misc_time += ((end-start) * 1000.0f);

			if((pack->num_segments + num_bf_segments > rtparams.MAX_SEGMENTS) || 
			  (pack->ray_buffer_occupied + num_bf_rays > rtparams.BUFFER_SIZE) || 
			  (pack->tri_buffer_occupied + num_bf_tris > rtparams.BUFFER_SIZE)) {

				   // do a looped insertion with brute force performed everytime.
				   size_t num_loops = 0;

			} else {
				// proceed ahead with insertion
				uint* d_tsegmentNo	     = NULL;
				uint* d_tsegmentBlocks   = NULL;
				uint* d_rsegmentNo	     = NULL;
				uint* d_rsegmentBlocks   = NULL;
				uint* d_tsegmentBlockCnt = NULL;
				uint* d_rsegmentBlockCnt = NULL;
				uint* d_tseg_offsets	 = NULL;
				uint* d_rseg_offsets	 = NULL;
				uint* d_triblock_offset	 = NULL;
				uint* d_rayblock_offset	 = NULL;
				checkCuda(cudaMalloc((void**)&d_tsegmentBlockCnt, sizeof(uint) * num_bf_segments));
				checkCuda(cudaMalloc((void**)&d_rsegmentBlockCnt, sizeof(uint) * num_bf_segments));
				checkCuda(cudaMalloc((void**)&d_tseg_offsets,	  sizeof(uint) * num_bf_segments));
				checkCuda(cudaMalloc((void**)&d_rseg_offsets,     sizeof(uint) * num_bf_segments));
				checkCuda(cudaMalloc((void**)&d_triblock_offset,  sizeof(uint) * num_bf_segments));
				checkCuda(cudaMalloc((void**)&d_rayblock_offset,  sizeof(uint) * num_bf_segments));
				allocated_memory += sizeof(uint) * num_bf_segments * 6;

				size_t NUM_THREADS_PER_BLOCK = 1024;
				size_t NUM_BLOCKS = (num_bf_segments / NUM_THREADS_PER_BLOCK) + (num_bf_segments % NUM_THREADS_PER_BLOCK != 0);

				Timer misctimer3("Misc Timer 3");
				misctimer3.start();
				calculateBlocks<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(next_bf_tsizes, next_bf_rsizes, d_tsegmentBlockCnt, 
					d_rsegmentBlockCnt, 256, num_bf_segments, depth);
				misctimer3.stop();
				ctr.misc_time += misctimer3.get_ms();

				// reduce to get count and then fill with data
				//Timer misctimer4("misc timer 4");
				//misctimer4.start();
				start = omp_get_wtime();
				size_t tseg_blocks = thrust::reduce(thrust::device_ptr<uint>(d_tsegmentBlockCnt), thrust::device_ptr<uint>(d_tsegmentBlockCnt) + num_bf_segments);
				size_t rseg_blocks = thrust::reduce(thrust::device_ptr<uint>(d_rsegmentBlockCnt), thrust::device_ptr<uint>(d_rsegmentBlockCnt) + num_bf_segments);
				thrust::exclusive_scan(thrust::device_ptr<uint>(next_bf_tsizes), thrust::device_ptr<uint>(next_bf_tsizes) + num_bf_segments, thrust::device_ptr<uint>(d_tseg_offsets));
				thrust::exclusive_scan(thrust::device_ptr<uint>(next_bf_rsizes), thrust::device_ptr<uint>(next_bf_rsizes) + num_bf_segments, thrust::device_ptr<uint>(d_rseg_offsets));
				thrust::exclusive_scan(thrust::device_ptr<uint>(d_tsegmentBlockCnt), thrust::device_ptr<uint>(d_tsegmentBlockCnt) + num_bf_segments, thrust::device_ptr<uint>(d_triblock_offset));
				thrust::exclusive_scan(thrust::device_ptr<uint>(d_rsegmentBlockCnt), thrust::device_ptr<uint>(d_rsegmentBlockCnt) + num_bf_segments, thrust::device_ptr<uint>(d_rayblock_offset));
				end = omp_get_wtime();
				ctr.misc_time += ((end-start) * 1000.0f);
				//misctimer4.stop();
				//ctr.misc_time += misctimer3.get_ms();
				// allocate space
				checkCuda(cudaMalloc((void**)&d_tsegmentNo, sizeof(uint) * tseg_blocks));
				checkCuda(cudaMalloc((void**)&d_rsegmentNo, sizeof(uint) * rseg_blocks));
				checkCuda(cudaMalloc((void**)&d_tsegmentBlocks, sizeof(uint) * tseg_blocks));
				checkCuda(cudaMalloc((void**)&d_rsegmentBlocks, sizeof(uint) * rseg_blocks));
				allocated_memory += sizeof(uint) * (2*tseg_blocks + 2*rseg_blocks);

				Timer tim2("Timer 2");
				tim2.start();
				calculateKeyBlocks<<<num_bf_segments, NUM_THREADS_PER_BLOCK>>>(next_bf_tsizes, next_bf_rsizes, d_tsegmentBlockCnt, 
					d_rsegmentBlockCnt, d_triblock_offset, d_rayblock_offset, d_tsegmentNo, d_tsegmentBlocks, d_rsegmentNo, d_rsegmentBlocks, 
					num_bf_segments, depth);
				tim2.stop();
				ctr.misc_time += tim2.get_ms();
				// we can gather data in the buffer now.!
				NUM_THREADS_PER_BLOCK = 256;

				Timer tim3("Timer 3");
				tim3.start();
				triGatherData<<<tseg_blocks, NUM_THREADS_PER_BLOCK>>>(tri_idx_array, next_bf_tpivots, d_tsegmentNo, d_tsegmentBlocks, 
					next_bf_tsizes, d_tseg_offsets, thrust::raw_pointer_cast(&pack->buffered_tri_idx[0]), pack->tri_buffer_occupied, 
					num_bf_tris, depth);		// set global offset to zero. 
				tim3.stop();
				ctr.mem_cpy_time += tim3.get_ms();
			
				Timer tim4("Timer 4");
				tim4.start();
				rayGatherData<<<rseg_blocks, NUM_THREADS_PER_BLOCK>>>(ray_idx_array, next_bf_rpivots, d_rsegmentNo, d_rsegmentBlocks, 
					next_bf_rsizes, d_rseg_offsets, thrust::raw_pointer_cast(&pack->buffered_ray_idx[0]), pack->ray_buffer_occupied, 
					num_bf_rays, depth);
				tim4.stop();
				ctr.mem_cpy_time += tim4.get_ms();

				// update the pack's contents.! 
				// this is serial code actually. We copy the brute force segment data and then we do 
				uint* hsegment_tsize = new uint[num_bf_segments];
				uint* hsegment_rsize = new uint[num_bf_segments];
				checkCuda(cudaMemcpy(hsegment_tsize, next_bf_tsizes, sizeof(uint) * num_bf_segments, cudaMemcpyDeviceToHost));
				checkCuda(cudaMemcpy(hsegment_rsize, next_bf_rsizes, sizeof(uint) * num_bf_segments, cudaMemcpyDeviceToHost));
				start = omp_get_wtime();
				for(size_t i = 0; i < num_bf_segments; i++) {
					pack->htri_segment_sizes[pack->num_segments] = hsegment_tsize[i];
					pack->hray_segment_sizes[pack->num_segments] = hsegment_rsize[i];
					pack->hsegment_ids[pack->num_segments] = pack->num_segments;
				
					int numblocks = (hsegment_rsize[i] / rtparams.NUM_RAYS_PER_BLOCK) + (hsegment_rsize[i] % rtparams.NUM_RAYS_PER_BLOCK != 0);
					pack->blockCnt += numblocks;
					int tempstart = 0;
					for(int j = pack->bstart; j < pack->blockCnt; j++) {
						pack->blockNos[j]   = pack->num_segments;
						pack->blockStart[j] = tempstart++;
					}
					pack->bstart += numblocks;
					pack->num_segments++;		// increment the count
					pack->ray_buffer_occupied += hsegment_rsize[i];
					pack->tri_buffer_occupied += hsegment_tsize[i];
				}
				

#ifdef _DEBUG
				SAFE_RELEASE(debug_tsizes);
				SAFE_RELEASE(debug_rsizes);
				SAFE_RELEASE(debug_tpivots);
				SAFE_RELEASE(debug_rpivots);
#endif
				
				SAFE_RELEASE(hsegment_tsize);
				SAFE_RELEASE(hsegment_rsize);
				CUDA_SAFE_RELEASE(d_tsegmentNo);
				CUDA_SAFE_RELEASE(d_tsegmentBlocks);
				CUDA_SAFE_RELEASE(d_rsegmentNo);
				CUDA_SAFE_RELEASE(d_rsegmentBlocks);
				CUDA_SAFE_RELEASE(d_tsegmentBlockCnt);
				CUDA_SAFE_RELEASE(d_rsegmentBlockCnt);
				CUDA_SAFE_RELEASE(d_tseg_offsets);
				CUDA_SAFE_RELEASE(d_rseg_offsets);
				CUDA_SAFE_RELEASE(d_triblock_offset);
				CUDA_SAFE_RELEASE(d_rayblock_offset);
				end = omp_get_wtime();
				ctr.misc_time += ((end-start) * 1000.0f);		// convert to milliseconds
			}
		}

#ifdef _DEBUG
		SAFE_RELEASE(debug_flags);
		SAFE_RELEASE(temp_tpivots);
		SAFE_RELEASE(temp_rpivots);
		SAFE_RELEASE(temp_tsizes);
		SAFE_RELEASE(temp_rsizes);
#endif

		CUDA_SAFE_RELEASE(next_bf_tsizes);
		CUDA_SAFE_RELEASE(next_bf_rsizes);
		CUDA_SAFE_RELEASE(next_bf_tpivots);
		CUDA_SAFE_RELEASE(next_bf_rpivots);
		return allocated_memory;
}


void gpuDacrtFullyCuda(
	AABB& root, 
	TriangleArray& d_triangles, int* tri_idx_array, int num_triangles, int tpivot,			// added a separate term for num_triangles as well as pivot. This is to facilitate some kind of LOD/back face culling mechanism
	RayArray& d_rays, int* ray_idx_array, int num_rays, int rpivot, 
	float* d_maxts, int* d_hitids, 
	DacrtRunTimeParameters& rtparams, Counters& ctr, Logger& logger) {

		double dstart, dend;
		CuLevel* level = new CuLevel();
		uint2 ray_pivot = make_uint2(0, rpivot);
		uint2 tri_pivot = make_uint2(0, tpivot);
		uint  raysizes  = rpivot;
		uint  trisizes  = tpivot;

		level->depth     = 0;
		level->num_nodes = 1;
		level->num_rays  = rpivot;
		level->num_tris  = tpivot;
		level->allocateMemory(1, tpivot, rpivot);
		Timer inittimer("Init timer");
		inittimer.start();
		checkCuda(cudaMemcpy(level->tri_idx, tri_idx_array, sizeof(int) * tpivot, cudaMemcpyDeviceToDevice));
		checkCuda(cudaMemcpy(level->ray_idx, ray_idx_array, sizeof(int) * rpivot, cudaMemcpyDeviceToDevice));
		checkCuda(cudaMemcpy(level->node_aabbs, &(root), sizeof(AABB), cudaMemcpyHostToDevice));
		checkCuda(cudaMemcpy(level->rpivots, &(ray_pivot), sizeof(uint2), cudaMemcpyHostToDevice));
		checkCuda(cudaMemcpy(level->tpivots, &(tri_pivot), sizeof(uint2), cudaMemcpyHostToDevice));
		checkCuda(cudaMemcpy(level->rsegment_sizes, &(raysizes), sizeof(uint), cudaMemcpyHostToDevice));
		checkCuda(cudaMemcpy(level->tsegment_sizes, &(trisizes), sizeof(uint), cudaMemcpyHostToDevice));
		inittimer.stop();
		ctr.mem_cpy_time += inittimer.get_ms();
		
		std::stack<CuLevel*> working_stack;
		working_stack.push(level);

		int depth			   = 0;
		// dev memory used for the root_level.
		size_t max_dev_memory  = (sizeof(int) * tpivot + sizeof(int) * rpivot + sizeof(AABB) + sizeof(uint2) + sizeof(uint2) + sizeof(uint) + sizeof(uint));
		size_t max_host_memory = 0;
		ParallelPackModified *pack = new ParallelPackModified(rtparams, num_rays);
		
		while(working_stack.top()->num_nodes != 0) {

#ifdef _DEBUG
			printf("Depth Reached : %d\n", depth);
#endif

			CuLevel* work_level = working_stack.top();
			working_stack.pop();

			size_t num_segments					= work_level->num_nodes;
			size_t per_iteration_device_memory	= 0;
			size_t trikeylen					= work_level->num_tris;
			size_t raykeylen					= work_level->num_rays;
			size_t tblocks						= 0;
			size_t rblocks						= 0;
			
			uint* d_tblocks				= NULL;
			uint* d_rblocks				= NULL;
			uint* d_tkeyblocks			= NULL;
			uint* d_rkeyblocks			= NULL;
			uint* d_tkeyblockStart		= NULL;
			uint* d_rkeyblockStart		= NULL;
			uint* d_tsegmentSizes		= NULL;
			uint* d_rsegmentSizes		= NULL;
			uint* d_trioffsets			= NULL;
			uint* d_rayoffsets			= NULL;
			uint* d_tblock_offset		= NULL;		// we need offset array to correctly fill values for the block and blockstart arrays
			uint* d_rblock_offset		= NULL;

			// #1. Calculate number of blocks are required for work parallel tri and ray filtering process.
			checkCuda(cudaMalloc((void**)&d_tblocks, sizeof(uint) * num_segments));
			checkCuda(cudaMalloc((void**)&d_rblocks, sizeof(uint) * num_segments));
			checkCuda(cudaMalloc((void**)&d_trioffsets, sizeof(uint) * num_segments));
			checkCuda(cudaMalloc((void**)&d_rayoffsets, sizeof(uint) * num_segments));
			checkCuda(cudaMalloc((void**)&d_tblock_offset, sizeof(uint) * num_segments));
			checkCuda(cudaMalloc((void**)&d_rblock_offset, sizeof(uint) * num_segments));
			per_iteration_device_memory += (sizeof(uint) * num_segments * 6);

			size_t NUM_THREADS_PER_BLOCK = 1024;
			size_t NUM_BLOCKS		     = (num_segments / NUM_THREADS_PER_BLOCK) + (num_segments % NUM_THREADS_PER_BLOCK != 0);

			Timer nbtimer("NB timer");
			nbtimer.start();
			calculateBlocks<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(work_level->tsegment_sizes, work_level->rsegment_sizes, d_tblocks, d_rblocks, 
				256, num_segments, depth);
			nbtimer.stop();
			ctr.misc_time += nbtimer.get_ms();
			
			Timer misc1timer("Misc1 timer");
			misc1timer.start();
			tblocks = thrust::reduce(thrust::device_ptr<uint>(d_tblocks), thrust::device_ptr<uint>(d_tblocks) + num_segments);
			rblocks = thrust::reduce(thrust::device_ptr<uint>(d_rblocks), thrust::device_ptr<uint>(d_rblocks) + num_segments);
			thrust::exclusive_scan(thrust::device_ptr<uint>(work_level->tsegment_sizes), thrust::device_ptr<uint>(work_level->tsegment_sizes) + num_segments, thrust::device_ptr<uint>(d_trioffsets));
			thrust::exclusive_scan(thrust::device_ptr<uint>(work_level->rsegment_sizes), thrust::device_ptr<uint>(work_level->rsegment_sizes) + num_segments, thrust::device_ptr<uint>(d_rayoffsets));
			thrust::exclusive_scan(thrust::device_ptr<uint>(d_tblocks), thrust::device_ptr<uint>(d_tblocks) + num_segments, thrust::device_ptr<uint>(d_tblock_offset));
			thrust::exclusive_scan(thrust::device_ptr<uint>(d_rblocks), thrust::device_ptr<uint>(d_rblocks) + num_segments, thrust::device_ptr<uint>(d_rblock_offset));
			misc1timer.stop();
			ctr.misc_time += misc1timer.get_ms();

			//allocate memory for the blocks and blockStart
			checkCuda(cudaMalloc((void**)&d_tkeyblocks, sizeof(uint) * tblocks));
			checkCuda(cudaMalloc((void**)&d_rkeyblocks, sizeof(uint) * rblocks));
			checkCuda(cudaMalloc((void**)&d_tkeyblockStart, sizeof(uint) * tblocks));
			checkCuda(cudaMalloc((void**)&d_rkeyblockStart, sizeof(uint) * rblocks));
			per_iteration_device_memory += (sizeof(uint) * (2*tblocks + 2*rblocks));

			Timer kbtimer("kb timer");
			kbtimer.start();
			calculateKeyBlocks<<<num_segments, NUM_THREADS_PER_BLOCK>>>(work_level->tsegment_sizes, work_level->rsegment_sizes, d_tblocks, d_rblocks, 
				d_tblock_offset, d_rblock_offset, d_tkeyblocks, d_tkeyblockStart, d_rkeyblocks, d_rkeyblockStart, num_segments, depth);
			kbtimer.stop();
			ctr.misc_time += kbtimer.get_ms();

#ifdef _DEBUG
			/*------------------------------ DEBUG CODE ---------------------------------*/
			uint* debugtblock = new uint[num_segments];
			uint* debugrblock = new uint[num_segments];
			uint* debugtkeyblock = new uint[tblocks];
			uint* debugrkeyblock = new uint[rblocks];
			uint* debugtkeyblockstart = new uint[tblocks];
			uint* debugrkeyblockstart = new uint[rblocks];
			checkCuda(cudaMemcpy(debugtblock, d_tblocks, sizeof(uint) * num_segments, cudaMemcpyDeviceToHost));
			checkCuda(cudaMemcpy(debugrblock, d_rblocks, sizeof(uint) * num_segments, cudaMemcpyDeviceToHost));
			checkCuda(cudaMemcpy(debugtkeyblock, d_tkeyblocks, sizeof(uint)*tblocks, cudaMemcpyDeviceToHost));
			checkCuda(cudaMemcpy(debugrkeyblock, d_rkeyblocks, sizeof(uint)*rblocks, cudaMemcpyDeviceToHost));
			checkCuda(cudaMemcpy(debugtkeyblockstart, d_tkeyblockStart, sizeof(uint)*tblocks, cudaMemcpyDeviceToHost));
			checkCuda(cudaMemcpy(debugrkeyblockstart, d_rkeyblockStart, sizeof(uint)*rblocks, cudaMemcpyDeviceToHost));
			/*------------------------------ END OF DEBUG CODE ---------------------------------*/
#endif
			
			// #2. Key and status calculation
			uint* d_trikeys		  = NULL;
			uint* d_raykeys		  = NULL;
			uint* d_split_axis    = NULL;
			float* d_split_pos    = NULL;
			int* d_tsegment_filter_status = NULL;
			int* d_rsegment_filter_status = NULL;
			checkCuda(cudaMalloc((void**)&d_split_axis, sizeof(unsigned int) * num_segments));
			checkCuda(cudaMalloc((void**)&d_split_pos, sizeof(float) * num_segments));
			checkCuda(cudaMalloc((void**)&d_trikeys, sizeof(uint) * trikeylen));
			checkCuda(cudaMalloc((void**)&d_raykeys, sizeof(uint) * raykeylen));
			checkCuda(cudaMalloc((void**)&d_tsegment_filter_status, sizeof(int) * num_segments * 3));
			checkCuda(cudaMalloc((void**)&d_rsegment_filter_status, sizeof(int) * num_segments * 4));
			cudaMemset(d_tsegment_filter_status, 0, sizeof(int) * num_segments * 3);
			cudaMemset(d_rsegment_filter_status, 0, sizeof(int) * num_segments * 4);
			per_iteration_device_memory += sizeof(uint) * (trikeylen + raykeylen);		// dont count memory allocated for status. We'll remove it shortly

			// call the two kernels
			/*
			cudaEvent_t start, stop;
			cudaStream_t stream1, stream2;
			checkCuda(cudaEventCreate(&start));
			checkCuda(cudaEventCreate(&stop));
			checkCuda(cudaEventRecord(start, 0));
			checkCuda(cudaStreamCreate(&stream1));
			checkCuda(cudaStreamCreate(&stream2));
			*/
			
			Timer tftimer("tftimer");
			tftimer.start();
			NUM_THREADS_PER_BLOCK = 256;
			uint NUM_BLOCKS_Y = (tblocks / rtparams.GRID_DIM_X) + (tblocks % rtparams.GRID_DIM_X != 0);
			uint NUM_BLOCKS_X = rtparams.GRID_DIM_X;
			dim3 t_grid(NUM_BLOCKS_X, NUM_BLOCKS_Y, rtparams.GRID_DIM_Z);
			dim3 t_threads(256, 1, 1);

			parallelTriFilter<<<t_grid, t_threads>>>(d_triangles.v0, d_triangles.v1, d_triangles.v2, 
																  work_level->node_aabbs, work_level->tri_idx, d_trikeys, d_tkeyblocks, 
																  d_tkeyblockStart, work_level->tsegment_sizes, d_trioffsets, 
																  d_tsegment_filter_status, d_split_axis, d_split_pos,
																  trikeylen, depth, tblocks);
			tftimer.stop();
			ctr.tri_filter_time += tftimer.get_ms();

			NUM_BLOCKS_Y = (rblocks / rtparams.GRID_DIM_X) + (rblocks % rtparams.GRID_DIM_X != 0);
			dim3 r_grid(NUM_BLOCKS_X, NUM_BLOCKS_Y, rtparams.GRID_DIM_Z);
			dim3 r_threads(256, 1, 1);

			Timer rftimer("Rf timer");
			rftimer.start();
			parallelRayFilter<<<r_grid, r_threads>>>(d_rays.o, d_rays.d, work_level->node_aabbs, work_level->ray_idx, d_raykeys, 
																  d_rkeyblocks, d_rkeyblockStart, work_level->rsegment_sizes,
																  d_rayoffsets, d_rsegment_filter_status, raykeylen, depth, rblocks);
			rftimer.stop();
			ctr.ray_filter_time += rftimer.get_ms();
			/*
			checkCuda(cudaStreamSynchronize(stream1));
			checkCuda(cudaStreamSynchronize(stream2));
			checkCuda(cudaEventRecord(stop, 0));
			checkCuda(cudaEventSynchronize(stop));
			float elapsed_time = 0;
			checkCuda(cudaEventElapsedTime(&elapsed_time, start, stop));
			ctr.tri_filter_time += elapsed_time;
			*/


			// #3. Sort and Reduce the keys to get individual segment counts
			Timer trisorttimer("trisorttimer");
			trisorttimer.start();
			thrust::sort_by_key(thrust::device_ptr<uint>(d_trikeys), thrust::device_ptr<uint>(d_trikeys) + trikeylen, thrust::device_ptr<int>(work_level->tri_idx));
			trisorttimer.stop();
			ctr.trisortbykey_time += trisorttimer.get_ms();

			Timer raysorttimer("raysorttimer");
			raysorttimer.start();
			thrust::sort_by_key(thrust::device_ptr<uint>(d_raykeys), thrust::device_ptr<uint>(d_raykeys) + raykeylen, thrust::device_ptr<int>(work_level->ray_idx));
			raysorttimer.stop();
			ctr.raysortbykey_time += raysorttimer.get_ms();

#ifdef _DEBUG
			/*------------------------------ DEBUG CODE ---------------------------------*/
			uint* debugtkeys = new uint[trikeylen];
			uint* debugrkeys = new uint[raykeylen];
			checkCuda(cudaMemcpy(debugtkeys, d_trikeys, sizeof(uint)*trikeylen, cudaMemcpyDeviceToHost));
			checkCuda(cudaMemcpy(debugrkeys, d_raykeys, sizeof(uint)*raykeylen, cudaMemcpyDeviceToHost));
			/*------------------------------ END OF DEBUG CODE ---------------------------------*/
#endif

			// #4. Do a reduction step
			uint* d_temp_tri_keys	= NULL;
			uint* d_temp_ray_keys	= NULL;
			uint* d_temp_tri_values = NULL;
			uint* d_temp_ray_values = NULL;
			checkCuda(cudaMalloc((void**)&d_temp_tri_keys, sizeof(uint) * num_segments * 4));
			checkCuda(cudaMalloc((void**)&d_temp_ray_keys, sizeof(uint) * num_segments * 4));
			checkCuda(cudaMalloc((void**)&d_temp_tri_values, sizeof(uint) * num_segments * 4));
			checkCuda(cudaMalloc((void**)&d_temp_ray_values, sizeof(uint) * num_segments * 4));

			cudaMemset(d_temp_tri_keys, 0, sizeof(uint) * num_segments * 4);
			cudaMemset(d_temp_ray_keys, 0, sizeof(uint) * num_segments * 4);
			cudaMemset(d_temp_tri_values, 0, sizeof(uint) * num_segments * 4);
			cudaMemset(d_temp_ray_values, 0, sizeof(uint) * num_segments * 4);
			per_iteration_device_memory += 4 * sizeof(uint) * num_segments;

			Timer triredtimer("tri red timer");
			triredtimer.start();
			thrust::reduce_by_key(thrust::device_ptr<uint>(d_trikeys), thrust::device_ptr<uint>(d_trikeys) + trikeylen, 
								  thrust::device_ptr<uint>(d_trikeys), thrust::device_ptr<uint>(d_temp_tri_keys), 
								  thrust::device_ptr<uint>(d_temp_tri_values));
			triredtimer.stop();
			ctr.trireduction_time += triredtimer.get_ms();

			Timer rayredtimer("ray red timer");
			rayredtimer.start();
			thrust::reduce_by_key(thrust::device_ptr<uint>(d_raykeys), thrust::device_ptr<uint>(d_raykeys) + raykeylen,
								  thrust::device_ptr<uint>(d_raykeys), thrust::device_ptr<uint>(d_temp_ray_keys),
								  thrust::device_ptr<uint>(d_temp_ray_values));
			rayredtimer.stop();
			ctr.rayreduction_time += rayredtimer.get_ms();

#ifdef _DEBUG
			uint* debugtrikeys = new uint[num_segments * 4];
			uint* debugtrivals = new uint[num_segments * 4];
			uint* debugraykeys = new uint[num_segments * 4];
			uint* debugrayvals = new uint[num_segments * 4];
			checkCuda(cudaMemcpy(debugtrikeys, d_temp_tri_keys, sizeof(uint) * num_segments * 4, cudaMemcpyDeviceToHost));
			checkCuda(cudaMemcpy(debugtrivals, d_temp_tri_values, sizeof(uint) * num_segments * 4, cudaMemcpyDeviceToHost));
			checkCuda(cudaMemcpy(debugraykeys, d_temp_ray_keys, sizeof(uint) * num_segments * 4, cudaMemcpyDeviceToHost));
			checkCuda(cudaMemcpy(debugrayvals, d_temp_ray_values, sizeof(uint) * num_segments * 4, cudaMemcpyDeviceToHost));

#endif

			// #5. Calculate child sizes and pivot points
			uint2* d_tsegment_child_pivots = NULL;
			uint2* d_rsegment_child_pivots = NULL;
			uint*  d_tsegment_child_sizes  = NULL;
			uint*  d_rsegment_child_sizes  = NULL;
			uint*  d_trisizes			   = NULL;
			uint*  d_raysizes			   = NULL;
			uint*  d_segment_flags		   = NULL;
			size_t num_child_segments	   = num_segments * 2;
			size_t num_bf_segments		   = 0;
			size_t num_next_level_segments = 0;
			size_t num_next_level_tris	   = 0;
			size_t num_next_level_rays	   = 0;
			checkCuda(cudaMalloc((void**)&d_tsegment_child_pivots, sizeof(uint2) * num_child_segments));
			checkCuda(cudaMalloc((void**)&d_rsegment_child_pivots, sizeof(uint2) * num_child_segments));
			checkCuda(cudaMalloc((void**)&d_tsegment_child_sizes, sizeof(uint) * num_child_segments));
			checkCuda(cudaMalloc((void**)&d_rsegment_child_sizes, sizeof(uint) * num_child_segments));
			checkCuda(cudaMalloc((void**)&d_segment_flags, sizeof(uint) * num_child_segments));
			checkCuda(cudaMalloc((void**)&d_trisizes, sizeof(uint) * num_segments * 4));					// individual left/right/both sizes
			checkCuda(cudaMalloc((void**)&d_raysizes, sizeof(uint) * num_segments * 4));					// individual left/right/both/none sizes
			per_iteration_device_memory += (num_child_segments * (sizeof(uint2) * 2 + sizeof(uint) * 2)) + 2 * sizeof(uint) * num_segments * 4;

			checkCuda(cudaMemset(d_trisizes, 0, sizeof(uint) * num_segments * 4));
			checkCuda(cudaMemset(d_raysizes, 0, sizeof(uint) * num_segments * 4));
			
			NUM_THREADS_PER_BLOCK = 256;
			NUM_BLOCKS = ((num_segments*4) / NUM_THREADS_PER_BLOCK) + ((num_segments*4) % NUM_THREADS_PER_BLOCK != 0);

			Timer crttimer("CRT timer");
			crttimer.start();
			calculateRayTriSizes<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(d_temp_tri_keys, d_temp_tri_values, d_temp_ray_keys, d_temp_ray_values, 
				d_trisizes, d_raysizes, num_segments * 4, num_segments * 4, depth);
			crttimer.stop();
			ctr.misc_time += crttimer.get_ms();

#ifdef _DEBUG
			uint* debugtrisizes = new uint[num_segments * 4];
			uint* debugraysizes = new uint[num_segments * 4];
			checkCuda(cudaMemcpy(debugtrisizes, d_trisizes, sizeof(uint) * num_segments * 4, cudaMemcpyDeviceToHost));
			checkCuda(cudaMemcpy(debugraysizes, d_raysizes, sizeof(uint) * num_segments * 4, cudaMemcpyDeviceToHost));
#endif
			// #5. Now that we have the individual sizes, we can calculate the actual number of nodes 
			// LAUNCH : Each thread calculates one size only.
			NUM_THREADS_PER_BLOCK = 256;
			NUM_BLOCKS = (num_segments / NUM_THREADS_PER_BLOCK) + (num_segments % NUM_THREADS_PER_BLOCK != 0);

			Timer cstimer("cs timer");
			cstimer.start();
			calculateChildSegmentSizes<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(d_trisizes, d_raysizes, work_level->tpivots, work_level->rpivots, 
				d_tsegment_child_sizes, d_rsegment_child_sizes, d_tsegment_child_pivots, d_rsegment_child_pivots, d_segment_flags, 
				num_segments, depth);
			cstimer.stop();
			ctr.misc_time += cstimer.get_ms();

#ifdef _DEBUG
			uint* debug_tsegment_child_sizes	= new uint[num_child_segments];
			uint* debug_rsegment_child_sizes	= new uint[num_child_segments];
			uint* debug_segment_flags			= new uint[num_child_segments];
			uint2* debug_tsegment_child_pivots	= new uint2[num_child_segments];
			uint2* debug_rsegment_child_pivots	= new uint2[num_child_segments];
			checkCuda(cudaMemcpy(debug_tsegment_child_sizes, d_tsegment_child_sizes, sizeof(uint) * num_child_segments, cudaMemcpyDeviceToHost));
			checkCuda(cudaMemcpy(debug_rsegment_child_sizes, d_rsegment_child_sizes, sizeof(uint) * num_child_segments, cudaMemcpyDeviceToHost));
			checkCuda(cudaMemcpy(debug_segment_flags, d_segment_flags, sizeof(uint) * num_child_segments, cudaMemcpyDeviceToHost));
			checkCuda(cudaMemcpy(debug_tsegment_child_pivots, d_tsegment_child_pivots, sizeof(uint2) * num_child_segments, cudaMemcpyDeviceToHost));
			checkCuda(cudaMemcpy(debug_rsegment_child_pivots, d_rsegment_child_pivots, sizeof(uint2) * num_child_segments, cudaMemcpyDeviceToHost));
			
#endif

			Timer misc2timer("Misc 2 timer");
			misc2timer.start();
			num_next_level_segments = thrust::reduce(thrust::device_ptr<uint>(d_segment_flags), thrust::device_ptr<uint>(d_segment_flags) + num_child_segments);
			misc2timer.stop();
			ctr.misc_time += misc2timer.get_ms();
			
			// #6. compute next level idx to output correctly to the locations
			// this is done by exclusive scan on the segment flags
			uint* d_next_level_idx	  = NULL;			// used to output where to write incase of next level segments
			checkCuda(cudaMalloc((void**)&d_next_level_idx, sizeof(uint) * num_child_segments));
			per_iteration_device_memory += sizeof(uint) * num_child_segments;

			Timer misc3timer("Misc3 timer");
			misc3timer.start();
			thrust::exclusive_scan(thrust::device_ptr<uint>(d_segment_flags), thrust::device_ptr<uint>(d_segment_flags) + num_child_segments, thrust::device_ptr<uint>(d_next_level_idx));
			misc3timer.stop();
			ctr.misc_time += misc3timer.get_ms();
		
			// #7. compute next level bounding boxes and pivot points and sizes for segments
			num_bf_segments = num_child_segments - num_next_level_segments;
			AABB* d_next_level_aabbs  = NULL;
			checkCuda(cudaMalloc((void**)&d_next_level_aabbs, sizeof(AABB) * num_next_level_segments));
			per_iteration_device_memory += sizeof(AABB) * num_next_level_segments;

			NUM_BLOCKS = (num_segments / NUM_THREADS_PER_BLOCK) + (num_segments % NUM_THREADS_PER_BLOCK != 0);

			Timer nlatimer("NLA timer");
			nlatimer.start();
			computeNextLevelAabbs<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(work_level->node_aabbs, d_next_level_aabbs, d_segment_flags, 
				d_next_level_idx, num_segments, num_next_level_segments, depth);
			nlatimer.stop();
			ctr.misc_time += nlatimer.get_ms();

			// next level data
			int* d_next_level_tri_idx	= NULL;
			int* d_next_level_ray_idx	= NULL;
			uint2* d_next_level_tpivots = NULL;
			uint2* d_next_level_rpivots = NULL;
			uint2* d_copied_tpivots		= NULL;
			uint2* d_copied_rpivots		= NULL;
			uint* d_next_level_tsizes	= NULL;
			uint* d_next_level_rsizes	= NULL;
			checkCuda(cudaMalloc((void**)&d_next_level_tpivots, sizeof(uint2) * num_next_level_segments));
			checkCuda(cudaMalloc((void**)&d_next_level_rpivots, sizeof(uint2) * num_next_level_segments));
			checkCuda(cudaMalloc((void**)&d_next_level_tsizes, sizeof(uint) * num_next_level_segments));
			checkCuda(cudaMalloc((void**)&d_next_level_rsizes, sizeof(uint) * num_next_level_segments));
			checkCuda(cudaMalloc((void**)&d_copied_tpivots, sizeof(uint2) * num_next_level_segments));
			checkCuda(cudaMalloc((void**)&d_copied_rpivots, sizeof(uint2) * num_next_level_segments));
			per_iteration_device_memory += sizeof(uint2) * num_next_level_segments * 4 + sizeof(uint) * num_next_level_segments * 2;

			Timer copyiftimer1("copy if timer1");
			copyiftimer1.start();
			thrust::copy_if(thrust::device_ptr<uint>(d_tsegment_child_sizes), thrust::device_ptr<uint>(d_tsegment_child_sizes) + num_child_segments,
				thrust::device_ptr<uint>(d_segment_flags), thrust::device_ptr<uint>(d_next_level_tsizes), IsNotBruteForce());
			thrust::copy_if(thrust::device_ptr<uint>(d_rsegment_child_sizes), thrust::device_ptr<uint>(d_rsegment_child_sizes) + num_child_segments,
				thrust::device_ptr<uint>(d_segment_flags), thrust::device_ptr<uint>(d_next_level_rsizes), IsNotBruteForce());
			thrust::copy_if(thrust::device_ptr<uint2>(d_tsegment_child_pivots), thrust::device_ptr<uint2>(d_tsegment_child_pivots) + num_child_segments,
				thrust::device_ptr<uint>(d_segment_flags), thrust::device_ptr<uint2>(d_copied_tpivots), IsNotBruteForce());
			thrust::copy_if(thrust::device_ptr<uint2>(d_rsegment_child_pivots), thrust::device_ptr<uint2>(d_rsegment_child_pivots) + num_child_segments,
				thrust::device_ptr<uint>(d_segment_flags), thrust::device_ptr<uint2>(d_copied_rpivots), IsNotBruteForce());
			copyiftimer1.stop();
			ctr.mem_cpy_time += copyiftimer1.get_ms();

#ifdef _DEBUG
			uint2* debug_next_level_tpivots = new uint2[num_next_level_segments];
			uint2* debug_next_level_rpivots = new uint2[num_next_level_segments];
			uint* debug_next_level_tsizes = new uint[num_next_level_segments];
			uint* debug_next_level_rsizes = new uint[num_next_level_segments];

			checkCuda(cudaMemcpy(debug_next_level_tpivots, d_copied_tpivots, sizeof(uint2) * num_next_level_segments, cudaMemcpyDeviceToHost));
			checkCuda(cudaMemcpy(debug_next_level_rpivots, d_copied_rpivots, sizeof(uint2) * num_next_level_segments, cudaMemcpyDeviceToHost));
			checkCuda(cudaMemcpy(debug_next_level_tsizes, d_next_level_tsizes, sizeof(uint) * num_next_level_segments, cudaMemcpyDeviceToHost));
			checkCuda(cudaMemcpy(debug_next_level_rsizes, d_next_level_rsizes, sizeof(uint) * num_next_level_segments, cudaMemcpyDeviceToHost));
						
#endif
			
			// reduce the size list to determine the sizes of all next level nodes
			dstart = omp_get_wtime();
			num_next_level_tris = thrust::reduce(thrust::device_ptr<uint>(d_next_level_tsizes), thrust::device_ptr<uint>(d_next_level_tsizes) + num_next_level_segments);
			num_next_level_rays = thrust::reduce(thrust::device_ptr<uint>(d_next_level_rsizes), thrust::device_ptr<uint>(d_next_level_rsizes) + num_next_level_segments);
			dend = omp_get_wtime();
			ctr.misc_time += ((dend-dstart) * 1000.0f);
						
			checkCuda(cudaMalloc((void**)&d_next_level_tri_idx, sizeof(int) * num_next_level_tris));
			checkCuda(cudaMalloc((void**)&d_next_level_ray_idx, sizeof(int) * num_next_level_rays));
			per_iteration_device_memory += sizeof(int) * (num_next_level_rays + num_next_level_tris);

			// I need not copy the new pivots. I have to calculate them again.
			uint* d_next_level_tsize_scan = NULL;
			uint* d_next_level_rsize_scan = NULL;
			checkCuda(cudaMalloc((void**)&d_next_level_tsize_scan, sizeof(uint) * num_next_level_segments));
			checkCuda(cudaMalloc((void**)&d_next_level_rsize_scan, sizeof(uint) * num_next_level_segments));
			per_iteration_device_memory += sizeof(uint) * num_next_level_segments * 2;

			Timer misctimer2("Misc timer 2");
			misctimer2.start();
			thrust::exclusive_scan(thrust::device_ptr<uint>(d_next_level_tsizes), thrust::device_ptr<uint>(d_next_level_tsizes) + num_next_level_segments, thrust::device_ptr<uint>(d_next_level_tsize_scan));
			thrust::exclusive_scan(thrust::device_ptr<uint>(d_next_level_rsizes), thrust::device_ptr<uint>(d_next_level_rsizes) + num_next_level_segments, thrust::device_ptr<uint>(d_next_level_rsize_scan));
			misctimer2.stop();
			ctr.misc_time += misctimer2.get_ms();

			NUM_THREADS_PER_BLOCK = 256;
			NUM_BLOCKS = (num_next_level_segments / NUM_THREADS_PER_BLOCK) + (num_next_level_segments % NUM_THREADS_PER_BLOCK != 0);
			Timer ritimer("Reindex timer");
			ritimer.start();
			reindexPivots<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(d_next_level_tsizes, d_next_level_tsize_scan, d_next_level_tpivots, d_next_level_rsizes, d_next_level_rsize_scan, 
				d_next_level_rpivots,num_next_level_segments, depth);
			ritimer.stop();
			ctr.misc_time += ritimer.get_ms();

#ifdef _DEBUG
			/************************** DEBUG ***********************************/

			uint2* debug_tpivots = new uint2[num_next_level_segments];
			uint2* debug_rpivots = new uint2[num_next_level_segments];
			uint*  debug_tsizes = new uint[num_next_level_segments];
			uint*  debug_rsizes = new uint[num_next_level_segments];
			checkCuda(cudaMemcpy(debug_tpivots, d_next_level_tpivots, sizeof(uint2) * num_next_level_segments, cudaMemcpyDeviceToHost));
			checkCuda(cudaMemcpy(debug_rpivots, d_next_level_rpivots, sizeof(uint2) * num_next_level_segments, cudaMemcpyDeviceToHost));
			checkCuda(cudaMemcpy(debug_tsizes, d_next_level_tsizes, sizeof(uint) * num_next_level_segments, cudaMemcpyDeviceToHost));
			checkCuda(cudaMemcpy(debug_rsizes, d_next_level_rsizes, sizeof(uint) * num_next_level_segments, cudaMemcpyDeviceToHost));
			
			/************************** End of Debug Code ***********************/
#endif
			// #8. Scatter data correctly.!
			// Need again blocks and blockStarts kind of lists to correctly determine how much data to copy within each segment
			/** 
			Idea :
				  We launch enough threads per segment in a blocked manner similar to the above ones. So, we know what is the threadId with 
				  respect to its local segment. Now, each segment is given a segment number which can be used to index into the actual list
				  and retrieve the elements and put them in correct location
			*/

			uint* d_tsegmentNo	     = NULL;
			uint* d_tsegmentBlocks   = NULL;
			uint* d_rsegmentNo	     = NULL;
			uint* d_rsegmentBlocks   = NULL;
			uint* d_tsegmentBlockCnt = NULL;
			uint* d_rsegmentBlockCnt = NULL;
			uint* d_tseg_offsets	 = NULL;
			uint* d_rseg_offsets	 = NULL;
			uint* d_triblock_offset	 = NULL;
			uint* d_rayblock_offset	 = NULL;
			checkCuda(cudaMalloc((void**)&d_tsegmentBlockCnt, sizeof(uint) * num_next_level_segments));
			checkCuda(cudaMalloc((void**)&d_rsegmentBlockCnt, sizeof(uint) * num_next_level_segments));
			checkCuda(cudaMalloc((void**)&d_tseg_offsets, sizeof(uint) * num_next_level_segments));
			checkCuda(cudaMalloc((void**)&d_rseg_offsets, sizeof(uint) * num_next_level_segments));
			checkCuda(cudaMalloc((void**)&d_triblock_offset, sizeof(uint) * num_next_level_segments));
			checkCuda(cudaMalloc((void**)&d_rayblock_offset, sizeof(uint) * num_next_level_segments));
			per_iteration_device_memory += sizeof(uint) * num_next_level_segments * 6;

			NUM_THREADS_PER_BLOCK = 1024;
			NUM_BLOCKS = (num_next_level_segments / NUM_THREADS_PER_BLOCK) + (num_next_level_segments % NUM_THREADS_PER_BLOCK != 0);

			Timer tim1("Tim 1");
			tim1.start();
			calculateBlocks<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(d_next_level_tsizes, d_next_level_rsizes, d_tsegmentBlockCnt, 
				d_rsegmentBlockCnt, 256, num_next_level_segments, depth);
			tim1.stop();
			ctr.misc_time += tim1.get_ms();

			// reduce to get count and then fill with data
			size_t tseg_blocks = thrust::reduce(thrust::device_ptr<uint>(d_tsegmentBlockCnt), thrust::device_ptr<uint>(d_tsegmentBlockCnt) + num_next_level_segments);
			size_t rseg_blocks = thrust::reduce(thrust::device_ptr<uint>(d_rsegmentBlockCnt), thrust::device_ptr<uint>(d_rsegmentBlockCnt) + num_next_level_segments);
			Timer misctimer3("Misc timer3");
			misctimer3.start();
			thrust::exclusive_scan(thrust::device_ptr<uint>(d_next_level_tsizes), thrust::device_ptr<uint>(d_next_level_tsizes) + num_next_level_segments, thrust::device_ptr<uint>(d_tseg_offsets));
			thrust::exclusive_scan(thrust::device_ptr<uint>(d_next_level_rsizes), thrust::device_ptr<uint>(d_next_level_rsizes) + num_next_level_segments, thrust::device_ptr<uint>(d_rseg_offsets));
			thrust::exclusive_scan(thrust::device_ptr<uint>(d_tsegmentBlockCnt), thrust::device_ptr<uint>(d_tsegmentBlockCnt) + num_next_level_segments, thrust::device_ptr<uint>(d_triblock_offset));
			thrust::exclusive_scan(thrust::device_ptr<uint>(d_rsegmentBlockCnt), thrust::device_ptr<uint>(d_rsegmentBlockCnt) + num_next_level_segments, thrust::device_ptr<uint>(d_rayblock_offset));
			misctimer3.stop();
			ctr.misc_time += misctimer3.get_ms();
			// allocate space
			checkCuda(cudaMalloc((void**)&d_tsegmentNo, sizeof(uint) * tseg_blocks));
			checkCuda(cudaMalloc((void**)&d_rsegmentNo, sizeof(uint) * rseg_blocks));
			checkCuda(cudaMalloc((void**)&d_tsegmentBlocks, sizeof(uint) * tseg_blocks));
			checkCuda(cudaMalloc((void**)&d_rsegmentBlocks, sizeof(uint) * rseg_blocks));

			Timer tim2("Timer 2");
			tim2.start();
			calculateKeyBlocks<<<num_next_level_segments, NUM_THREADS_PER_BLOCK>>>(d_next_level_tsizes, d_next_level_rsizes, d_tsegmentBlockCnt, 
				d_rsegmentBlockCnt, d_triblock_offset, d_rayblock_offset, d_tsegmentNo, d_tsegmentBlocks, d_rsegmentNo, d_rsegmentBlocks, 
				num_next_level_segments, depth);
			tim2.stop();
			ctr.misc_time += tim2.get_ms();

#ifdef _DEBUG
			/*************************** DEBUG CODE**********************************************/
			uint* tdebug1 = new uint[tseg_blocks];
			uint* tdebug2 = new uint[tseg_blocks];
			uint* rdebug1 = new uint[rseg_blocks];
			uint* rdebug2 = new uint[rseg_blocks];
			checkCuda(cudaMemcpy(tdebug1, d_tsegmentNo, sizeof(uint) * tseg_blocks, cudaMemcpyDeviceToHost));
			checkCuda(cudaMemcpy(rdebug1, d_rsegmentNo, sizeof(uint) * rseg_blocks, cudaMemcpyDeviceToHost));
			checkCuda(cudaMemcpy(tdebug2, d_tsegmentBlocks, sizeof(uint) * tseg_blocks, cudaMemcpyDeviceToHost));
			checkCuda(cudaMemcpy(rdebug2, d_rsegmentBlocks, sizeof(uint) * rseg_blocks, cudaMemcpyDeviceToHost));
			/********************************* END OF DEBUG CODE *********************************/
#endif
			
			// now we can launch so and so threads to retrieve the values
			NUM_THREADS_PER_BLOCK = 256;

			Timer tim3("Timer 3");
			tim3.start();
			triGatherData<<<tseg_blocks, NUM_THREADS_PER_BLOCK>>>(work_level->tri_idx, d_copied_tpivots, d_tsegmentNo, d_tsegmentBlocks, 
				d_next_level_tsizes, d_tseg_offsets, d_next_level_tri_idx, 0, num_next_level_tris, depth);		// set global offset to zero. 
			tim3.stop();
			ctr.mem_cpy_time += tim3.get_ms();

			Timer tim4("Timer 4");
			tim4.start();
			rayGatherData<<<rseg_blocks, NUM_THREADS_PER_BLOCK>>>(work_level->ray_idx, d_copied_rpivots, d_rsegmentNo, d_rsegmentBlocks, 
				d_next_level_rsizes, d_rseg_offsets, d_next_level_ray_idx, 0, num_next_level_rays, depth);
			tim4.stop();
			ctr.mem_cpy_time += tim4.get_ms();

			// do a brute force scatter
			// Brute force scatter can return a list of its allocated size, but which it clears within itself. We are just taking a limit of all data
			size_t brute_alloc_memory = bruteforceScatter(work_level->tri_idx, work_level->ray_idx, work_level->num_tris, work_level->num_rays, 
														  d_tsegment_child_pivots, d_rsegment_child_pivots, d_tsegment_child_sizes, 
														  d_rsegment_child_sizes, d_segment_flags, num_child_segments, pack, rtparams, ctr, depth);
			per_iteration_device_memory += brute_alloc_memory;


#ifdef _DEBUG
			int* debug_next_tri_ids = new int[num_next_level_tris];
			int* debug_next_ray_ids = new int[num_next_level_rays];
			checkCuda(cudaMemcpy(debug_next_tri_ids, d_next_level_tri_idx, sizeof(int) * num_next_level_tris, cudaMemcpyDeviceToHost));
			checkCuda(cudaMemcpy(debug_next_ray_ids, d_next_level_ray_idx, sizeof(int) * num_next_level_rays, cudaMemcpyDeviceToHost));
#endif


			// create new node
			depth++;
			CuLevel* next_work_level		= new CuLevel();
			next_work_level->depth			= depth;
			next_work_level->num_nodes		= num_next_level_segments;
			next_work_level->num_tris		= num_next_level_tris;
			next_work_level->num_rays		= num_next_level_rays;
			next_work_level->node_aabbs		= d_next_level_aabbs;
			next_work_level->tri_idx		= d_next_level_tri_idx;
			next_work_level->ray_idx		= d_next_level_ray_idx;
			next_work_level->tpivots		= d_next_level_tpivots;
			next_work_level->rpivots		= d_next_level_rpivots;
			next_work_level->tsegment_sizes	= d_next_level_tsizes;
			next_work_level->rsegment_sizes	= d_next_level_rsizes;
			working_stack.push(next_work_level);

			// write memory to log
			memlog<<per_iteration_device_memory<<"\n";
			max_dev_memory = std::max(max_dev_memory, per_iteration_device_memory);
			// FREE MEMORY
#ifdef _DEBUG
			// Free all the debug data.
			SAFE_RELEASE(debug_tsegment_child_sizes);
			SAFE_RELEASE(debug_rsegment_child_sizes);
			SAFE_RELEASE(debug_segment_flags);
			SAFE_RELEASE(debug_tsegment_child_pivots);
			SAFE_RELEASE(debug_rsegment_child_pivots);
			SAFE_RELEASE(debugtblock);
			SAFE_RELEASE(debugrblock);
			SAFE_RELEASE(debugtkeyblock);
			SAFE_RELEASE(debugrkeyblock);
			SAFE_RELEASE(debugtkeyblockstart);
			SAFE_RELEASE(debugrkeyblockstart);
			SAFE_RELEASE(debugtkeys);
			SAFE_RELEASE(debugrkeys);
			SAFE_RELEASE(debugtrikeys);
			SAFE_RELEASE(debugtrivals);
			SAFE_RELEASE(debugraykeys);
			SAFE_RELEASE(debugrayvals);
			SAFE_RELEASE(debugtrisizes);
			SAFE_RELEASE(debugraysizes);
			SAFE_RELEASE(debug_next_level_tpivots);
			SAFE_RELEASE(debug_next_level_rpivots);
			SAFE_RELEASE(debug_next_level_tsizes);
			SAFE_RELEASE(debug_next_level_rsizes);
			SAFE_RELEASE(debug_tpivots);
			SAFE_RELEASE(debug_rpivots);
			SAFE_RELEASE(debug_tsizes);
			SAFE_RELEASE(debug_rsizes);
			SAFE_RELEASE(tdebug1);
			SAFE_RELEASE(tdebug2);
			SAFE_RELEASE(rdebug1);
			SAFE_RELEASE(rdebug2);
			SAFE_RELEASE(debug_next_ray_ids);
			SAFE_RELEASE(debug_next_tri_ids);
#endif
			dstart = omp_get_wtime();
			CUDA_SAFE_RELEASE(d_split_pos);
			CUDA_SAFE_RELEASE(d_split_axis);
			CUDA_SAFE_RELEASE(d_tblocks);
			CUDA_SAFE_RELEASE(d_rblocks);
			CUDA_SAFE_RELEASE(d_tkeyblocks);
			CUDA_SAFE_RELEASE(d_rkeyblocks);
			CUDA_SAFE_RELEASE(d_tkeyblockStart);
			CUDA_SAFE_RELEASE(d_rkeyblockStart);
			CUDA_SAFE_RELEASE(d_tsegmentSizes);
			CUDA_SAFE_RELEASE(d_rsegmentSizes);
			CUDA_SAFE_RELEASE(d_tblock_offset);
			CUDA_SAFE_RELEASE(d_rblock_offset);
			CUDA_SAFE_RELEASE(d_trioffsets);
			CUDA_SAFE_RELEASE(d_rayoffsets);
			CUDA_SAFE_RELEASE(d_trikeys);
			CUDA_SAFE_RELEASE(d_raykeys);
			CUDA_SAFE_RELEASE(d_tsegment_filter_status);
			CUDA_SAFE_RELEASE(d_rsegment_filter_status);
			CUDA_SAFE_RELEASE(d_temp_tri_keys);
			CUDA_SAFE_RELEASE(d_temp_ray_keys);
			CUDA_SAFE_RELEASE(d_temp_tri_values);
			CUDA_SAFE_RELEASE(d_temp_ray_values);
			CUDA_SAFE_RELEASE(d_tsegment_child_pivots);
			CUDA_SAFE_RELEASE(d_rsegment_child_pivots);
			CUDA_SAFE_RELEASE(d_tsegment_child_sizes);
			CUDA_SAFE_RELEASE(d_rsegment_child_sizes);
			CUDA_SAFE_RELEASE(d_trisizes);
			CUDA_SAFE_RELEASE(d_raysizes);
			CUDA_SAFE_RELEASE(d_segment_flags);
			CUDA_SAFE_RELEASE(d_next_level_tsize_scan);
			CUDA_SAFE_RELEASE(d_next_level_rsize_scan);
			CUDA_SAFE_RELEASE(d_tsegmentNo);
			CUDA_SAFE_RELEASE(d_tsegmentBlocks);
			CUDA_SAFE_RELEASE(d_rsegmentNo);
			CUDA_SAFE_RELEASE(d_rsegmentBlocks);
			CUDA_SAFE_RELEASE(d_tsegmentBlockCnt);
			CUDA_SAFE_RELEASE(d_rsegmentBlockCnt);
			CUDA_SAFE_RELEASE(d_tseg_offsets);
			CUDA_SAFE_RELEASE(d_rseg_offsets);
			CUDA_SAFE_RELEASE(d_copied_tpivots);
			CUDA_SAFE_RELEASE(d_copied_rpivots);
			CUDA_SAFE_RELEASE(d_triblock_offset);
			CUDA_SAFE_RELEASE(d_rayblock_offset);
			delete work_level;
			dend = omp_get_wtime();
			ctr.misc_time += ((dend - dstart) * 1000.0f);
		}

		// dump the contents of pack
#ifdef _DEBUG
		thrust::host_vector<int> debug_tribuffer(pack->tri_buffer_occupied);
		thrust::host_vector<int> debug_raybuffer(pack->ray_buffer_occupied);
		thrust::copy(pack->buffered_ray_idx.begin(), pack->buffered_ray_idx.begin() + pack->ray_buffer_occupied, debug_raybuffer.begin());
		thrust::copy(pack->buffered_tri_idx.begin(), pack->buffered_tri_idx.begin() + pack->tri_buffer_occupied, debug_tribuffer.begin());
		std::ofstream ofile("dump_fully_parallel_cuda.txt");
		ofile<<"Block Cnt : "<<pack->blockCnt<<"\n"<<"RBF occ : "<<pack->ray_buffer_occupied<<"\n"<<"TBF occ : "<<pack->tri_buffer_occupied<<"\n";
		ofile<<"BStart : "<<pack->bstart<<"\n";
		ofile.close();
		
		ofile.open("fully_parallel_ray_idx_cuda.txt");
		ofile<<"-------------------------------\n";
		ofile<<"Dumping Buffered Ray Ids : \n";
		for(size_t i = 0; i < pack->ray_buffer_occupied; i++) {
			ofile<<debug_raybuffer[i]<<"\n";
		}
		ofile.close();
		ofile.open("fully_parallel_tri_idx_cuda.txt");
		ofile<<"--------------------------------\n";
		ofile<<"Dumping Buffered Tri Ids : \n";
		for(size_t i = 0; i < pack->tri_buffer_occupied; i++) {
			ofile<<debug_tribuffer[i]<<"\n";
		}
		ofile.close();
#endif
		memoryusage(max_dev_memory);
		// complete the render
		if(pack->num_segments > 0) {
			completeBruteForceModified(*pack, d_triangles, d_rays, rtparams, ctr);
		}

		// copy the results back
		// copy the data back to the device
		dstart = omp_get_wtime();
		thrust::copy(pack->dev_ray_maxts.begin(), pack->dev_ray_maxts.end(), thrust::device_ptr<float>(d_maxts));
		thrust::copy(pack->dev_hitids.begin(), pack->dev_hitids.end(), thrust::device_ptr<int>(d_hitids));
		dend = omp_get_wtime();
		ctr.other_time1 += ((dend-dstart) * 1000.0f);

		delete pack;
		if(!working_stack.empty()) {
			CuLevel* level = working_stack.top();
			working_stack.pop();
			if(level != NULL) delete level;
		}
}