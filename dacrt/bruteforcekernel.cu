#include <dacrt/dacrt.h>
#include <util/cutimer.h>

#define TRI_SHARED_MEMORY_SPACE 256

extern "C"
__global__ 
void segmentedBruteForce(RayArray rays, TriangleArray triangles, 
	int* buffered_ray_ids, int ray_buffer_occupied,
	int* buffered_tri_ids, int tri_buffer_occupied,
	int* ray_segment_sizes, int* tri_segment_sizes,
	int* ray_segment_start,															// this would be an exclusive scan array that gives us where each segment starts
	int* tri_segment_start,															// the same applies here
	int num_segments,		// data not required.!! REMOVE
	float* maxts,																	
	int* hitids,																	
	int num_threads_launched,														// DEBUG INFO - Not required REMOVE
	int num_blocks_launched															// DEBUG INFO - Not required REMOVE
	) {

		// we have to find a way to count number of segments this block of threads will be doing
		// then we have to split the workingset size so that each thread group within the block can work on its copy of shared memory only
		// update accordingly
		// this ensures we dont lose parallelism at all
		
		__shared__ float3 v0[TRI_SHARED_MEMORY_SPACE];
		__shared__ float3 v1[TRI_SHARED_MEMORY_SPACE];
		__shared__ float3 v2[TRI_SHARED_MEMORY_SPACE];
		__shared__ int triangle_ids[TRI_SHARED_MEMORY_SPACE];
		__shared__ int num_tris_to_process;
		__shared__ int num_rays_to_process;
		__shared__ int ray_offset;
		__shared__ int tri_offset;
		__shared__ int tri_batches_to_process;
		__shared__ int ray_batches_to_process;

		// we can put the FLT_MAX and hitid variables also inside shared variables
		__shared__ float fmaxts[TRI_SHARED_MEMORY_SPACE];
		__shared__ float hitid[TRI_SHARED_MEMORY_SPACE];
		
		//int tidx = threadIdx.x + blockIdx.x * blockDim.x;
		// we might have an unequal number of rays/triangles for threads to process
		/****************************************************************************/
		// NOTE: We are launching one block of fixed size threads for each segment
		//       This simplifies the working model a bit, but there might be cases where the work load is not uniform. But that is in the next step.
		
		// these data are all common to all the blocks
		if(threadIdx.x == 0) {
			num_tris_to_process		= tri_segment_sizes[blockIdx.x];			
			num_rays_to_process		= ray_segment_sizes[blockIdx.x];
			ray_offset				= ray_segment_start[blockIdx.x];
			tri_offset				= tri_segment_start[blockIdx.x];

			tri_batches_to_process	= (num_tris_to_process/blockDim.x) + (num_tris_to_process % blockDim.x != 0);
			ray_batches_to_process	= (num_rays_to_process/blockDim.x) + (num_rays_to_process % blockDim.x != 0);
		}
		int	ridx					= 0;
		int ray_batch				= 0;
		int this_time_rays			= 0;
		int rayid;
		__syncthreads();
		// I can put these variables outside also..~
		// we do a batch wise loading of rays which in turn do their operation batch wise on triangles
		do {
			// these two variables below are per ray variables
			// THey have to be refreshed every time a new ray is picked up for processing.
			// Hence I've put them inside this loop and not outside.
			//float fmaxts				= FLT_MAX;
			//int hitid					= -1;
			fmaxts[threadIdx.x]			= FLT_MAX;
			hitid[threadIdx.x]			= -1;

			if(num_rays_to_process - ridx > blockDim.x) {
				this_time_rays = blockDim.x;
				ridx += blockDim.x;
			} else {
				this_time_rays = num_rays_to_process - ridx;
			}

			// now do a batch load of triangles
			int tid = 0;
			int tri_batch = 0;
			int this_time_tris = 0;
			int temp;
			do {
				if(num_tris_to_process - tid > TRI_SHARED_MEMORY_SPACE) {
					this_time_tris	= TRI_SHARED_MEMORY_SPACE;
					tid				+= TRI_SHARED_MEMORY_SPACE;
				} else {
					this_time_tris = num_tris_to_process - tid;
				}

				if(threadIdx.x < this_time_tris) {
					int triid = buffered_tri_ids[tri_offset + threadIdx.x + tri_batch * TRI_SHARED_MEMORY_SPACE];
					v0[threadIdx.x] = triangles.v0[triid];
					v1[threadIdx.x] = triangles.v1[triid];
					v2[threadIdx.x] = triangles.v2[triid];
					triangle_ids[threadIdx.x] = triid;
				}

				__syncthreads();

				// Note: the logic behind using this 'if' condition here and not outside block is that, when we might have total rays to be less than
				//		 number of threads, we might have a large number of triangles to load. So we just let all threads take some part in loading
				//		 as many number of triangles as possible, while intersection is carried out only with the required rays
				if(threadIdx.x < this_time_rays) {
					temp = ray_offset + threadIdx.x + ray_batch * blockDim.x;		// for writing into final global array
					rayid = buffered_ray_ids[temp];		
					Ray ir(rays.o[rayid], rays.d[rayid]);;
					for(int t = 0; t < this_time_tris; t++) {
						Triangle it(v0[t], v1[t], v2[t]);
						double u, v, xt;
						if(rayIntersect<double>(it, ir, u, v, xt)) {
							if(xt > 0 && (float)xt < fmaxts[threadIdx.x]) {
								fmaxts[threadIdx.x] = xt;
								// calculating the id of the hit variable?
								hitid[threadIdx.x] = triangle_ids[t];		
							}
						}
					}
					// now update the global buffered_hit_id and buffered_maxts array
					// NOTE: logic is also wrong here.!!! we have to find the correct threadid location to put our variables.
					//maxts[tidx] = fmaxts;
					//hitids[tidx] = hitid;
				}
				__syncthreads();
				tri_batch++;
			}while(tri_batch < tri_batches_to_process);
			// update rays and their global ids
			if(threadIdx.x < this_time_rays) {
				maxts[temp]  = fmaxts[threadIdx.x];
				hitids[temp] = hitid[threadIdx.x];
			}
			__syncthreads();
			ray_batch++;
		} while(ray_batch < ray_batches_to_process);
}

// update min values in the global array
extern "C"
__global__ void updateMinKernel(int* ray_id, float* min_hits, int* minhit_ids, float* global_min, int* global_hits, int num_rays) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < num_rays) {
		int rayid	= ray_id[tid];
		float xhit	= min_hits[tid];
		int hitid	= minhit_ids[tid];
		if(xhit < global_min[rayid]) {
			global_min[rayid] = xhit;
			global_hits[rayid] = hitid;
		}
	}
}

extern "C"
__global__ void dacrtBruteForce(TriangleArray dev_triangles, int num_triangles, RayArray dev_rays, int num_rays,
	int* tri_idx_array,				// this array will have the triangles
	int tricnt,						// number of triangles to be considered
	int* ray_idx_array,				// ray ids
	int raycnt,						// number of rays to be considered
	float* maxts,					// maxts value
	int* hitids
	) {

		__shared__ float3 v0[TRI_SHARED_MEMORY_SPACE];
		__shared__ float3 v1[TRI_SHARED_MEMORY_SPACE];
		__shared__ float3 v2[TRI_SHARED_MEMORY_SPACE];
				
		int tidx = threadIdx.x + blockDim.x * blockIdx.x;
		float fmaxts = FLT_MAX;
		int hitid = -1;
		int to_load = tricnt / TRI_SHARED_MEMORY_SPACE + (tricnt % TRI_SHARED_MEMORY_SPACE != 0);
		int loaded = 0;
		int idx = 0;
		int this_time_triangles = 0;
		do {
			// first load the first TRI_SHARED_MEMORY_SPACE triangles or less
			if((tricnt - idx) >= TRI_SHARED_MEMORY_SPACE) { this_time_triangles = TRI_SHARED_MEMORY_SPACE; idx += TRI_SHARED_MEMORY_SPACE; }
			else this_time_triangles = tricnt - idx;

			if(threadIdx.x < this_time_triangles) {
				v0[threadIdx.x] = dev_triangles.v0[tri_idx_array[threadIdx.x + loaded * TRI_SHARED_MEMORY_SPACE]];		// move to the next 512
				v1[threadIdx.x] = dev_triangles.v1[tri_idx_array[threadIdx.x + loaded * TRI_SHARED_MEMORY_SPACE]];
				v2[threadIdx.x] = dev_triangles.v2[tri_idx_array[threadIdx.x + loaded * TRI_SHARED_MEMORY_SPACE]];
			}

			__syncthreads();
				
			// now perform ray intersection
			if(tidx < raycnt) {
				double u, v, xt;
				Ray ir(dev_rays.o[ray_idx_array[tidx]], dev_rays.d[ray_idx_array[tidx]]);
				for(int t = 0; t < this_time_triangles; t++) {
					if(rayIntersect<double>(Triangle(v0[t], v1[t], v2[t]), ir, u, v, xt)) {
						if(xt > 0 && static_cast<float>(xt) < fmaxts) {
							fmaxts = static_cast<float>(xt);
							hitid = tri_idx_array[t + loaded * TRI_SHARED_MEMORY_SPACE];			
						}
					}
				}
			}
			__syncthreads();
			// go to next segment
			// NOTE: This was a bug. Keeping the loaded variable increment within the if condition, made all other threads go in loop forever.!!! because they
			//       never incremented the value.!!
			loaded++;
		} while(loaded < to_load);
		
		// update the arrays
		if(tidx < raycnt) {
			//maxts[tidx] = fmaxts;
			//hitids[tidx] = hitid;
			if(maxts[ray_idx_array[tidx]] > fmaxts && hitid != -1) {
				maxts[ray_idx_array[tidx]] = fmaxts;
				hitids[ray_idx_array[tidx]] = hitid;
			}
		}
			
		__syncthreads();		// expensive? check correctness and remove
}

// add a modified segmentedBruteForce kernel
extern "C"
__global__ void modifiedSegmentedBruteForce(RayArray rays, TriangleArray triangles, int* buffered_ray_ids, int ray_buffer_occupied, int* buffered_tri_ids, int tri_buffer_occupied,
	int* ray_segment_sizes, int* tri_segment_sizes, int* ray_segment_start, int* tri_segment_start, 
	int* segment_no, int* blockStart,
	float* maxts, int* hitids
	) {
		
		__shared__ float3 v0[TRI_SHARED_MEMORY_SPACE];
		__shared__ float3 v1[TRI_SHARED_MEMORY_SPACE];
		__shared__ float3 v2[TRI_SHARED_MEMORY_SPACE];
		__shared__ int triangle_ids[TRI_SHARED_MEMORY_SPACE];
		__shared__ float fmaxts[TRI_SHARED_MEMORY_SPACE];
		__shared__ float hitid[TRI_SHARED_MEMORY_SPACE];
		__shared__ int num_tris_to_process;
		__shared__ int num_rays_to_process;
		__shared__ int ray_offset;
		__shared__ int tri_offset;
		__shared__ int tri_batches_to_process;
		__shared__ int blockNo;
		__shared__ int whichBlock;
		__shared__ int tid;
		__shared__ int this_time_tris;
		__shared__ int threadId_within_segment[TRI_SHARED_MEMORY_SPACE];
		__shared__ int temp[TRI_SHARED_MEMORY_SPACE];
		__shared__ int tri_batch[TRI_SHARED_MEMORY_SPACE];
		
		if(threadIdx.x == 0) {
			blockNo = segment_no[blockIdx.x];		// load which segment you are
			whichBlock = blockStart[blockIdx.x];
			tid = 0;
			num_tris_to_process		= tri_segment_sizes[blockNo];			
			num_rays_to_process		= ray_segment_sizes[blockNo];
			ray_offset				= ray_segment_start[blockNo];		// this where this block's threads actually start
			tri_offset				= tri_segment_start[blockNo];

			tri_batches_to_process	= (num_tris_to_process/blockDim.x) + (num_tris_to_process % blockDim.x != 0);
			//ray_batches_to_process	= (num_rays_to_process/blockDim.x) + (num_rays_to_process % blockDim.x != 0);
		}
		__syncthreads();

		//int threadId_within_segment = threadIdx.x + whichBlock * 256;
		threadId_within_segment[threadIdx.x] = threadIdx.x + whichBlock * TRI_SHARED_MEMORY_SPACE;			// this is supposed to be whichBlock * num_Rays_per_block
		// this id should be less than each segment's total rays to be handled
		//if(threadId_within_segment < num_rays_to_process) {

			// here we do batch loading of triangles and process them.
			fmaxts[threadIdx.x] = FLT_MAX;
			hitid[threadIdx.x] = -1;

			// QUESTION? can all these be shared variables?
			//__shared__ int tid = 0;
			//int tri_batch = 0;
			tri_batch[threadIdx.x] = 0;
			//int this_time_tris = 0;
			//int temp;
			do {
				if(threadIdx.x == 0) {
					if(num_tris_to_process - tid > TRI_SHARED_MEMORY_SPACE) {
						this_time_tris	= TRI_SHARED_MEMORY_SPACE;
						tid				+= TRI_SHARED_MEMORY_SPACE;
					} else {
						this_time_tris = num_tris_to_process - tid;
					}
				}
				__syncthreads();

				// load the triangles
				if(threadIdx.x < this_time_tris) {
					int triid = buffered_tri_ids[tri_offset + threadIdx.x + tri_batch[threadIdx.x] * TRI_SHARED_MEMORY_SPACE];
					v0[threadIdx.x] = triangles.v0[triid];
					v1[threadIdx.x] = triangles.v1[triid];
					v2[threadIdx.x] = triangles.v2[triid];
					triangle_ids[threadIdx.x] = triid;
				}
				__syncthreads();

				if(threadId_within_segment[threadIdx.x] < num_rays_to_process) {
					temp[threadIdx.x] = ray_offset + threadId_within_segment[threadIdx.x];			// starting ray + within this segment which ray
					int rid = buffered_ray_ids[temp[threadIdx.x]];
					Ray ir(rays.o[rid], rays.d[rid]);
					for(int t = 0; t < this_time_tris; t++) {
						Triangle it(v0[t], v1[t], v2[t]);
						double u, v, xt;
						if(rayIntersect<double>(it, ir, u, v, xt)) {
							if(xt > 0 && (float)xt < fmaxts[threadIdx.x]) {
								fmaxts[threadIdx.x] = xt;
								// calculating the id of the hit variable?
								hitid[threadIdx.x] = triangle_ids[t];		
							}
						}
					}
				}
				__syncthreads();
				// increment the batch
				tri_batch[threadIdx.x]++;
			} while(tri_batch[threadIdx.x] < tri_batches_to_process);
		//}
		
			// this condition takes care of extra rays launched as a result of bringing the count to arbitrary multiple
			// num_rays_to_process is within a segment.
			if(threadId_within_segment[threadIdx.x] < num_rays_to_process) {			
				maxts[temp[threadIdx.x]]  = fmaxts[threadIdx.x];
				hitids[temp[threadIdx.x]] = hitid[threadIdx.x];
			}
			__syncthreads();
			
}

// complete the incomplete segments
// This method can be called from other segment approaches. Hence external linkage
extern "C"
void dacrtCompleteRender(ParallelPack& pack, TriangleArray& dev_triangles, RayArray& dev_rays, DacrtRunTimeParameters& rtparams, Counters& ctr) {
	thrust::device_vector<int> ray_segment_start(pack.num_segments);
	thrust::device_vector<int> tri_segment_start(pack.num_segments);
	thrust::exclusive_scan(pack.tri_segment_sizes.begin(), pack.tri_segment_sizes.begin() + pack.num_segments, tri_segment_start.begin());
	thrust::exclusive_scan(pack.ray_segment_sizes.begin(), pack.ray_segment_sizes.begin() + pack.num_segments, ray_segment_start.begin());

	int num_blocks = pack.num_segments;
	int num_threads_per_block = rtparams.NUM_RAYS_PER_BLOCK;
				
	Timer seg_brute_timer("SegmentedBruteForce Timer");
				
	seg_brute_timer.start();
	segmentedBruteForce<<<num_blocks, num_threads_per_block>>>(dev_rays, dev_triangles, thrust::raw_pointer_cast(&pack.buffered_ray_idx[0]), 
			pack.ray_buffer_occupied, thrust::raw_pointer_cast(&pack.buffered_tri_idx[0]), pack.tri_buffer_occupied, 
			thrust::raw_pointer_cast(&pack.ray_segment_sizes[0]), thrust::raw_pointer_cast(&pack.tri_segment_sizes[0]), 
			thrust::raw_pointer_cast(&ray_segment_start[0]), thrust::raw_pointer_cast(&tri_segment_start[0]), 
			pack.num_segments, thrust::raw_pointer_cast(&pack.buffered_ray_maxts[0]),
			thrust::raw_pointer_cast(&pack.buffered_ray_hitids[0]), num_threads_per_block * num_blocks, num_blocks);
	
	seg_brute_timer.stop();
	ctr.brute_force_time += seg_brute_timer.get_ms();

	Timer seg_sort_timer("Seg Sort Timer");
	seg_sort_timer.start();
	thrust::sort_by_key(pack.buffered_ray_idx.begin(), pack.buffered_ray_idx.begin() + pack.ray_buffer_occupied,
		thrust::make_zip_iterator(thrust::make_tuple(pack.buffered_ray_maxts.begin(), pack.buffered_ray_hitids.begin())));
	
	seg_sort_timer.stop();
	ctr.seg_sort_time += seg_sort_timer.get_ms();
				
	// now we have to reduce according to the key, which is the ray id
	static thrust::device_vector<int> ray_idx(rtparams.BUFFER_SIZE);
	static thrust::device_vector<float> ray_maxts(rtparams.BUFFER_SIZE);
	static thrust::device_vector<int> ray_hitids(rtparams.BUFFER_SIZE);
	static thrust::equal_to<int> pred;
				
	typedef thrust::device_vector<int>::iterator iter;
	typedef thrust::device_vector<float>::iterator fiter;
	typedef thrust::zip_iterator<thrust::tuple<fiter, iter> > zippy;
	thrust::pair<iter, zippy> minend;
				
	MinHitFunctor<thrust::tuple<float, int> > min_hit_functor;

	Timer reduction_timer("Reduction Timer");
	reduction_timer.start();
				
	minend = thrust::reduce_by_key(pack.buffered_ray_idx.begin(), pack.buffered_ray_idx.begin() + pack.ray_buffer_occupied,
					thrust::make_zip_iterator(thrust::make_tuple(pack.buffered_ray_maxts.begin(), pack.buffered_ray_hitids.begin())),
					ray_idx.begin(), thrust::make_zip_iterator(thrust::make_tuple(ray_maxts.begin(), ray_hitids.begin())),
					pred,
					min_hit_functor);
	reduction_timer.stop();
	ctr.reduction_time += reduction_timer.get_ms();
				
			
	// now we can update our global max_ts and hitid array
	int num_valid_keys = minend.first - ray_idx.begin();
	num_threads_per_block = 512;
	num_blocks = num_valid_keys / num_threads_per_block + (num_valid_keys % num_threads_per_block != 0);
				
	Timer update_min_timer("Update Min Timer");
	update_min_timer.start();
	updateMinKernel<<<num_blocks, num_threads_per_block>>>(thrust::raw_pointer_cast(&ray_idx[0]), thrust::raw_pointer_cast(&ray_maxts[0]), thrust::raw_pointer_cast(&ray_hitids[0]),
				thrust::raw_pointer_cast(&pack.dev_ray_maxts[0]), thrust::raw_pointer_cast(&pack.dev_hitids[0]), num_valid_keys);
	update_min_timer.stop();
	ctr.update_min_time += update_min_timer.get_ms();
					
	// reset the counters
	ray_idx.clear();
	ray_maxts.clear();
	ray_hitids.clear();
	pack.buffered_ray_idx.clear();
	pack.buffered_tri_idx.clear();
	pack.tri_segment_sizes.clear();
	pack.ray_segment_sizes.clear();
	pack.segment_ids.clear();
							
	pack.ray_buffer_occupied = 0;
	pack.tri_buffer_occupied = 0;
	pack.num_segments = 0;
}