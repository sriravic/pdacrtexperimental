/**
GPU algorithms making use of SoA rather than AoS.
Both spatial and object partitioning codes to be done here only.
*/

#include <dacrt/dacrt.h>
#include <util/cutimer.h>
#include <util/util.h>

#define MAX_BLOCK_THREADS 256

// assuming 256 triangles per block, each handled by a thread
__global__ void triFilteringAos(Triangle3* triangles, int* tri_ids, int num_tris, float3 bmin, float3 bmax, int* occupy) {

	// each triangle has 36 bytes of vertex data as well as 24 bytes of edge data plus 4 bytes of pad data
	// so we can service 64 bytes of data if 
	// So, in one warp the threads can read 2 bytes of data each.

	/// NOTE: for now, we are going with each thread reading 4 bytes of data
	/**
			  so group of 16 threads within a block can read one element. in one go, we can read up to 2 elements per warp in probably 2 cycles at the max.
			  So, assuming 256 threads, we have 8 warps, so that brings us 16 elements per block. brought to us in about 32 cycles at the max.
			  So, 
	*/

	// we store 9 vertices per triangle, so we can process 256 triangles per iteration of candidate loop
	// For now we store all data
	__shared__ short shared_data[18 * MAX_BLOCK_THREADS];
	__shared__ float3 centroid; 
	__shared__ float3 extents;
	//__shared__ int to_read;
		
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(threadIdx.x == 0) {
		centroid = (bmin + bmax) * 0.5f;
		extents = bmax - bmin;
		
	}
	__syncthreads();

	//int to_read = (num_tris / 256) + (num_tris % 256 != 0);
	//int extra_elements = num_tris % 256;

	/**
	Why are we even putting the values inside shared memory in the first place? Well, for starters, once we have say 256 triangles in SM,
	we can calculate their results and update in global list in 
	*/

	// This is the load iteration.! we have to output the results of triangle filtering here itself.!!!
	//for(int i = 0; i < to_read; i++) {
		// each element is read by 16 threads (each thread reads 4 bytes of data. So 64/4 = 16 threads);
		// we need 16 cycles to read 256 elements. After each iteration we would have read 16 elements
		/// NOTE: changing number of threads per block increases the number of elements that can be read as a part of read cycle. so change appropriately.
		for(int read_cycle = 0; read_cycle < 32; read_cycle++) {
			// int element_id = threadIdx.x % 16 + read_cycle * 16;
			// every read cycle reads 8 elements.! (in case of 32 threads read one element read structure)
			int element_id = threadIdx.x / 32 + read_cycle * 8 + blockIdx.x * 256;
			// by imposing this element_id condition, I am guaranteed that no extra data is read. I need not compute
			// how many triangles or more data i have to load in each iteration. 
			if(element_id < num_tris) {
				// we get edge data and pad data for free.! we dont use them anyways here
				int tri_id = tri_ids[element_id];

				// All threads in a warp now read the entire 64 bytes of the triangle.
				// float tdata = triangles[tri_id].data.d[threadIdx.x % 16]
				short tdata = triangles[tri_id].data.d[threadIdx.x % 32];			
				// now only those threads which picked up meaningful data have to update the shared memory
				
				//int warp = threadIdx.x / 32;
				//int half_warp = (threadIdx.x % 32) / 16;
				int id_within_warp = threadIdx.x % 32;
				// only those id_within_warp is 9 actually has triangle vertex data
				// if(id_within_warp < 9) {
				if(id_within_warp < 18) {
					// element_id tells which triangle this half warp(16) is responsible for
					// so we put data as such.
					// every read cycle pushes 18 * 8 elements (18 elements per warp and so we have 8 active warps)
					// so 18 * 8 = 144
					//shared_data[element_id * 18 + id_within_warp + read_cycle * 144] = tdata;
					shared_data[((element_id % 256) * 18) + id_within_warp] = tdata;
					
				}
			}
			/// NOTE: Do we need a sync threads here? I dont think so. Since its just write of shared data only, we can skip this I suppose.
			//__syncthreads();
		}
		//__syncthreads();	
		// now for each triangle we have to calculate occupancy for each triangle
		// DEBUG : The pointer manipulation was wrong. when we increment by x, 
		//the implicit type of pointe is used. So use 2 because 2 shorts = 1 float
		if(tid < num_tris) {
			float triverts[3][3] = {{*((float*)&shared_data[threadIdx.x * 18]),      *((float*)&shared_data[threadIdx.x * 18 + 2]),  *((float*)&shared_data[threadIdx.x * 18 + 4])},
								   {*((float*)&shared_data[threadIdx.x * 18 + 6]), *((float*)&shared_data[threadIdx.x * 18 + 8]), *((float*)&shared_data[threadIdx.x * 18 + 10])},
								   {*((float*)&shared_data[threadIdx.x * 18 + 12]), *((float*)&shared_data[threadIdx.x * 18 + 14]), *((float*)&shared_data[threadIdx.x * 18 + 16])}};
			float boxhalf[3] = {extents.x * 0.5f, extents.y * 0.5f, extents.z * 0.5f};
			float boxcenter[3] = {centroid.x, centroid.y, centroid.z};
			int o = triBoxOverlap(boxcenter, boxhalf, triverts);
			// update the global list
			occupy[tid] = o;
		}
		// load the next batch of triangles
		//__syncthreads();
	//}

}

// assuming the same 256 rays per block
/// NOTE: if we can reduce the register usage by 2, we can increase our performance
__global__ void rayFilteringAos(Ray4* rays, int* ray_ids, int num_rays, float4 bmin, float4 bmax, int* occupy) {

	// since the ray data structure is of size 32 bytes, each thread now has to read one byte
	// each block will have 256 threads
	__shared__ char shared_data[32 * MAX_BLOCK_THREADS];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	// one warp reads one ray value per cycle. 
	// 256 threads has 8 warps. 8 warps can read 256 ray values in 32 cycles
	// change cycle value if you read more data
	for(int read_cycle = 0; read_cycle < 32; read_cycle++) {
		int element_id = threadIdx.x / 32 + read_cycle * 8 + blockIdx.x * 256;
		if(element_id < num_rays) {
				int ray_id = ray_ids[element_id];
				char rdata = rays[ray_id].data.d[threadIdx.x % 32];			
				int id_within_warp = threadIdx.x % 32;
				// our shared memory at the max can hold only 256 values.!! be careful.
				shared_data[((element_id % 256) * 32) + id_within_warp] = rdata;
		}
	}

	// now perform ray box test
	if(tid < num_rays) {
		float4 origin = make_float4(*((float*)&shared_data[threadIdx.x * 32]),     *((float*)&shared_data[threadIdx.x * 32 + 4]), 
								    *((float*)&shared_data[threadIdx.x * 32 + 8]), *((float*)&shared_data[threadIdx.x * 32 + 12]));
		float4 direction = make_float4(*((float*)&shared_data[threadIdx.x * 32 + 16]), *((float*)&shared_data[threadIdx.x * 32 + 20]), 
								       *((float*)&shared_data[threadIdx.x * 32 + 24]), *((float*)&shared_data[threadIdx.x * 32 + 28]));

		AABB4 bbox(bmin, bmax);
		Ray4 ray(origin, direction);
		float thit;
		bool occ = bbox.rayIntersect(ray, thit);
		occupy[tid] = (int)occ;
	}
}

// The normal brute force algorithm. The same way we did 
__global__ void bruteForceAosSegmented(TriangleArrayAos triangles, int num_triangles, RayArrayAos rays, int num_rays, 
	int* tri_idx_array, int tpivot, int* ray_idx_array, int rpivot,
	int* ray_segment_sizes, int* tri_segment_sizes, int* ray_segment_start, int* tri_segment_start, 
	int* segmentNo, int* blockStart,
	float* maxts, int* hitids) {

		// the same way we load rays and triangles.
		// allocate shared memory for both rays and triangles
		__shared__ char  shared_ray_data[32 * MAX_BLOCK_THREADS];
		__shared__ short shared_tri_data[18 * MAX_BLOCK_THREADS];
		__shared__ int	 triangle_ids[256];
		__shared__ int   num_tris_to_process;
		__shared__ int   num_rays_to_process;
		__shared__ int   ray_offset;
		__shared__ int   tri_offset;
		__shared__ int   tri_batches_to_process;
		__shared__ int   blockNo;
		__shared__ int   whichBlock;
		__shared__ int   tid;										// this is not thread id but rather triangle id
		__shared__ int   this_time_tris;
		__shared__ int   threadId_within_segment[256];
		__shared__ int   temp[256];
		__shared__ int   tri_batch[256];
		__shared__ float fmaxts[256];
		__shared__ float hitid[256];


		int threadid = threadIdx.x + blockIdx.x * blockDim.x;
		if(threadIdx.x == 0) {
			blockNo					= segmentNo[blockIdx.x];		// load which segment you are
			whichBlock				= blockStart[blockIdx.x];
			tid						= 0;
			num_tris_to_process		= tri_segment_sizes[blockNo];			
			num_rays_to_process		= ray_segment_sizes[blockNo];
			ray_offset				= ray_segment_start[blockNo];	// this where this block's threads actually start
			tri_offset				= tri_segment_start[blockNo];
			tri_batches_to_process	= (num_tris_to_process/blockDim.x) + (num_tris_to_process % blockDim.x != 0);
		}
		__syncthreads();

		/**
		We'll follow the modified brute force kernel approach here. So, we'll launch as many rays threads as possible inside blocks of 256 threads each.
		Each block will batch load the triangles. 
		*/

		//#1. Load all 256 or less rays.
		for(int read_cycle = 0; read_cycle < 32; read_cycle++) {
			int element_id = threadIdx.x / 32 + read_cycle * 8 + blockIdx.x * 256;
			if(element_id < num_rays_to_process) {
				// calculate offsetted ray_id
				int ray_id = ray_idx_array[element_id + ray_offset];
				char rdata = rays.rays[ray_id].data.d[threadIdx.x % 32];			
				int id_within_warp = threadIdx.x % 32;
				shared_ray_data[((element_id % 256) * 32) + id_within_warp] = rdata;
			}
		}

		// Loading of all rays is done.
		// now we do a batch loading of triangles
		// in read_cycles of 256 triangles

}

__global__ void bruteForceAos(TriangleArrayAos triangles, int num_triangles, RayArrayAos rays, int num_rays, 
	int* tri_idx_array, int tpivot, int* ray_idx_array, int rpivot,
	float* maxts, int* hitids) {

		__shared__ char  shared_ray_data[32 * MAX_BLOCK_THREADS];
		__shared__ short shared_tri_data[18 * MAX_BLOCK_THREADS];
		__shared__ int   tri_batches;
		float fmaxts = FLT_MAX;
		int   hitid = -1;
			
		int tidx = threadIdx.x + blockIdx.x * blockDim.x;
		if(threadIdx.x == 0) {
			tri_batches = (tpivot / 256) + (tpivot % 256 != 0);
		}
	
		//#1. Load all 256 or less rays.
		for(int read_cycle = 0; read_cycle < 32; read_cycle++) {
			int element_id = threadIdx.x / 32 + read_cycle * 8 + blockIdx.x * 256;
			if(element_id < rpivot) {
				// calculate offsetted ray_id
				int ray_id = ray_idx_array[element_id];
				char rdata = rays.rays[ray_id].data.d[threadIdx.x % 32];			
				int id_within_warp = threadIdx.x % 32;
				shared_ray_data[((element_id % 256) * 32) + id_within_warp] = rdata;
			}
		}
		__syncthreads();
		// now do a batch load of triangles
		int idx = 0;
		int this_time_triangles = 0;
		for(int tri_load_cycle = 0; tri_load_cycle < tri_batches; tri_load_cycle++) {
			for(int read_cycle = 0; read_cycle < 32; read_cycle++) {
				int element_id = threadIdx.x / 32 + read_cycle * 8 + tri_load_cycle * 256;
				if(element_id < tpivot) {
					int tri_id = tri_idx_array[element_id];
					short tdata = triangles.triangles[tri_id].data.d[threadIdx.x % 32];			
					int id_within_warp = threadIdx.x % 32;
					if(id_within_warp < 18) {
						shared_tri_data[((element_id % 256) * 18) + id_within_warp] = tdata;
					}
				}
			}
			__syncthreads();

			// now do brute force tests
			if((tpivot - idx) >= 256) { this_time_triangles = 256; idx += 256; }
			else this_time_triangles = tpivot - idx;
			if(tidx < rpivot) {
				double u, v, xt;
				float4 origin = make_float4(*((float*)&shared_ray_data[threadIdx.x * 32]), *((float*)&shared_ray_data[threadIdx.x * 32 + 4]), 
								    *((float*)&shared_ray_data[threadIdx.x * 32 + 8]), *((float*)&shared_ray_data[threadIdx.x * 32 + 12]));
				float4 direction = make_float4(*((float*)&shared_ray_data[threadIdx.x * 32 + 16]), *((float*)&shared_ray_data[threadIdx.x * 32 + 20]), 
								    *((float*)&shared_ray_data[threadIdx.x * 32 + 24]), *((float*)&shared_ray_data[threadIdx.x * 32 + 28]));
				Ray4 ir(origin, direction);
				for(int t = 0; t < this_time_triangles; t++) {
					// make triangle data
					Triangle3 it(make_float3(*((float*)&shared_tri_data[t * 18]), *((float*)&shared_tri_data[t * 18 + 2]), *((float*)&shared_tri_data[t * 18 + 4])),
						make_float3(*((float*)&shared_tri_data[t * 18 + 6]), *((float*)&shared_tri_data[t * 18 + 8]), *((float*)&shared_tri_data[t * 18 + 10])),
						make_float3(*((float*)&shared_tri_data[t * 18 + 12]), *((float*)&shared_tri_data[t * 18 + 14]), *((float*)&shared_tri_data[t * 18 + 16])));;
					double u, v, xt;
					if(rayIntersectAos<double>(it, ir, u, v, xt)) {
						if(xt > 0 && static_cast<float>(xt) < fmaxts) {
							fmaxts = static_cast<float>(xt);
							hitid = tri_idx_array[t + tri_load_cycle * 256];			
						}
					}
				}
			}
			__syncthreads();
		}
		
		// update the global list
		if(tidx < rpivot) {
			if(maxts[ray_idx_array[tidx]] > fmaxts && hitid != -1) {
				maxts[ray_idx_array[tidx]] = fmaxts;
				hitids[ray_idx_array[tidx]] = hitid;
			}
		}

}


void gpuDacrtSpatialAosMethod(const AABB4& space, TriangleArrayAos& d_triangles, int* dtri_idx_array, int num_triangles, int tpivot, 
	RayArrayAos& d_rays, int* dray_idx_array, int num_rays, int rpivot, float* d_maxts, int* d_hitids, 
	DacrtRunTimeParameters& rtparams, Counters& ctr, Logger& logger) {

		if(tpivot <rtparams.PARALLEL_TRI_THRESHOLD || rpivot < rtparams.PARALLEL_RAY_THRESHOLD) {
			
			Timer bruteforcetimer("bruteforcetimer");
			int NUM_THREADS_PER_BLOCK = rtparams.NUM_RAYS_PER_BLOCK;
			int needed_threads = max(tpivot, rpivot);
			int NUM_BLOCKS = (needed_threads / NUM_THREADS_PER_BLOCK) + (needed_threads % NUM_THREADS_PER_BLOCK != 0);
			bruteforcetimer.start();
			bruteForceAos<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(d_triangles, num_triangles, d_rays, num_rays, dtri_idx_array, tpivot, dray_idx_array, rpivot, d_maxts, d_hitids);
			bruteforcetimer.stop();
			ctr.brute_force_time += bruteforcetimer.get_ms();


		} else {

			{
				//  left
				AABB4 left = space;
				Timer trifiltertimer("trifiltertimer"), rayfiltertimer("rayfiltertimer");
				Timer trisortbykeytimer("trisortbykeytimer"), trireductiontimer("trireductiontimer");
				Timer raysortbykeytimer("raysortbykeytimer"), rayreductiontimer("rayreductiontimer");
				
				int newtpivot, newrpivot;
				float4 extent = space.data.bmax - space.data.bmin;
				if(extent.x > extent.y && extent.x > extent.z) {
					left.data.bmax.x = (space.data.bmax.x + space.data.bmin.x) * 0.5f;
				} else if(extent.y > extent.x && extent.y > extent.z) {
					left.data.bmax.y = (space.data.bmax.y + space.data.bmin.y) * 0.5f;
				} else {
					left.data.bmax.z = (space.data.bmax.z + space.data.bmin.z) * 0.5f;
				}

				int NUM_THREADS_PER_BLOCK = rtparams.NUM_RAYS_PER_BLOCK;
				int NUM_BLOCKS			  = (tpivot / NUM_THREADS_PER_BLOCK) + (tpivot % NUM_THREADS_PER_BLOCK != 0);
				int* trioccupy;
				checkCuda(cudaMalloc((void**)&trioccupy, sizeof(int) * tpivot));
				trifiltertimer.start();			
				triFilteringAos<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(d_triangles.triangles, dtri_idx_array, tpivot, make_float3(left.data.bmin.x, left.data.bmin.y, left.data.bmin.z), make_float3(left.data.bmax.x, left.data.bmax.y, left.data.bmax.z), trioccupy);
				trifiltertimer.stop();
				ctr.tri_filter_time += trifiltertimer.get_ms();

				trisortbykeytimer.start();
				thrust::sort_by_key(thrust::device_ptr<int>(trioccupy), thrust::device_ptr<int>(trioccupy) + tpivot, thrust::device_ptr<int>(dtri_idx_array), thrust::greater<int>());
				trisortbykeytimer.stop();
				ctr.trisortbykey_time += trisortbykeytimer.get_ms();

				trireductiontimer.start();
				newtpivot = thrust::reduce(thrust::device_ptr<int>(trioccupy), thrust::device_ptr<int>(trioccupy) + tpivot);
				trireductiontimer.stop();
				ctr.trireduction_time += trireductiontimer.get_ms();
				
				NUM_BLOCKS = (rpivot / NUM_THREADS_PER_BLOCK) + (rpivot % NUM_THREADS_PER_BLOCK != 0);
				int* roccupy;
				checkCuda(cudaMalloc((void**)&roccupy, sizeof(int) * rpivot));
				
				rayfiltertimer.start();
				rayFilteringAos<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(d_rays.rays, dray_idx_array, rpivot, left.data.bmin, left.data.bmax, roccupy);
				rayfiltertimer.stop();
				ctr.ray_filter_time += rayfiltertimer.get_ms();

				raysortbykeytimer.start();
				thrust::sort_by_key(thrust::device_ptr<int>(roccupy), thrust::device_ptr<int>(roccupy) + rpivot, thrust::device_ptr<int>(dray_idx_array), thrust::greater<int>());
				raysortbykeytimer.stop();
				ctr.raysortbykey_time += raysortbykeytimer.get_ms();
				
				rayreductiontimer.start();
				newrpivot = thrust::reduce(thrust::device_ptr<int>(roccupy) , thrust::device_ptr<int>(roccupy) + rpivot);
				rayreductiontimer.stop();
				ctr.rayreduction_time += rayreductiontimer.get_ms();
				
				//printf("NewTpivot : %d\tNew Rpiovt : %d\n", newtpivot, newrpivot);
				
				cudaFree(trioccupy);
				cudaFree(roccupy);

				gpuDacrtSpatialAosMethod(left, d_triangles, dtri_idx_array, num_triangles, newtpivot, d_rays, dray_idx_array, num_rays, newrpivot, d_maxts, d_hitids, rtparams, ctr, logger);

			}
			{
				// right
				AABB4 right = space;

				int newtpivot, newrpivot;
				Timer trifiltertimer("trifiltertimer"), rayfiltertimer("rayfiltertimer");
				Timer trisortbykeytimer("trisortbykeytimer"), trireductiontimer("trireductiontimer");
				Timer raysortbykeytimer("raysortbykeytimer"), rayreductiontimer("rayreductiontimer");
				
				float4 extent = space.data.bmax - space.data.bmin;
				if(extent.x > extent.y && extent.x > extent.z) {
					right.data.bmin.x = (space.data.bmax.x + space.data.bmin.x) * 0.5f;
				} else if(extent.y > extent.x && extent.y > extent.z) {
					right.data.bmin.y = (space.data.bmax.y + space.data.bmin.y) * 0.5f;
				} else {
					right.data.bmin.z = (space.data.bmax.z + space.data.bmin.z) * 0.5f;
				}

				int NUM_THREADS_PER_BLOCK = rtparams.NUM_RAYS_PER_BLOCK;
				int NUM_BLOCKS = (tpivot / NUM_THREADS_PER_BLOCK) + (tpivot % NUM_THREADS_PER_BLOCK != 0);
				int* trioccupy, *roccupy;
				checkCuda(cudaMalloc((void**)&trioccupy, sizeof(int) * tpivot));
				
				trifiltertimer.start();	
				triFilteringAos<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(d_triangles.triangles, dtri_idx_array, tpivot, make_float3(right.data.bmin.x, right.data.bmin.y, right.data.bmin.z), make_float3(right.data.bmax.x, right.data.bmax.y, right.data.bmax.z), trioccupy);
				trifiltertimer.stop();
				ctr.tri_filter_time += trifiltertimer.get_ms();

				trisortbykeytimer.start();
				thrust::sort_by_key(thrust::device_ptr<int>(trioccupy), thrust::device_ptr<int>(trioccupy) + tpivot, thrust::device_ptr<int>(dtri_idx_array), thrust::greater<int>());
				trisortbykeytimer.stop();
				ctr.trisortbykey_time += trisortbykeytimer.get_ms();

				trireductiontimer.start();
				newtpivot = thrust::reduce(thrust::device_ptr<int>(trioccupy), thrust::device_ptr<int>(trioccupy) + tpivot);
				trireductiontimer.stop();
				ctr.trireduction_time += trireductiontimer.get_ms();

				NUM_BLOCKS = (rpivot / NUM_THREADS_PER_BLOCK) + (rpivot % NUM_THREADS_PER_BLOCK != 0);
				checkCuda(cudaMalloc((void**)&roccupy, sizeof(int) * rpivot));

				rayfiltertimer.start();
				rayFilteringAos<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(d_rays.rays, dray_idx_array, rpivot, right.data.bmin, right.data.bmax, roccupy);
				rayfiltertimer.stop();
				ctr.ray_filter_time += rayfiltertimer.get_ms();

				raysortbykeytimer.start();
				thrust::sort_by_key(thrust::device_ptr<int>(roccupy), thrust::device_ptr<int>(roccupy) + rpivot, thrust::device_ptr<int>(dray_idx_array), thrust::greater<int>());
				raysortbykeytimer.stop();
				ctr.raysortbykey_time += raysortbykeytimer.get_ms();
				
				rayreductiontimer.start();
				newrpivot = thrust::reduce(thrust::device_ptr<int>(roccupy), thrust::device_ptr<int>(roccupy) + rpivot);
				rayreductiontimer.stop();
				ctr.rayreduction_time += rayreductiontimer.get_ms();
				
				//printf("NewTpivot : %d\tNew Rpiovt : %d\n", newtpivot, newrpivot);

				cudaFree(trioccupy);
				cudaFree(roccupy);
				gpuDacrtSpatialAosMethod(right, d_triangles, dtri_idx_array, num_triangles, newtpivot, d_rays, dray_idx_array, num_rays, newrpivot, d_maxts, d_hitids, rtparams, ctr, logger);
			}
		}
}

