#include <dacrt/dacrt.h>
#include <util/cutimer.h>

/**
The gpu dacrt algorithm with the brute force performed in a segmented manner
*/

//extern defintions

extern "C" __global__ void trianglePartitionKernel(float3* v0, float3* v1, float3* v2, int* tri_ids, int num_tris, float3 bmin, float3 bmax, int* occupy);
extern "C" __global__ void rayPartitionKernel(float3* o, float3* dir, int* ray_ids, int num_rays, float3 bmin, float3 bmax, int* occupy);
extern "C" void dacrtCompleteRender(ParallelPack& pack, TriangleArray& dev_triangles, RayArray& dev_rays, DacrtRunTimeParameters& rtparams, Counters& ctr);
extern "C" __global__ void updateMinKernel(int* ray_id, float* min_hits, int* minhit_ids, float* global_min, int* global_hits, int num_rays);
extern "C" __global__ void segmentedBruteForce(RayArray rays, TriangleArray triangles, int* buffered_ray_ids, int ray_buffer_occupied, int* buffered_tri_ids, int tri_buffer_occupied,
	int* ray_segment_sizes, int* tri_segment_sizes, int* ray_segment_start, int* tri_segment_start, int num_segments, float* maxts,	int* hitids,
	int num_threads_launched, int num_blocks_launched);

extern unsigned int total_device_memory;

void gpuDacrtSpatialSegmentedFunction(const AABB& space, TriangleArray& d_triangles, int* dtri_idx_array, int num_triangles, int tpivot, 
	RayArray& d_rays, int* dray_idx_array, int num_rays, int rpivot, /*float* d_maxts, int* d_hitids -- These two values are passed in the parallel pack*/ 
	ParallelPack& pack,
	DacrtRunTimeParameters& rtparams, 
	Counters& ctr,
	Logger& logger) {
		if(tpivot < rtparams.PARALLEL_TRI_THRESHOLD || rpivot < rtparams.PARALLEL_RAY_THRESHOLD) {
			// check if we are within segmented range
			if(tpivot != 0 && rpivot != 0)
			if((pack.ray_buffer_occupied + rpivot) < rtparams.BUFFER_SIZE && (pack.tri_buffer_occupied + tpivot) < rtparams.BUFFER_SIZE && pack.num_segments < rtparams.MAX_SEGMENTS) {
				ctr.raytri += tpivot * rpivot;				// brute force count
				Timer mem_cpy_timer("Memcpy Timer");
				mem_cpy_timer.start();
				thrust::copy(thrust::device_ptr<int>(dray_idx_array), thrust::device_ptr<int>(dray_idx_array) + rpivot, pack.buffered_ray_idx.begin() + pack.ray_buffer_occupied);
				thrust::copy(thrust::device_ptr<int>(dtri_idx_array), thrust::device_ptr<int>(dtri_idx_array) + tpivot, pack.buffered_tri_idx.begin() + pack.tri_buffer_occupied);
				pack.tri_segment_sizes[pack.num_segments] = tpivot;
				pack.ray_segment_sizes[pack.num_segments] = rpivot;
				pack.segment_ids[pack.num_segments] = pack.num_segments;
				pack.num_segments++;		// increment the count
				pack.ray_buffer_occupied += rpivot;
				pack.tri_buffer_occupied += tpivot;
				mem_cpy_timer.stop();
				ctr.mem_cpy_time += mem_cpy_timer.get_ms();
			} else {
				thrust::device_vector<int> ray_segment_start(pack.num_segments);
				thrust::device_vector<int> tri_segment_start(pack.num_segments);
				thrust::exclusive_scan(pack.tri_segment_sizes.begin(), pack.tri_segment_sizes.begin() + pack.num_segments, tri_segment_start.begin());
				thrust::exclusive_scan(pack.ray_segment_sizes.begin(), pack.ray_segment_sizes.begin() + pack.num_segments, ray_segment_start.begin());
				
				int num_blocks = pack.num_segments;
				int num_threads_per_block = rtparams.NUM_RAYS_PER_BLOCK;
				
				Timer seg_brute_timer("SegmentedBruteForce Timer");
				
				seg_brute_timer.start();
				segmentedBruteForce<<<num_blocks, num_threads_per_block>>>(d_rays, d_triangles, thrust::raw_pointer_cast(&pack.buffered_ray_idx[0]), 
					pack.ray_buffer_occupied, thrust::raw_pointer_cast(&pack.buffered_tri_idx[0]), pack.tri_buffer_occupied, 
					thrust::raw_pointer_cast(&pack.ray_segment_sizes[0]), thrust::raw_pointer_cast(&pack.tri_segment_sizes[0]), 
					thrust::raw_pointer_cast(&ray_segment_start[0]), thrust::raw_pointer_cast(&tri_segment_start[0]), 
					pack.num_segments, thrust::raw_pointer_cast(&pack.buffered_ray_maxts[0]),
					thrust::raw_pointer_cast(&pack.buffered_ray_hitids[0]), num_threads_per_block * num_blocks, num_blocks);
				
				seg_brute_timer.stop();
				ctr.brute_force_time += seg_brute_timer.get_ms();

				/// FREE MEMORY DECLARED ABOVE
				//ray_segment_start.clear(); ray_segment_start.shrink_to_fit();
				//tri_segment_start.clear(); tri_segment_start.shrink_to_fit();

				// call kernel
				// now we can do a sort operation on the hitids, and maxts based on ray ids
				Timer seg_sort_timer("Seg Sorted Timer");
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
				//ray_idx.clear(); ray_idx.shrink_to_fit();
				//ray_maxts.clear(); ray_maxts.shrink_to_fit();
				//ray_hitids.clear(); ray_hitids.shrink_to_fit();
				pack.buffered_ray_idx.clear();
				pack.buffered_tri_idx.clear();
				pack.tri_segment_sizes.clear();
				pack.ray_segment_sizes.clear();
				pack.segment_ids.clear();
							
				pack.ray_buffer_occupied = 0;
				pack.tri_buffer_occupied = 0;
				pack.num_segments = 0;

				// now insert them into the buffer 
				// NOTE: this condition would have occured if we didnt have free space in the beginning itself. Now that we have cleared up space
				//       we can insert the original stuff we were supposed to insert.
				ctr.raytri += rpivot * tpivot;
				Timer mem_cpy_timer("Mem cpy Timer");
				mem_cpy_timer.start();
				thrust::copy(thrust::device_ptr<int>(dray_idx_array), thrust::device_ptr<int>(dray_idx_array) + rpivot, pack.buffered_ray_idx.begin() + pack.ray_buffer_occupied);
				thrust::copy(thrust::device_ptr<int>(dtri_idx_array), thrust::device_ptr<int>(dtri_idx_array) + tpivot, pack.buffered_tri_idx.begin() + pack.tri_buffer_occupied);
				pack.tri_segment_sizes[pack.num_segments] = tpivot;
				pack.ray_segment_sizes[pack.num_segments] = rpivot;
				pack.segment_ids[pack.num_segments]		= pack.num_segments;
				pack.num_segments++;
				pack.ray_buffer_occupied += rpivot;
				pack.tri_buffer_occupied += tpivot;
				mem_cpy_timer.stop();
				ctr.mem_cpy_time += mem_cpy_timer.get_ms();
				//return;
			}
		} else {
			// do the triangle filtering and ray filtering for both left and right nodes
			// full gpu solution
			{
				
				AABB left = space;
				Timer trifiltertimer("trifiltertimer"), rayfiltertimer("rayfiltertimer");
				Timer trisortbykeytimer("trisortbykeytimer"), trireductiontimer("trireductiontimer");
				Timer raysortbykeytimer("raysortbykeytimer"), rayreductiontimer("rayreductiontimer");
				
				int newtpivot, newrpivot;
				float3 extent = space.bmax - space.bmin;
				if(extent.x > extent.y && extent.x > extent.z) {
					left.bmax.x = (space.bmax.x + space.bmin.x) * 0.5f;
				} else if(extent.y > extent.x && extent.y > extent.z) {
					left.bmax.y = (space.bmax.y + space.bmin.y) * 0.5f;
				} else {
					left.bmax.z = (space.bmax.z + space.bmin.z) * 0.5f;
				}
								
				// Step 1. triangle filtering
				// Calculate effective number of ray blocks to begin
				// As always we define the max threads in this case to be 512
				int NUM_THREADS_PER_BLOCK = rtparams.NUM_RAYS_PER_BLOCK;
				int NUM_BLOCKS			  = (tpivot / NUM_THREADS_PER_BLOCK) + (tpivot % NUM_THREADS_PER_BLOCK != 0);
				int* trioccupy;
				checkCuda(cudaMalloc((void**)&trioccupy, sizeof(int) * tpivot));
				trifiltertimer.start();			
				trianglePartitionKernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(d_triangles.v0, d_triangles.v1, d_triangles.v2, dtri_idx_array, tpivot, left.bmin, left.bmax, trioccupy);
				trifiltertimer.stop();
				ctr.tri_filter_time += trifiltertimer.get_ms();
				
				
				// now we have to a scatter operation of the original id vectors based on this new scheme
				/// NOTE: algorithmically we are not concerned with the order of indices in the original device vector
				///       So we can use a sort by key operation to reorder the vertices
				///		  Its guaranteeed that sort action on keys only works on size of key which in this case is just tpivot!
				trisortbykeytimer.start();
				ctr.trifilter_sort_cnt += 1;
				thrust::sort_by_key(thrust::device_ptr<int>(trioccupy), thrust::device_ptr<int>(trioccupy) + tpivot, thrust::device_ptr<int>(dtri_idx_array), thrust::greater<int>());
				//thrust::inclusive_scan(thrust::device_ptr<int>(trioccupy), thrust::device_ptr<int>(trioccupy) + tpivot, thrust::device_ptr<int>(trioccupy));		// do inplace. Order is all screwed up. BEWARE.!!
				//checkCuda(cudaMemcpy((void*)&newtpivot, trioccupy + tpivot - 1, sizeof(int), cudaMemcpyDeviceToHost));
				trisortbykeytimer.stop();
				ctr.trisortbykey_time += trisortbykeytimer.get_ms();
				ctr.tri_sort_times.push_back(std::make_pair(tpivot, trisortbykeytimer.get_ms()));
				
				trireductiontimer.start();
				newtpivot = thrust::reduce(thrust::device_ptr<int>(trioccupy), thrust::device_ptr<int>(trioccupy) + tpivot);
				trireductiontimer.stop();
				ctr.trireduction_time += trireductiontimer.get_ms();
								
				NUM_BLOCKS = (rpivot / NUM_THREADS_PER_BLOCK) + (rpivot % NUM_THREADS_PER_BLOCK != 0);
				int* roccupy;
				checkCuda(cudaMalloc((void**)&roccupy, sizeof(int) * rpivot));
				
				rayfiltertimer.start();
				rayPartitionKernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(d_rays.o, d_rays.d, dray_idx_array, rpivot, left.bmin, left.bmax, roccupy);
				rayfiltertimer.stop();
				ctr.ray_filter_time += rayfiltertimer.get_ms();

				raysortbykeytimer.start();
				ctr.rayfilter_sort_cnt += 1;
				thrust::sort_by_key(thrust::device_ptr<int>(roccupy), thrust::device_ptr<int>(roccupy) + rpivot, thrust::device_ptr<int>(dray_idx_array), thrust::greater<int>());
				//thrust::inclusive_scan(thrust::device_ptr<int>(roccupy), thrust::device_ptr<int>(roccupy) + rpivot, thrust::device_ptr<int>(roccupy));
				//checkCuda(cudaMemcpy((void*)&newrpivot, roccupy + rpivot - 1, sizeof(int), cudaMemcpyDeviceToHost));
				raysortbykeytimer.stop();
				ctr.raysortbykey_time += raysortbykeytimer.get_ms();
				ctr.ray_sort_times.push_back(std::make_pair(rpivot, raysortbykeytimer.get_ms()));
				
				rayreductiontimer.start();
				newrpivot = thrust::reduce(thrust::device_ptr<int>(roccupy) , thrust::device_ptr<int>(roccupy) + rpivot);
				rayreductiontimer.stop();
				ctr.rayreduction_time += rayreductiontimer.get_ms();
				
				CUDA_SAFE_RELEASE(trioccupy);
				CUDA_SAFE_RELEASE(roccupy);

				gpuDacrtSpatialSegmentedFunction(left, d_triangles, dtri_idx_array, num_triangles, newtpivot, d_rays, dray_idx_array, num_rays, newrpivot, pack, rtparams, ctr, logger);
			}
			{
				// right space.
				AABB right = space;
				int newtpivot, newrpivot;
				Timer trifiltertimer("trifiltertimer"), rayfiltertimer("rayfiltertimer");
				Timer trisortbykeytimer("trisortbykeytimer"), trireductiontimer("trireductiontimer");
				Timer raysortbykeytimer("raysortbykeytimer"), rayreductiontimer("rayreductiontimer");
				
				float3 extent = space.bmax - space.bmin;
				if(extent.x > extent.y && extent.x > extent.z) {
					right.bmin.x = (space.bmax.x + space.bmin.x) * 0.5f;
				} else if(extent.y > extent.x && extent.y > extent.z) {
					right.bmin.y = (space.bmax.y + space.bmin.y) * 0.5f;
				} else {
					right.bmin.z = (space.bmax.z + space.bmin.z) * 0.5f;
				}

				int NUM_THREADS_PER_BLOCK = rtparams.NUM_RAYS_PER_BLOCK;
				int NUM_BLOCKS = (tpivot / NUM_THREADS_PER_BLOCK) + (tpivot % NUM_THREADS_PER_BLOCK != 0);
				int* trioccupy, *roccupy;
				checkCuda(cudaMalloc((void**)&trioccupy, sizeof(int) * tpivot));
				
				trifiltertimer.start();	
				trianglePartitionKernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(d_triangles.v0, d_triangles.v1, d_triangles.v2, dtri_idx_array, tpivot, right.bmin, right.bmax, trioccupy);
				trifiltertimer.stop();
				ctr.tri_filter_time += trifiltertimer.get_ms();

				trisortbykeytimer.start();
				ctr.trifilter_sort_cnt += 1;
				thrust::sort_by_key(thrust::device_ptr<int>(trioccupy), thrust::device_ptr<int>(trioccupy) + tpivot, thrust::device_ptr<int>(dtri_idx_array), thrust::greater<int>());
				//thrust::inclusive_scan(thrust::device_ptr<int>(trioccupy), thrust::device_ptr<int>(trioccupy) + tpivot, thrust::device_ptr<int>(trioccupy));
				//checkCuda(cudaMemcpy((void*)&newtpivot, trioccupy + tpivot - 1, sizeof(int), cudaMemcpyDeviceToHost));
				trisortbykeytimer.stop();
				ctr.trisortbykey_time += trisortbykeytimer.get_ms();
				ctr.tri_sort_times.push_back(std::make_pair(tpivot, trisortbykeytimer.get_ms()));

				trireductiontimer.start();
				newtpivot = thrust::reduce(thrust::device_ptr<int>(trioccupy), thrust::device_ptr<int>(trioccupy) + tpivot);
				trireductiontimer.stop();
				ctr.trireduction_time += trireductiontimer.get_ms();
				
				NUM_BLOCKS = (rpivot / NUM_THREADS_PER_BLOCK) + (rpivot % NUM_THREADS_PER_BLOCK != 0);
				checkCuda(cudaMalloc((void**)&roccupy, sizeof(int) * rpivot));

				rayfiltertimer.start();
				rayPartitionKernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(d_rays.o, d_rays.d, dray_idx_array, rpivot, right.bmin, right.bmax, roccupy);
				rayfiltertimer.stop();
				ctr.ray_filter_time += rayfiltertimer.get_ms();

				raysortbykeytimer.start();
				ctr.rayfilter_sort_cnt += 1;
				thrust::sort_by_key(thrust::device_ptr<int>(roccupy), thrust::device_ptr<int>(roccupy) + rpivot, thrust::device_ptr<int>(dray_idx_array), thrust::greater<int>());
				//thrust::inclusive_scan(thrust::device_ptr<int>(roccupy), thrust::device_ptr<int>(roccupy) + rpivot, thrust::device_ptr<int>(roccupy));
				//checkCuda(cudaMemcpy((void*)&newrpivot, roccupy + rpivot - 1, sizeof(int), cudaMemcpyDeviceToHost));
				raysortbykeytimer.stop();
				ctr.raysortbykey_time += raysortbykeytimer.get_ms();
				ctr.ray_sort_times.push_back(std::make_pair(rpivot, raysortbykeytimer.get_ms()));

				rayreductiontimer.start();
				newrpivot = thrust::reduce(thrust::device_ptr<int>(roccupy), thrust::device_ptr<int>(roccupy) + rpivot);
				rayreductiontimer.stop();
				ctr.rayreduction_time += rayreductiontimer.get_ms();
				
				cudaFree(trioccupy);
				cudaFree(roccupy);
				gpuDacrtSpatialSegmentedFunction(right, d_triangles, dtri_idx_array, num_triangles, newtpivot, d_rays, dray_idx_array, num_rays, newrpivot, pack, rtparams, ctr, logger);
			}
		}
}



void gpuDacrtSpatialSegmented(const AABB& space, 
	TriangleArray& d_triangles, int* dtri_idx_array, int num_triangles, int tpivot,
	RayArray& d_rays, int* dray_idx_array, int num_rays, int rpivot, 
	float* h_maxts, int* h_hitids, 
	DacrtRunTimeParameters& rtparams, 
	Counters& ctr,
	Logger& logger
	) {
	// setup the parallel pack structure and pass it to the function
		thrust::device_vector<int>		buffered_ray_idx(rtparams.BUFFER_SIZE);
		thrust::device_vector<int>		buffered_tri_idx(rtparams.BUFFER_SIZE);
		thrust::device_vector<int>		segment_ids(rtparams.MAX_SEGMENTS);
		thrust::device_vector<int>		ray_segment_sizes(rtparams.MAX_SEGMENTS);
		thrust::device_vector<int>		tri_segment_sizes(rtparams.MAX_SEGMENTS);
		thrust::device_vector<float>	buffered_ray_maxts(rtparams.BUFFER_SIZE, FLT_MAX);
		thrust::device_vector<int>		buffered_ray_hitids(rtparams.BUFFER_SIZE, -1);
		thrust::device_vector<float>    dev_ray_maxts(num_rays, FLT_MAX);
		thrust::device_vector<int>	    dev_hitids(num_rays, -1);

		int ray_buffer_occupied = 0;
		int tri_buffer_occupied = 0;
		int num_segments = 0;

		ParallelPack pack(buffered_ray_idx, buffered_tri_idx, segment_ids, ray_segment_sizes, tri_segment_sizes, buffered_ray_maxts, buffered_ray_hitids,
			dev_ray_maxts, dev_hitids, ray_buffer_occupied, tri_buffer_occupied, num_segments);

		printf("DACRT Full GPU implementation - Segmented Brute Force\n");
		gpuDacrtSpatialSegmentedFunction(space, d_triangles, dtri_idx_array, num_triangles, tpivot, d_rays, dray_idx_array, num_rays, rpivot, pack, rtparams, ctr, logger);

		if(pack.num_segments > 0) 
			dacrtCompleteRender(pack, d_triangles, d_rays, rtparams, ctr);
				
		thrust::copy(dev_ray_maxts.begin(), dev_ray_maxts.end(), h_maxts);
		thrust::copy(dev_hitids.begin(), dev_hitids.end(), h_hitids);
}

void gpuDacrtShadowSegmented(const AABB& space, TriangleArray& d_triangles, int* dtri_idx_array, int num_triangles, int tpivot,
	RayArray& d_shadowrays, int* d_shadowray_idx_array, int* d_primshadow_idx_array, int num_rays, int rpivot,
	bool* shadow, DacrtRunTimeParameters& rtparams, Counters& ctr) {
}


