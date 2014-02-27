#include <dacrt/dacrt.h>
#include <util/cutimer.h>

#define ENABLE_LOGGING

// Both kernels are defined in "bruteforcekernel.cu"
extern "C"
__global__ void segmentedBruteForce(RayArray rays, TriangleArray triangles, int* buffered_ray_ids, int ray_buffer_occupied, int* buffered_tri_ids, int tri_buffer_occupied,
	int* ray_segment_sizes, int* tri_segment_sizes, int* ray_segment_start, int* tri_segment_start, int num_segments, float* maxts,	int* hitids,
	int num_threads_launched, int num_blocks_launched);

extern "C" __global__ void updateMinKernel(int* ray_id, float* min_hits, int* minhit_ids, float* global_min, int* global_hits, int num_rays);

// complete the segmented operation
extern "C" void dacrtCompleteRender(ParallelPack& pack, TriangleArray& dev_triangles, RayArray& dev_rays, DacrtRunTimeParameters& rtparams, Counters& ctr);

// one global host only data
thrust::host_vector<int> hbuffered_ray_idx_array;
thrust::host_vector<int> hbuffered_tri_idx_array;
thrust::host_vector<int> htri_segment_sizes;
thrust::host_vector<int> hray_segment_sizes;
thrust::host_vector<int> hsegment_ids;

void cpuGpuDacrtSpatialFunction(const AABB& space, 
	AabbArray& tri_aabbs,																							
	TriangleArray& triangles, TriangleArray& dev_triangles, int num_triangles, int* tri_idx_array, int tpivot,		
	RayArray& rays, RayArray& dev_rays, int num_rays, int* ray_idx_array, int rpivot,								
	ParallelPack& pack,
	float* maxts, int* hitids,																						
	DacrtRunTimeParameters& rtparams,
	Counters& ctr,
	Logger& logger
	) {
		
		if(tpivot != 0 && rpivot != 0)
		if(tpivot < rtparams.PARALLEL_TRI_THRESHOLD || rpivot < rtparams.PARALLEL_RAY_THRESHOLD) {
#ifdef ENABLE_LOGGING
			logger.write(tpivot, rpivot);
#endif
			// we create a work queue and copy stuff?
			if((pack.ray_buffer_occupied + rpivot) < rtparams.BUFFER_SIZE && (pack.tri_buffer_occupied + tpivot) < rtparams.BUFFER_SIZE && pack.num_segments < rtparams.MAX_SEGMENTS) {
				ctr.raytri += tpivot * rpivot;				// brute force count
				Timer mem_cpy_timer("Memcpy Timer");
				mem_cpy_timer.start();
			
				// host copies
				thrust::copy(ray_idx_array, ray_idx_array + rpivot, hbuffered_ray_idx_array.begin() + pack.ray_buffer_occupied);
				thrust::copy(tri_idx_array, tri_idx_array + tpivot, hbuffered_tri_idx_array.begin() + pack.tri_buffer_occupied);
				htri_segment_sizes[pack.num_segments] = tpivot;
				hray_segment_sizes[pack.num_segments] = rpivot;
				hsegment_ids[pack.num_segments] = pack.num_segments;

				pack.num_segments++;		// increment the count
				pack.ray_buffer_occupied += rpivot;
				pack.tri_buffer_occupied += tpivot;
				
				mem_cpy_timer.stop();
				ctr.mem_cpy_time += mem_cpy_timer.get_ms();
				//return;
			} else {

				// copy data from host to device in one shot
				Timer mem_cpy_timer1("Memcpy Timer");
				mem_cpy_timer1.start();
				thrust::copy(hbuffered_ray_idx_array.begin(), hbuffered_ray_idx_array.begin() + pack.ray_buffer_occupied, pack.buffered_ray_idx.begin());
				thrust::copy(hbuffered_tri_idx_array.begin(), hbuffered_tri_idx_array.begin() + pack.tri_buffer_occupied, pack.buffered_tri_idx.begin());
				thrust::copy(htri_segment_sizes.begin(), htri_segment_sizes.begin() + pack.num_segments, pack.tri_segment_sizes.begin());
				thrust::copy(hray_segment_sizes.begin(), hray_segment_sizes.begin() + pack.num_segments, pack.ray_segment_sizes.begin());
				thrust::copy(hsegment_ids.begin(), hsegment_ids.begin() + pack.num_segments, pack.segment_ids.begin());
				mem_cpy_timer1.stop();
				ctr.mem_cpy_time += mem_cpy_timer1.get_ms();

				// clear data from host
				hbuffered_ray_idx_array.clear(); hbuffered_tri_idx_array.clear(); htri_segment_sizes.clear(); hray_segment_sizes.clear(); 
				hsegment_ids.clear(); 
								
				// brute force parallel kernel
				thrust::device_vector<int> ray_segment_start(pack.num_segments);
				thrust::device_vector<int> tri_segment_start(pack.num_segments);
				thrust::exclusive_scan(pack.tri_segment_sizes.begin(), pack.tri_segment_sizes.begin() + pack.num_segments, tri_segment_start.begin());
				thrust::exclusive_scan(pack.ray_segment_sizes.begin(), pack.ray_segment_sizes.begin() + pack.num_segments, ray_segment_start.begin());
				
				// LOGIC1: We will launch one block for all segments. [very bad though]..
				int num_blocks = pack.num_segments;
				int num_threads_per_block = rtparams.NUM_RAYS_PER_BLOCK;
				//int num_threads_per_block = 320;
				
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

				// now insert them into the buffer 
				// NOTE: this condition would have occured if we didnt have free space in the beginning itself. Now that we have cleared up space
				//       we can insert the original stuff we were supposed to insert.
				ctr.raytri += rpivot * tpivot;
				Timer mem_cpy_timer("Mem cpy Timer");
				mem_cpy_timer.start();
				
				thrust::copy(ray_idx_array, ray_idx_array + rpivot, hbuffered_ray_idx_array.begin() + pack.ray_buffer_occupied);
				thrust::copy(tri_idx_array, tri_idx_array + tpivot, hbuffered_tri_idx_array.begin() + pack.tri_buffer_occupied);
				htri_segment_sizes[pack.num_segments] = tpivot;
				hray_segment_sizes[pack.num_segments] = rpivot;
				hsegment_ids[pack.num_segments] = pack.num_segments;
				
				mem_cpy_timer.stop();
				ctr.mem_cpy_time += mem_cpy_timer.get_ms();

				pack.num_segments++;
				pack.ray_buffer_occupied += rpivot;
				pack.tri_buffer_occupied += tpivot;
			}
			
		} else {
			
			float3 extents = space.bmax - space.bmin;
			{
			
				// left child
				int newtpivot, splitaxis;
				AABB left = space;
				double start = omp_get_wtime();
				if(extents.x > extents.y && extents.x > extents.z) {
					left.bmax.x = (space.bmax.x + space.bmin.x) * 0.5f;
					splitaxis = 0;
					newtpivot = filterTrianglesSpatial(triangles, num_triangles, tri_idx_array, left, tpivot, splitaxis, ctr.tribox);
				} else if(extents.y > extents.x && extents.y > extents.z) {
					left.bmax.y = (space.bmax.y + space.bmin.y) * 0.5f;
					splitaxis = 1;
					newtpivot = filterTrianglesSpatial(triangles, num_triangles, tri_idx_array, left, tpivot, splitaxis, ctr.tribox);
				} else {
					left.bmax.z = (space.bmax.z + space.bmin.z) * 0.5f;
					splitaxis = 2;
					newtpivot = filterTrianglesSpatial(triangles, num_triangles, tri_idx_array, left, tpivot, splitaxis, ctr.tribox);
				}
				
				double end = omp_get_wtime();
				ctr.tri_filter_time += static_cast<float>(end-start);

				start = omp_get_wtime();
				int newrpivot = filterRaysSpatial(rays, num_rays, ray_idx_array, left, rpivot, ctr.raybox);		// no change here..!! same code for ray filtering.
				end = omp_get_wtime();
				ctr.ray_filter_time += static_cast<float>(end-start);

				cpuGpuDacrtSpatialFunction(left, tri_aabbs, triangles, dev_triangles, num_triangles, tri_idx_array, newtpivot, rays, dev_rays, num_rays, ray_idx_array,
					newrpivot, pack, maxts, hitids, rtparams, ctr, logger);
				
			}
			{
				double start = omp_get_wtime();
				AABB right = space;
				int newtpivot, splitaxis;
				if(extents.x > extents.y && extents.x > extents.z) {
					right.bmin.x = (space.bmax.x + space.bmin.x) * 0.5f;
					splitaxis = 0;
					newtpivot = filterTrianglesSpatial(triangles, num_triangles, tri_idx_array, right, tpivot, splitaxis, ctr.tribox);
				} else if(extents.y > extents.x && extents.y > extents.z) {
					right.bmin.y = (space.bmax.y + space.bmin.y) * 0.5f;
					splitaxis = 1;
					newtpivot = filterTrianglesSpatial(triangles, num_triangles, tri_idx_array, right, tpivot, splitaxis, ctr.tribox);
				} else {
					right.bmin.z = (space.bmax.z + space.bmin.z) * 0.5f;
					splitaxis = 2;
					newtpivot = filterTrianglesSpatial(triangles, num_triangles, tri_idx_array, right, tpivot, splitaxis, ctr.tribox);
				}

				double end = omp_get_wtime();
				ctr.tri_filter_time += static_cast<float>(end-start);

				start = omp_get_wtime();
				int newrpivot = filterRaysSpatial(rays, num_rays, ray_idx_array, right, rpivot, ctr.raybox);
				end = omp_get_wtime();
				ctr.ray_filter_time += static_cast<float>(end-start);

				cpuGpuDacrtSpatialFunction(right, tri_aabbs, triangles, dev_triangles, num_triangles, tri_idx_array, newtpivot, rays, dev_rays, num_rays, ray_idx_array, newrpivot,
					pack, maxts, hitids, rtparams, ctr, logger);
			}
		}
		
}


void cpuGpuDacrtSpatialPartitioning(const AABB& sceneBox, 
	TriangleArray& triangles, AabbArray& tri_aabbs, RayArray& rays,
	TriangleArray& dev_triangles, RayArray& dev_rays,
	int num_triangles, int* tri_idx_array, int tpivot, 
	int num_rays, int* ray_idx_array, int rpivot,															
	float* maxts, int* hitids,
	DacrtRunTimeParameters& rtparams,
	Counters& ctr,
	Logger& logger
	) {

		// resize the host data
		hbuffered_ray_idx_array.resize(rtparams.BUFFER_SIZE);
		hbuffered_tri_idx_array.resize(rtparams.BUFFER_SIZE);
		htri_segment_sizes.resize(rtparams.MAX_SEGMENTS);
		hray_segment_sizes.resize(rtparams.MAX_SEGMENTS);
		hsegment_ids.resize(rtparams.MAX_SEGMENTS);
				
		thrust::device_vector<int>		buffered_ray_idx(rtparams.BUFFER_SIZE);
		thrust::device_vector<int>		buffered_tri_idx(rtparams.BUFFER_SIZE);
		thrust::device_vector<int>		segment_ids(rtparams.MAX_SEGMENTS);
		thrust::device_vector<int>		ray_segment_sizes(rtparams.MAX_SEGMENTS);
		thrust::device_vector<int>		tri_segment_sizes(rtparams.MAX_SEGMENTS);
		
		thrust::device_vector<float>	buffered_ray_maxts(rtparams.BUFFER_SIZE, FLT_MAX);
		thrust::device_vector<int>		buffered_ray_hitids(rtparams.BUFFER_SIZE, -1);
		thrust::device_vector<float>	dev_ray_maxts(num_rays, FLT_MAX);
		thrust::device_vector<int>		dev_hitids(num_rays, -1);


		int ray_buffer_occupied = 0;
		int tri_buffer_occupied = 0;
		int num_segments = 0;

		ParallelPack pack(buffered_ray_idx, buffered_tri_idx, segment_ids, ray_segment_sizes, tri_segment_sizes, buffered_ray_maxts, buffered_ray_hitids,
			dev_ray_maxts, dev_hitids, ray_buffer_occupied, tri_buffer_occupied, num_segments);

		printf("Starting dacrt in parallel\n");

		cpuGpuDacrtSpatialFunction(sceneBox, tri_aabbs, triangles, dev_triangles, num_triangles, tri_idx_array, tpivot, rays, dev_rays, num_rays, ray_idx_array, rpivot,
			pack, maxts, hitids, rtparams, ctr, logger);

		// complete the incomplete segments
		// now we might have come out with a half full buffer. We just have to complete that also.
		// copy to pack elements
		Timer mem_cpy_timer("Memcpy Timer");
		mem_cpy_timer.start();
		thrust::copy(hbuffered_ray_idx_array.begin(), hbuffered_ray_idx_array.begin() + pack.ray_buffer_occupied, pack.buffered_ray_idx.begin());
		thrust::copy(hbuffered_tri_idx_array.begin(), hbuffered_tri_idx_array.begin() + pack.tri_buffer_occupied, pack.buffered_tri_idx.begin());
		thrust::copy(htri_segment_sizes.begin(), htri_segment_sizes.begin() + pack.num_segments, pack.tri_segment_sizes.begin());
		thrust::copy(hray_segment_sizes.begin(), hray_segment_sizes.begin() + pack.num_segments, pack.ray_segment_sizes.begin());
		thrust::copy(hsegment_ids.begin(), hsegment_ids.begin() + pack.num_segments, pack.segment_ids.begin());
		
		// copy the two more stuff
		
		mem_cpy_timer.stop();
		ctr.mem_cpy_time += mem_cpy_timer.get_ms();
		
		// clear memory
		hbuffered_ray_idx_array.clear(); hbuffered_tri_idx_array.clear(); htri_segment_sizes.clear(); hray_segment_sizes.clear(); hsegment_ids.clear();

		if(pack.num_segments > 0) 
			dacrtCompleteRender(pack, dev_triangles, dev_rays, rtparams, ctr);
		
		// now pack.dev_ray_maxts will have all the required values
		// and the pack.dev_ray_hitids will have hitids.
		// copy them into host buffer for rendering
		thrust::copy(dev_ray_maxts.begin(), dev_ray_maxts.end(), maxts);
		thrust::copy(dev_hitids.begin(), dev_hitids.end(), hitids);
}
