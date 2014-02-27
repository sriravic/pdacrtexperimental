#include <dacrt/dacrt.h>
#include <util/cutimer.h>

extern "C"
__global__ void dacrtBruteForce(TriangleArray dev_triangles, int num_triangles, RayArray dev_rays, int num_rays,
	int* tri_idx_array,	int tricnt,	int* ray_idx_array,	int raycnt,	float* maxts, int* hitids);

extern "C" __global__ void updateMinKernel(int* ray_id, float* min_hits, int* minhit_ids, float* global_min, int* global_hits, int num_rays);

// The kernel loads all triangles and then computes the occupy 
extern "C"
__global__ void trianglePartitionKernel(float3* v0, float3* v1, float3* v2, int* tri_ids, int num_tris, float3 bmin, float3 bmax, int* occupy) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if(tid < num_tris) {
		// this is the more costlier but accurate test version
		float3 centroid = (bmin + bmax) * 0.5f;
		float3 extents = bmax - bmin;
		int triangle_id = tri_ids[tid];
	
	    float triverts[3][3] = {{v0[triangle_id].x, v0[triangle_id].y, v0[triangle_id].z}, 
								{v1[triangle_id].x, v1[triangle_id].y, v1[triangle_id].z}, 
								{v2[triangle_id].x, v2[triangle_id].y, v2[triangle_id].z}};
		float boxhalf[3] = {extents.x * 0.5f, extents.y * 0.5f, extents.z * 0.5f};
		float boxcenter[3] = {centroid.x, centroid.y, centroid.z};
		int o = triBoxOverlap(boxcenter, boxhalf, triverts);
		occupy[tid] = o;
	}
}

extern "C" __global__ void rayPartitionKernel(float3* o, float3* dir, int* ray_ids, int num_rays, float3 bmin, float3 bmax, int* occupy) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if(tid < num_rays) {
		AABB bbox(bmin, bmax);
		int ray_id = ray_ids[tid];
		Ray ray(o[ray_id], dir[ray_id]);
		float thit;
		bool occ = bbox.rayIntersect(ray, thit);
		occupy[tid] = (int)occ;
	}
}

/**
Complete working of dacrt method on the gpu only, with gpu doing work on all the nodes in a sequential manner
We do the spatial splitting method here.!
*/
void gpuDacrtMethod(const AABB& space, TriangleArray& d_triangles, int* dtri_idx_array, int num_triangles, int tpivot, RayArray& d_rays, int* dray_idx_array,
	int num_rays, int rpivot, float* d_maxts, int* d_hitids, DacrtRunTimeParameters& rtparams, Counters& ctr, Logger& logger) {

		if(tpivot < rtparams.PARALLEL_TRI_THRESHOLD || rpivot < rtparams.PARALLEL_RAY_THRESHOLD) {
			
			// call simple brute force kernel on only this node
			Timer bruteforcetimer("bruteforcetimer");
			int NUM_THREADS_PER_BLOCK = rtparams.NUM_RAYS_PER_BLOCK;
			int needed_threads = max(tpivot, rpivot);
			int NUM_BLOCKS = (needed_threads / NUM_THREADS_PER_BLOCK) + (needed_threads % NUM_THREADS_PER_BLOCK != 0);
			bruteforcetimer.start();
			dacrtBruteForce<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(d_triangles, num_triangles, d_rays, num_rays, dtri_idx_array, tpivot, dray_idx_array, rpivot, d_maxts, d_hitids);
			bruteforcetimer.stop();
			ctr.brute_force_time += bruteforcetimer.get_ms();
			

		} else {
			// This part was done initially fully on cpu. Now we are trying to do entirely on gpu

			/// Note: Idea:
			///		  The idea is that all ray and triangle id vectors, actual ray and triangle data are always on the gpu only.
			///		  We do ray tracing as the original dacrt i.e we dont club the segments. All we try to do is parallelize the per node process
			///		  only i.e triangle and ray filtering process
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
				thrust::sort_by_key(thrust::device_ptr<int>(trioccupy), thrust::device_ptr<int>(trioccupy) + tpivot, thrust::device_ptr<int>(dtri_idx_array), thrust::greater<int>());
				//thrust::inclusive_scan(thrust::device_ptr<int>(trioccupy), thrust::device_ptr<int>(trioccupy) + tpivot, thrust::device_ptr<int>(trioccupy));		// do inplace. Order is all screwed up. BEWARE.!!
				//checkCuda(cudaMemcpy((void*)&newtpivot, trioccupy + tpivot - 1, sizeof(int), cudaMemcpyDeviceToHost));
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
				rayPartitionKernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(d_rays.o, d_rays.d, dray_idx_array, rpivot, left.bmin, left.bmax, roccupy);
				rayfiltertimer.stop();
				ctr.ray_filter_time += rayfiltertimer.get_ms();

				raysortbykeytimer.start();
				thrust::sort_by_key(thrust::device_ptr<int>(roccupy), thrust::device_ptr<int>(roccupy) + rpivot, thrust::device_ptr<int>(dray_idx_array), thrust::greater<int>());
				//thrust::inclusive_scan(thrust::device_ptr<int>(roccupy), thrust::device_ptr<int>(roccupy) + rpivot, thrust::device_ptr<int>(roccupy));
				//checkCuda(cudaMemcpy((void*)&newrpivot, roccupy + rpivot - 1, sizeof(int), cudaMemcpyDeviceToHost));
				raysortbykeytimer.stop();
				ctr.raysortbykey_time += raysortbykeytimer.get_ms();
				
				rayreductiontimer.start();
				newrpivot = thrust::reduce(thrust::device_ptr<int>(roccupy) , thrust::device_ptr<int>(roccupy) + rpivot);
				rayreductiontimer.stop();
				ctr.rayreduction_time += rayreductiontimer.get_ms();
				
				//printf("NewTpivot : %d\tNew Rpiovt : %d\n", newtpivot, newrpivot);

				cudaFree(trioccupy);
				cudaFree(roccupy);

				gpuDacrtMethod(left, d_triangles, dtri_idx_array, num_triangles, newtpivot, d_rays, dray_idx_array, num_rays, newrpivot, d_maxts, d_hitids, rtparams, ctr, logger);
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
				thrust::sort_by_key(thrust::device_ptr<int>(trioccupy), thrust::device_ptr<int>(trioccupy) + tpivot, thrust::device_ptr<int>(dtri_idx_array), thrust::greater<int>());
				//thrust::inclusive_scan(thrust::device_ptr<int>(trioccupy), thrust::device_ptr<int>(trioccupy) + tpivot, thrust::device_ptr<int>(trioccupy));
				//checkCuda(cudaMemcpy((void*)&newtpivot, trioccupy + tpivot - 1, sizeof(int), cudaMemcpyDeviceToHost));
				trisortbykeytimer.stop();
				ctr.trisortbykey_time += trisortbykeytimer.get_ms();

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
				thrust::sort_by_key(thrust::device_ptr<int>(roccupy), thrust::device_ptr<int>(roccupy) + rpivot, thrust::device_ptr<int>(dray_idx_array), thrust::greater<int>());
				//thrust::inclusive_scan(thrust::device_ptr<int>(roccupy), thrust::device_ptr<int>(roccupy) + rpivot, thrust::device_ptr<int>(roccupy));
				//checkCuda(cudaMemcpy((void*)&newrpivot, roccupy + rpivot - 1, sizeof(int), cudaMemcpyDeviceToHost));
				raysortbykeytimer.stop();
				ctr.raysortbykey_time += raysortbykeytimer.get_ms();
				
				rayreductiontimer.start();
				newrpivot = thrust::reduce(thrust::device_ptr<int>(roccupy), thrust::device_ptr<int>(roccupy) + rpivot);
				rayreductiontimer.stop();
				ctr.rayreduction_time += rayreductiontimer.get_ms();
				
				//printf("NewTpivot : %d\tNew Rpiovt : %d\n", newtpivot, newrpivot);

				cudaFree(trioccupy);
				cudaFree(roccupy);
				gpuDacrtMethod(right, d_triangles, dtri_idx_array, num_triangles, newtpivot, d_rays, dray_idx_array, num_rays, newrpivot, d_maxts, d_hitids, rtparams, ctr, logger);
			}
		}
}
