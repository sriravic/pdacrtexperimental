#include <dacrt/dacrt.h>
#include <util/renderdata.h>

// This is the central location for all the kernel calls
// Method determines which function/kernel to call
// It is assumed that called has already reset all the values of the ctr so that following code can work seamlessly
void dacrt(void* data, Method method, DacrtRunTimeParameters& rtparams, Counters& ctr, Logger& logger) {
	switch(method) {
	case CPU_OBJECT_PRIMARY:
		{
			// calls the cpu code with object partitioning for primary rays
			CpuPrimaryPassRenderData *pdata = (CpuPrimaryPassRenderData*)data;
			cpuDacrtObjectPartitioning(pdata->space, pdata->triangles, pdata->tri_aabbs, pdata->num_triangles, pdata->tri_idx_array, pdata->num_triangles,
				pdata->rays, pdata->num_rays, pdata->ray_idx_array, pdata->num_rays, pdata->maxts, pdata->hitids, ctr);
		}
		break;
	case CPU_SPATIAL_PRIMARY:
		{
			CpuPrimaryPassRenderData *pdata = (CpuPrimaryPassRenderData*)data;
			cpuDacrtSpatialPartitioning(pdata->space, pdata->triangles, pdata->num_triangles, pdata->tri_idx_array, pdata->num_triangles,
				pdata->rays, pdata->num_rays, pdata->ray_idx_array, pdata->num_rays, pdata->maxts, pdata->hitids, ctr);			
		}
		break;
	case CPU_SPATIAL_SHADOW:
		{
			CpuShadowPassData *pdata = (CpuShadowPassData*)data;
			cpuDacrtSpatialShadows(pdata->space, pdata->triangles, pdata->num_triangles, pdata->tri_idx_array, pdata->tpivot, pdata->rays,
				pdata->num_rays, pdata->sray_idx_array, pdata->pray_idx_array, pdata->rpivot, pdata->shadows, ctr);
		}
		break;
	case CPU_GPU_OBJECT_PRIMARY:
		{
			CpuGpuPrimaryPassRenderData* pdata = (CpuGpuPrimaryPassRenderData*)data;
			/*cpuGpuDacrtObjectPartitioning(pdata->space, pdata->triangles, pdata->tri_aabbs, pdata->rays, pdata->dev_triangles, pdata->dev_rays,
				pdata->num_triangles, pdata->tri_idx_array, pdata->num_triangles, pdata->num_rays, pdata->ray_idx_array, pdata->num_rays, pdata->maxts,
				pdata->hitids, rtparams, ctr, logger);
				*/
		}
		break;
	case CPU_GPU_SPATIAL_PRIMARY:
		{
			CpuGpuPrimaryPassRenderData *pdata = (CpuGpuPrimaryPassRenderData*)data;
			/*cpuGpuDacrtSpatialPartitioning(pdata->space, pdata->triangles, pdata->tri_aabbs, pdata->rays, pdata->dev_triangles, pdata->dev_rays,
				pdata->num_triangles, pdata->tri_idx_array, pdata->num_triangles, pdata->num_rays, pdata->ray_idx_array, pdata->num_rays, pdata->maxts,
				pdata->hitids, rtparams, ctr, logger);
				*/
		}
		break;
	case CPU_GPU_SPATIAL_MODIFIED:
		{
			CpuGpuPrimaryPassRenderData *pdata = (CpuGpuPrimaryPassRenderData*)data;
			//cpuGpuDacrtSpatialPartitioningModified(pdata->space, pdata->triangles, pdata->tri_aabbs, pdata->rays, pdata->dev_triangles, pdata->dev_rays,
			//	pdata->num_triangles, pdata->tri_idx_array, pdata->num_triangles, pdata->num_rays, pdata->ray_idx_array, pdata->num_rays, pdata->maxts,
			//	pdata->hitids, rtparams, ctr, logger);
		}
		break;
	case CPU_GPU_DBUFFER_SPATIAL_PRIMARY:
		{

		}
		break;
	case GPU_SPATIAL_PRIMARY:
		{
			GpuPrimaryPassRenderData* pdata = (GpuPrimaryPassRenderData*)data;
			printf("DEBUG : Num triangles called : %d\n", pdata->num_triangles);
			/*gpuDacrtMethod(pdata->space, pdata->d_triangles, pdata->dtri_idx_array, pdata->num_triangles, pdata->num_triangles, pdata->d_rays,
				pdata->dray_idx_array, pdata->num_rays, pdata->num_rays, pdata->d_maxts, pdata->d_hitids, rtparams, ctr, logger);
				*/
		}
		break;
	case GPU_SPATIAL_PRIMARY_SEGMENTED:
		{
			/// NOTE: The pdata->d_maxts and pdata->d_hitids below are actually host allocated values. This convention has been used so that 
			///		  there is no actual structure change. The architecture of the renderer requests a valid pointer with the d_ suffix name
			///		  and hence. So dont be confused. The memory for device maxts and hitids are actually allocated within the called function
			///		  The callee in the renderer 'NEED NOT' allocate memory and should pass in host values only.!!
			GpuPrimaryPassRenderData* pdata = (GpuPrimaryPassRenderData*)data;
			/*gpuDacrtSpatialSegmented(pdata->space, pdata->d_triangles, pdata->dtri_idx_array, pdata->num_triangles, pdata->num_triangles,
				pdata->d_rays, pdata->dray_idx_array, pdata->num_rays, pdata->num_rays, pdata->d_maxts, pdata->d_hitids, rtparams, ctr, logger);
				*/
		}
		break;
	case GPU_SPATIAL_CELL:
		{
			GpuCellData *pdata = (GpuCellData*)data;
			/*
			gpuDacrtCell(pdata->scene_box, pdata->d_triangles, pdata->dtri_idx_array, pdata->tpivot, pdata->d_rays, pdata->dray_idx_array, pdata->rpivot, 
				pdata->h_maxts, pdata->h_hitids, rtparams, ctr, logger);
				*/
		}
		break;
	case GPU_SPATIAL_FULLY_PARALLEL:
		{
			GpuFullyParallelData *pdata = (GpuFullyParallelData*)data;
			/*gpuDacrtFullyParallel(pdata->scene_box, pdata->d_triangles, pdata->dtri_idx_array, pdata->num_triangles, pdata->tpivot, pdata->d_rays, pdata->dray_idx_array,
				pdata->num_rays, pdata->rpivot, pdata->h_maxts, pdata->h_hitids, rtparams, ctr, logger);
				*/
			
		}
		break;
	case GPU_SPATIAL_FULLY_PARALLEL_MODIFIED:
		{
			GpuFullyParallelData *pdata = (GpuFullyParallelData*)data;
			gpuDacrtFullyParallelModified(pdata->scene_box, pdata->d_triangles, pdata->dtri_idx_array, pdata->num_triangles, pdata->tpivot, pdata->d_rays, pdata->dray_idx_array,
				pdata->num_rays, pdata->rpivot, pdata->h_maxts, pdata->h_hitids, rtparams, ctr, logger);
		}
		break;
	case GPU_DACRT_FULLY_CUDA:
	case GPU_DACRT_FULLY_CUDA_SHADOW:
		{
			GpuFullyParallelData *pdata = (GpuFullyParallelData*)data;
			gpuDacrtFullyCuda(pdata->scene_box, pdata->d_triangles, pdata->dtri_idx_array, pdata->num_triangles, pdata->tpivot, pdata->d_rays, pdata->dray_idx_array,
				pdata->num_rays, pdata->rpivot, pdata->h_maxts, pdata->h_hitids, rtparams, ctr, logger);
		}
		break;
	case CPU_GPU_TWIN_TREES:
		{
			CpuGpuTwinData *pdata = (CpuGpuTwinData*)data;
			/*cpuGpuDacrtBranched(pdata->scene_box, pdata->h_triangles, pdata->d_triangles, pdata->htri_idx_array, pdata->dtri_idx_array, pdata->num_triangles,
				pdata->tpivot, pdata->h_rays, pdata->d_rays, pdata->hray_idx_array, pdata->dray_idx_array, pdata->num_rays, pdata->rpivot, pdata->h_maxts, pdata->h_hitids,
				rtparams, ctr, logger);
				*/

		}
		break;

		// AoS methods come here
	case GPU_SPATIAL_PRIMARY_AOS:
		{
			GpuPrimaryPassRenderDataAos* pdata = (GpuPrimaryPassRenderDataAos*)data;
			//gpuDacrtSpatialAosMethod(pdata->space, pdata->d_triangles, pdata->dtri_idx_array, pdata->num_triangles, pdata->num_triangles, pdata->d_rays,
			//	pdata->dray_idx_array, pdata->num_rays, pdata->num_rays, pdata->d_maxts, pdata->d_hitids, rtparams, ctr, logger);
		}
		break;
	}

}


// cpu shade functions
void cpuShade(const PointLight& pl, RayArray& rays, int width, int height, int raycnt, TriangleArray& triangles, int tricnt, float* maxts, int* hitids, float3* buffer,
	bool* shadow_hits, int num_shadow_rays, int* shadow_ray_idx, int* pray_association, MortonCode* mcodes, bool enable_shadows, bool use_morton_codes) {
		size_t hitpixels = 0;
//#pragma omp parallel for
		for(int r = 0; r < raycnt; r++) {
			float3 c = make_float3(0, 0, 0);
			int pixloc = r;
			if(use_morton_codes) pixloc = mcodes[r].x + mcodes[r].y * width;
			if(hitids[r] != -1) {
				// get the ray
				hitpixels++;
				Ray ir(rays.o[r], rays.d[r]);
				Triangle it(triangles.v0[hitids[r]], triangles.v1[hitids[r]], triangles.v2[hitids[r]]);
				float3 xt_pt = ir(maxts[r]);
				float3 normal = computeNormal(it);
				c = std::abs(dot(normal, normalize(xt_pt-pl.position))) * make_float3(0.80f, 0.50f, 0.20f);
			}
			buffer[pixloc] = c;
		}

		// now add the shadow contribution
		int debugcnt = 0;
		if(enable_shadows) {
			for(int i = 0; i < num_shadow_rays; i++) {
				// we are calculating like this because, the ids can be mangled due to the nature of the dacrt algorithm
				// order is not maintained.!
				
				int sray_id = shadow_ray_idx[i];
				if(shadow_hits[sray_id] == true) {
					// get the corresponding primary ray associated
					int pray_id = pray_association[i];
					// get corresponding pixloc in case of morton codes
					int pixloc = pray_id;
					if(use_morton_codes) pixloc = mcodes[pray_id].x + mcodes[pray_id].y * width;
					buffer[pray_id] *= 0.2f;			// make it a dull color
					debugcnt++;
				}
			}
		}
		std::cout<<"Pixel coverage : "<<static_cast<float>(hitpixels)/static_cast<float>(raycnt) * 100.0f<<" %\n";
}


void cpuShadeAos(const PointLight& pl, RayArrayAos& rays, int width, int height, int raycnt, TriangleArrayAos& triangles, int tricnt, float* maxts, int* hitids, float3* buffer,
	bool* shadow_hits, int num_shadow_rays, int* shadow_ray_idx, int* pray_association, MortonCode* mcodes, bool enable_shadows, bool use_morton_codes) {
//#pragma omp parallel for
		for(int r = 0; r < raycnt; r++) {
			float3 c = make_float3(0, 0, 0);
			int pixloc = r;
			if(use_morton_codes) pixloc = mcodes[r].x + mcodes[r].y * width;
			if(hitids[r] != -1) {
				// get the ray
				Ray4 ir = rays.rays[r];
				Triangle3 it = triangles.triangles[hitids[r]];
				float4 xt_pt = ir(maxts[r]);
				float3 normal = computeNormal(it);
				c = std::abs(dot(normal, normalize(make_float3(xt_pt.x, xt_pt.y, xt_pt.z)-pl.position))) * make_float3(0.80f, 0.50f, 0.20f);
			}
			buffer[pixloc] = c;
		}

		// now add the shadow contribution
		int debugcnt = 0;
		if(enable_shadows) {
			for(int i = 0; i < num_shadow_rays; i++) {
				// we are calculating like this because, the ids can be mangled due to the nature of the dacrt algorithm
				// order is not maintained.!
				
				int sray_id = shadow_ray_idx[i];
				if(shadow_hits[sray_id] == true) {
					// get the corresponding primary ray associated
					int pray_id = pray_association[i];
					// get corresponding pixloc in case of morton codes
					int pixloc = pray_id;
					if(use_morton_codes) pixloc = mcodes[pray_id].x + mcodes[pray_id].y * width;
					buffer[pray_id] *= 0.2f;			// make it a dull color
					debugcnt++;
				}
			}
		}
		std::cout<<"DEBUG : "<<debugcnt<<"\n";
}

// ambient occlusion shade function
void cpuShadeAmbientOcclusion(int width, int height, int raycnt, int* hitids, float3* buffer,
	int num_candidates, int num_samples, float* ao, int* ao_rays_id, int* pray_association) {

		for(int i = 0; i < raycnt; i++) {
			float3 c = make_float3(0, 0, 0);
			if(hitids[i] != -1) {
				c = make_float3(1.0f, 1.0f, 1.0f);
			}
			buffer[i] = c;
		}
		// now fill in ao stuff
		for(int i = 0; i < num_candidates; i++) {
			//std::cout<<i<<"\n";
			int ao_id = ao_rays_id[i*num_samples];
			int pray_id = pray_association[i*num_samples];
			buffer[pray_id] *= (1.0f - ao[i]);
		}
}