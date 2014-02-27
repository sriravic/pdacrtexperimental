#ifndef __DACRT_SPATIAL_PARTITIONING_H__
#define __DACRT_SPATIAL_PARTITIONING_H__

#pragma once
#include <global.h>
#include <math/tribox.h>
#include <primitives/primitives.h>
#include <primitives/primitives_aos.h>
#include <dacrt/dacrt_util.h>
#include <util/logger.h>

#define TRI_TEST_THRESHOLD 1000

// CPU code
int filterRaysSpatial(RayArray& rays, int count, int* ray_ids, const AABB& bounds, int rpivot, int& testcnt);
int filterTrianglesSpatial(TriangleArray& triangles, int count, int* tri_ids, const AABB& bounds, int tpivot, int splitaxis, int& testcnt);
void cpuDacrtSpatialPartitioning(const AABB& space, TriangleArray& triangles, int num_triangles, int* tri_idx_array, int tpivot,
	RayArray& rays, int num_rays, int* ray_idx_array, int rpivot, float* maxts, int* hitids, Counters& ctr);

void cpuDacrtSpatialShadows(const AABB& space, TriangleArray& triangles, int num_triangles, int* tri_idx_array, int tpivot, 
	RayArray& rays, int num_rays, int* ray_idx_array, int* pray_idx_array, int rpivot, bool* shadows, Counters& ctr);

// cpu partitioning - brute force on the gpu
void cpuGpuDacrtSpatialPartitioning(const AABB& sceneBox, TriangleArray& triangles, AabbArray& tri_aabbs, RayArray& rays, TriangleArray& dev_triangles, RayArray& dev_rays,
	int num_triangles, int* tri_idx_array, int tpivot, int num_rays, int* ray_idx_array, int rpivot, float* maxts, int* hitids, DacrtRunTimeParameters& rtparams,
	Counters& ctr, Logger& logger);

void cpuGpuDacrtSpatialPartitioningModified(const AABB& sceneBox, TriangleArray& triangles, AabbArray& tri_aabbs, RayArray& rays, TriangleArray& dev_triangles, RayArray& dev_rays,
	int num_triangles, int* tri_idx_array, int tpivot, int num_rays, int* ray_idx_array, int rpivot, float* maxts, int* hitids, DacrtRunTimeParameters& rtparams,
	Counters& ctr, Logger& logger);



// hybrid methods
/// TODO : Modify the code below to get method input params for both cpu and gpu.
void cpuGpuDacrtBranched(AABB& root, TriangleArray& hTriangles, TriangleArray& dTriangles,
	int* h_tri_idx_array, int* d_tri_idx_array, int num_triangles, int tpivot, RayArray& hRays, RayArray& dRays, int* h_ray_idx_array, int* d_ray_idx_array,
	int num_rays, int rpivot, float* h_maxts, int* h_hitids, DacrtRunTimeParameters& rtparams, Counters& ctr, Logger& logger);


// pure gpu methods
void gpuDacrtMethod(const AABB& space, TriangleArray& d_triangles, int* dtri_idx_array, int num_triangles, int tpivot, RayArray& d_rays, int* dray_idx_array,
	int num_rays, int rpivot, float* d_maxts, int* d_hitids, DacrtRunTimeParameters& rtparams, Counters& ctr, Logger& logger);

void gpuDacrtSpatialSegmented(const AABB& sceneBox, TriangleArray& d_triangles, int* dtri_idx_array, int num_triangles, int tpivot,
	RayArray& d_rays, int* dray_idx_array, int num_rays, int rpivot, float* h_maxts, int* h_hitids, DacrtRunTimeParameters& rtparams, Counters& ctr, Logger& logger);
	
void gpuDacrtCell(AABB& root, TriangleArray& d_triangles, int* tri_idx_array, int tpivot, RayArray& rays, int* ray_idx_array, int rpivot, 
	float* h_maxts, int* h_hitids, DacrtRunTimeParameters& rtparams, Counters& ctr, Logger& logger);

void gpuDacrtFullyParallel(AABB& root, TriangleArray& d_triangles, int* tri_idx_array, int num_triangles, int tpivot, RayArray& d_rays, int* ray_idx_array, int num_rays, int rpivot, 
	float* h_maxts, int* h_hitids, DacrtRunTimeParameters& rtparams, Counters& ctr, Logger& logger);

void gpuDacrtFullyParallelModified(AABB& root, TriangleArray& d_triangles, int* tri_idx_array, int num_triangles, int tpivot, RayArray& d_rays, int* ray_idx_array, int num_rays, int rpivot, 
	float* h_maxts, int* h_hitids, DacrtRunTimeParameters& rtparams, Counters& ctr, Logger& logger);

void gpuDacrtFullyCuda( AABB& root, TriangleArray& d_triangles, int* tri_idx_array, int num_triangles, int tpivot, RayArray& d_rays, int* ray_idx_array, int num_rays, int rpivot, 
	float* h_maxts, int* h_hitids, DacrtRunTimeParameters& rtparams, Counters& ctr, Logger& logger);


// pure gpu but AoS methods
void gpuDacrtSpatialAosMethod(const AABB4& space, TriangleArrayAos& d_triangles, int* dtri_idx_array, int num_triangles, int tpivot, RayArrayAos& d_rays, int* dray_idx_array,
	int num_rays, int rpivot, float* d_maxts, int* d_hitids, DacrtRunTimeParameters& rtparams, Counters& ctr, Logger& logger);

#endif