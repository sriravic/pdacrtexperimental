#ifndef __DACRT_OBJECT_PARTITIONING_H__
#define __DACRT_OBJECT_PARTITIONING_H__

#pragma once
#include <global.h>
#include <primitives/primitives.h>
#include <dacrt/dacrt_util.h>			// for the counters
#include <util/logger.h>

// Methods used in object partitioning
inline bool splitTriangleLeftX(const float3& position, float median) { return position.x <= median; }
inline bool splitTriangleLeftY(const float3& position, float median) { return position.y <= median; }
inline bool splitTriangleLeftZ(const float3& position, float median) { return position.z <= median; }
inline bool splitTriangleRightX(const float3& position, float median) { return position.x > median; }
inline bool splitTriangleRightY(const float3& position, float median) { return position.y > median; }
inline bool splitTriangleRightZ(const float3& position, float median) { return position.z > median; }

// CPU code
int filterTrianglesObject(AabbArray& tri_aabbs, int count, int* tri_ids, int tpivot, float median, bool (*test)(const float3& , float), int& testcnt);

int filterRaysObject(RayArray& rays, int count, int* ray_ids, const AABB& bounds, int rpivot, int& testcnt);

int filterShadowRaysObject(RayArray& rays, int count, int* ray_ids, int* pray_ids, const AABB& bounds, int rpivot, int& testcnt);

void cpuDacrtObjectPartitioning(const AABB& space, TriangleArray& triangles, AabbArray& tri_aabbs, int num_triangles, int* tri_idx_array, int tpivot,
	RayArray& rays, int num_rays, int* ray_idx_array, int rpivot, float* maxts, int* hitids, Counters& ctr);


// Object partitioning - But brute force done on the gpu
void cpuGpuDacrtObjectPartitioning(const AABB& sceneBox, TriangleArray& triangles, AabbArray& tri_aabbs, RayArray& rays,
	TriangleArray& dev_triangles, RayArray& dev_rays, int num_triangles, int* tri_idx_array, int tpivot, int num_rays, int* ray_idx_array, int rpivot,															
	float* maxts, int* hitids, DacrtRunTimeParameters& rtparams, Counters& ctr, Logger& logger);


#endif