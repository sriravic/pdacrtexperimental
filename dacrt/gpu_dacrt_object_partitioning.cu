#include <dacrt/dacrt.h>
#include <util/cutimer.h>

extern "C"
__global__ void dacrtBruteForce(TriangleArray dev_triangles, int num_triangles, RayArray dev_rays, int num_rays,
	int* tri_idx_array,	int tricnt,	int* ray_idx_array,	int raycnt,	float* maxts, int* hitids);

extern "C" __global__ void updateMinKernel(int* ray_id, float* min_hits, int* minhit_ids, float* global_min, int* global_hits, int num_rays);

extern "C" __global__ void rayPartitionKernel(float3* o, float3* dir, int* ray_ids, int num_rays, float3 bmin, float3 bmax, int* occupy);

