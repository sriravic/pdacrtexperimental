/**
Fully cuda implementation of the dacrt algorithm but with object partitioning
*/
#include <dacrt/dacrt.h>
#include <util/cutimer.h>
#include <util/util.h>

extern "C" __global__ void modifiedSegmentedBruteForce(RayArray rays, TriangleArray triangles, int* buffered_ray_ids, int ray_buffer_occupied, int* buffered_tri_ids, int tri_buffer_occupied,
													   int* ray_segment_sizes, int* tri_segment_sizes, int* ray_segment_start, int* tri_segment_start, 
													   int* segment_no, int* blockStart, float* maxts, int* hitids);
extern "C" void completeBruteForceModified(ParallelPackModified& pack, TriangleArray& d_triangles, RayArray& d_rays, DacrtRunTimeParameters& rtparams, Counters& ctr);
extern "C" __global__ void parallelRayFilter(float3* origin, float3* direction, AABB* boxes, int* ray_ids, unsigned int* keys, unsigned int* segment_no, unsigned int* segment_block_no, unsigned int* segment_sizes, unsigned int* rayoffsets, int* rsegment_filter_status, unsigned int total_elements, unsigned int depth);

extern "C" 
__global__ void parallelTriFilterObject(float3* v0, float3* v1, float3* v2, AABB* boxes, int* tri_ids, uint* keys, uint* segment_no, uint* segment_block_no, unsigned* segment_sizes, uint* trioffsets, int* tsegment_filter_status, uint total_elements, uint depth) {




}
