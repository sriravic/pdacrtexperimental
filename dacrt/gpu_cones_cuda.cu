#include <dacrt/dacrt.h>
#include <util/cutimer.h>
#include <util/util.h>

// extern declarations
extern "C" __global__ void modifiedSegmentedBruteForce(RayArray rays, TriangleArray triangles, int* buffered_ray_ids, int ray_buffer_occupied, int* buffered_tri_ids, int tri_buffer_occupied,
													   int* ray_segment_sizes, int* tri_segment_sizes, int* ray_segment_start, int* tri_segment_start, 
													   int* segment_no, int* blockStart, float* maxts, int* hitids);

extern "C" void completeBruteForceModified(ParallelPackModified& pack, TriangleArray& d_triangles, RayArray& d_rays, DacrtRunTimeParameters& rtparams, Counters& ctr);
extern "C" __global__ void parallelTriFilter(float3* v0, float3* v1, float3* v2, AABB* boxes, int* tri_ids, uint* keys, uint* segment_no, uint* segment_block_no, unsigned* segment_sizes, uint* trioffsets, int* tsegment_filter_status, uint total_elements, uint depth);
extern "C" __global__ void parallelRayFilter(float3* origin, float3* direction, AABB* boxes, int* ray_ids, uint* keys, uint* segment_no, uint* segment_block_no, uint* segment_sizes, uint* rayoffsets, int* rsegment_filter_status, uint total_elements, uint depth);

__device__ __host__ bool coneLineIntersect(const Cone& cone, const Edge& e) {
	float  AdD = dot((cone.dir), (e.edge));
	float  cosSqr = cos((float)cone.angle) * cos((float)cone.angle);
	float3 E = (e.v[0]) - (cone.apex);
	float  AdE = dot((cone.dir), E); 
	float  DdE = dot((e.edge), E); 
	float  EdE = dot(E, E); 
	float  c2 = AdD*AdD - cosSqr;
	float  c1 = AdD*AdE - cosSqr*DdE;
	float  c0 = AdE*AdE - cosSqr*EdE;
	float  fdot;
	int    mQuantity;
	float3 mPoint[2];
	if (fabs(c2) >= ZERO_TOLERANCE) {
		// c2 != 0
		float discr = c1*c1 - c0*c2;
		if (discr < (float)0) {
			// Q(t) = 0 has no real-valued roots.  The line does not
			// intersect the float-sided cone.
			return false;
		} else if (discr > ZERO_TOLERANCE) {
			// Q(t) = 0 has two distinct real-valued roots.  However, one or
			// both of them might intersect the portion of the float-sided
			// cone "behind" the vertex.  We are interested only in those
			// intersections "in front" of the vertex.
			float root = sqrt(discr);
			float invC2 = ((float)1)/c2;
			mQuantity = 0;

			float t = (-c1 - root)*invC2;
			mPoint[mQuantity] = (e.v[0]) + t*(e.edge);
			E = mPoint[mQuantity] - (cone.apex);
			fdot = dot((cone.dir), E);
			if (fdot > (float)0) {
				return true;
			}

			t = (-c1 + root)*invC2;
			mPoint[mQuantity] = (e.v[0]) + t*(e.edge);
			E = mPoint[mQuantity] - (cone.apex);
			fdot = dot((cone.dir), E);
			if (fdot > (float)0) {
				return true;
			}
			return false;
		} else {
			// One repeated real root (line is tangent to the cone).
			mPoint[0] = (e.v[0]) - (c1/c2)*(e.edge);
			E = mPoint[0] - (cone.apex);
			if (dot(E, (cone.dir)) > (float)0) {
				return true;
			}
			else {
				return false;
			}
		}
	} else if (fabs(c1) >= ZERO_TOLERANCE) {
		// c2 = 0, c1 != 0 (D is a direction vector on the cone boundary)
		mPoint[0] = (e.v[0]) - (((float)0.5)*c0/c1)*(e.edge);
		E = mPoint[0] - (cone.apex);
		fdot = dot(E, (cone.dir));
		if (fdot > (float)0) {
			return true;
		} else {
			return false;
		}
	} else if (fabs(c0) >= ZERO_TOLERANCE) {
		// c2 = c1 = 0, c0 != 0
		return false;
	} else {
		// c2 = c1 = c0 = 0, cone contains ray V+t*D where V is cone vertex
		// and D is the line direction.
		return true;
	}
}

__device__ __host__ bool coneBoxIntersect(const Cone& C, const AABB& b, Edge e[12]) {
	float hit = 0;
	if(b.rayIntersect(Ray(C.apex, C.dir), hit)) {
		return true;
	} else {
#pragma unroll
		for(size_t i = 0; i < 12; i++) {
			if(coneLineIntersect(C, e[i])) {
				return true;
			}
		}
	}
	return false;
}

extern "C" __global__ void parallelConeFilter(CameraConeArray cones, AABB* boxes, uint* keys, uint* segment_no, uint* segment_block_no, uint* segment_sizes, uint* cone_offsets, uint total_elements, uint depth) {
	unsigned int segment				= segment_no[blockIdx.x];
	unsigned int block_within_segment	= segment_block_no[blockIdx.x];
	unsigned int num_elements			= segment_sizes[segment];
	unsigned int offset					= cone_offsets[segment];
	unsigned int tid_within_segment		= block_within_segment * 256 + threadIdx.x;
	unsigned int tid					= offset + tid_within_segment;
	if(tid_within_segment < num_elements) {
		AABB lbox, rbox;
		AABB parent			= boxes[segment];
		int cid = cones.cone_id_array[tid];
		Cone c(cones.cones.apex_array[cid], cones.cones.cone_dir_array[cid], cones.cones.angle_array[cid]);
		splitSpatialMedian(parent, lbox, rbox);
		// form the Edge structure
		float3 vertices[16] = {
			(lbox.bmin), 
			make_float3(lbox.bmin.x, lbox.bmax.y, lbox.bmin.z), 
			make_float3(lbox.bmax.x, lbox.bmax.y, lbox.bmin.z), 
			make_float3(lbox.bmax.x, lbox.bmin.y, lbox.bmin.z),
			make_float3(lbox.bmin.x, lbox.bmin.y, lbox.bmax.z), 
			make_float3(lbox.bmin.x, lbox.bmax.y, lbox.bmax.z), 
			(lbox.bmax), 
			make_float3(lbox.bmax.x, lbox.bmin.y, lbox.bmax.z),
			(rbox.bmin), 
			make_float3(rbox.bmin.x, rbox.bmax.y, rbox.bmin.z), 
			make_float3(rbox.bmax.x, rbox.bmax.y, rbox.bmin.z), 
			make_float3(rbox.bmax.x, rbox.bmin.y, rbox.bmin.z),
			make_float3(rbox.bmin.x, rbox.bmin.y, rbox.bmax.z), 
			make_float3(rbox.bmin.x, rbox.bmax.y, rbox.bmax.z), 
			(rbox.bmax), 
			make_float3(rbox.bmax.x, rbox.bmin.y, rbox.bmax.z)
		};

		Edge edges1[12] = {
			Edge(normalize(vertices[1] - vertices[0]), vertices[0], vertices[1]),
			Edge(normalize(vertices[2] - vertices[1]), vertices[1], vertices[2]),
			Edge(normalize(vertices[3] - vertices[2]), vertices[2], vertices[3]),
			Edge(normalize(vertices[0] - vertices[3]), vertices[3], vertices[0]),
			Edge(normalize(vertices[5] - vertices[4]), vertices[4], vertices[5]),
			Edge(normalize(vertices[6] - vertices[5]), vertices[5], vertices[6]),
			Edge(normalize(vertices[7] - vertices[6]), vertices[6], vertices[7]),
			Edge(normalize(vertices[4] - vertices[7]), vertices[7], vertices[4]),
			Edge(normalize(vertices[0] - vertices[4]), vertices[4], vertices[0]),
			Edge(normalize(vertices[5] - vertices[1]), vertices[1], vertices[5]),
			Edge(normalize(vertices[3] - vertices[7]), vertices[7], vertices[3]),
			Edge(normalize(vertices[6] - vertices[2]), vertices[2], vertices[6])
		};

			// right box edges
		Edge edges2[12] = {
			Edge(normalize(vertices[9] - vertices[8]),   vertices[8],  vertices[9]),
			Edge(normalize(vertices[10] - vertices[9]),  vertices[9],  vertices[10]),
			Edge(normalize(vertices[11] - vertices[10]), vertices[12], vertices[11]),
			Edge(normalize(vertices[8] - vertices[11]),  vertices[11], vertices[8]),
			Edge(normalize(vertices[13] - vertices[12]), vertices[12], vertices[13]),
			Edge(normalize(vertices[14] - vertices[13]), vertices[13], vertices[14]),
			Edge(normalize(vertices[15] - vertices[14]), vertices[14], vertices[15]),
			Edge(normalize(vertices[12] - vertices[15]), vertices[15], vertices[12]),
			Edge(normalize(vertices[8] - vertices[12]),  vertices[12], vertices[8]),
			Edge(normalize(vertices[13] - vertices[9]),  vertices[9],  vertices[13]),
			Edge(normalize(vertices[11] - vertices[15]), vertices[15], vertices[11]),
			Edge(normalize(vertices[14] - vertices[10]), vertices[10], vertices[14]),
		};

		bool lo = coneBoxIntersect(c, lbox, edges1);
		bool ro = coneBoxIntersect(c, rbox, edges2);
		unsigned int val	= lo && ro ? 1 : (lo ? 0 : (ro ? 2 : 3));
		unsigned int key	= ((segment + 1) << 2) | val;
		keys[tid]			= key;


	}
}