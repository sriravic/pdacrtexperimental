#include <dacrt/dacrt.h>
#include <dacrt/csd.h>
#include <dacrt/kdtree.h>
#include <util/cutimer.h>
#include <util/util.h>

#define MAX_BLOCKS_X 1024

extern "C" __global__ void modifiedSegmentedBruteForce(RayArray rays, TriangleArray triangles, int* buffered_ray_ids, int ray_buffer_occupied, int* buffered_tri_ids, int tri_buffer_occupied,
													   int* ray_segment_sizes, int* tri_segment_sizes, int* ray_segment_start, int* tri_segment_start, 
													   int* segment_no, int* blockStart, float* maxts, int* hitids);

extern "C" void completeBruteForceModified(ParallelPackModified& pack, TriangleArray& d_triangles, RayArray& d_rays, DacrtRunTimeParameters& rtparams, Counters& ctr);
extern "C" __global__ void updateMinKernel(int* ray_id, float* min_hits, int* minhit_ids, float* global_min, int* global_hits, int num_rays);

// NOTE keys are one to one mapping between rays and themselves. They are not dependent on ray_id but on their own they dont make sense!! So understand this relation before you proceed.
extern "C"
__global__ void parallelRayFilter(float3* origin, float3* direction, AABB* boxes, int* ray_ids, unsigned int* keys, unsigned int* segment_no, unsigned int* segment_block_no, unsigned int* segment_sizes, unsigned int* rayoffsets, int* rsegment_filter_status, unsigned int total_elements, unsigned int depth, unsigned int total_valid_blocks) {
	if((blockIdx.x + blockIdx.y * gridDim.x) < total_valid_blocks) {
		unsigned int segment				= segment_no[blockIdx.x + blockIdx.y * gridDim.x];
		unsigned int block_within_segment	= segment_block_no[blockIdx.x + blockIdx.y * gridDim.x];
		unsigned int num_elements			= segment_sizes[segment];
		unsigned int offset					= rayoffsets[segment];
		unsigned int tid_within_segment		= block_within_segment * 256 + threadIdx.x;
		unsigned int tid					= offset + tid_within_segment;
		//unsigned int tid					= threadIdx.x + blockIdx.x * blockDim.x;
		if(tid_within_segment < num_elements) {
			// load the appropriate element and then proceed
			// in case of AOS, here we have to add the load code + plus shared memory stuff. For SOA we dont need shared memory
			float hit;
			int rid = ray_ids[tid];
			Ray ir(origin[rid], direction[rid]);
			AABB lbox, rbox;
			AABB parent			= boxes[segment];
			float split_pos;
			splitSpatialMedian(parent, lbox, rbox, split_pos);
			bool lo				= lbox.rayIntersect(ir, hit);
			bool ro				= rbox.rayIntersect(ir, hit);
			unsigned int val	= lo && ro ? 1 : (lo ? 0 : (ro ? 2 : 3));
			// do an OR with the result(val) and store it.!
			/// KEY IDEA: Well the initial idea to launch separate kernels to fill keys is not correct because we already have split boxes inside the aabb buffer, but segments will always be /2 of them.
			///           This is because left/right form one segment. So we cant do that.
			unsigned int key	= ((segment + 1) << 2) | val;
			keys[tid]			= key;
		}
	}
}

extern "C"
__global__ void parallelTriFilter(float3* v0, float3* v1, float3* v2, AABB* boxes, AABB* children, int* tri_ids, 
                                  unsigned int* keys, unsigned int* segment_no, unsigned int* segment_block_no, 
								  unsigned* segment_sizes, unsigned int* trioffsets, int* tsegment_filter_status, 
								  unsigned int* split_axis, float* split_pos,						// locate where you split
								  unsigned int total_elements, unsigned int depth, 
								  unsigned int total_valid_blocks									// indicates how many are valid blocks since we are launching a two dimensional array
								  ) {

	if((blockIdx.x + blockIdx.y * gridDim.x) < total_valid_blocks) {
		unsigned int segment				= segment_no[blockIdx.x + blockIdx.y * gridDim.x];
		unsigned int block_within_segment	= segment_block_no[blockIdx.x + blockIdx.y * gridDim.x];
		unsigned int num_elements			= segment_sizes[segment];
		unsigned int offset				    = trioffsets[segment];
		unsigned int tid_within_segment		= block_within_segment * 256 + threadIdx.x;
		unsigned int tid					= offset + tid_within_segment;
		//unsigned int tid					= threadIdx.x + blockIdx.x * blockDim.x;
		if(tid_within_segment < num_elements) {
			int triangle_id = tri_ids[tid];
			Triangle it(v0[triangle_id], v1[triangle_id], v2[triangle_id]);
			AABB left, right;
			AABB parent				= boxes[segment];
			float splitpos;
			int axis = splitSpatialMedian(parent, left, right, splitpos);
			float3 trimin = fminf(it.v[0], fminf(it.v[1], it.v[2]));
			float3 trimax = fmaxf(it.v[0], fmaxf(it.v[1], it.v[2]));
			float triboxmin[3] = {trimin.x, trimin.y, trimin.z};
			float triboxmax[3] = {trimax.x, trimax.y, trimax.z};
			float lboxmin[3] = {left.bmin.x, left.bmin.y, left.bmin.z};
			float lboxmax[3] = {left.bmax.x, left.bmax.y, left.bmax.z};
			float rboxmin[3] = {right.bmin.x, right.bmin.y, right.bmin.z};
			float rboxmax[3] = {right.bmax.x, right.bmax.y, right.bmax.z};
			/*
			float3 lcentroid		= (left.bmin + left.bmax) * 0.5f;
			float3 rcentroid		= (right.bmin + right.bmax) * 0.5f;
			float3 lextents			= left.bmax - left.bmin;
			float3 rextents			= right.bmax - right.bmin;
		
	
			float triverts[3][3]	= {{v0[triangle_id].x, v0[triangle_id].y, v0[triangle_id].z}, 
									   {v1[triangle_id].x, v1[triangle_id].y, v1[triangle_id].z}, 
									   {v2[triangle_id].x, v2[triangle_id].y, v2[triangle_id].z}};
			float lboxhalf[3]		= {lextents.x * 0.5f, lextents.y * 0.5f, lextents.z * 0.5f};
			float rboxhalf[3]		= {rextents.x * 0.5f, rextents.y * 0.5f, rextents.z * 0.5f};
			float lboxcenter[3]		= {lcentroid.x, lcentroid.y, lcentroid.z};
			float rboxcenter[3]		= {rcentroid.x, rcentroid.y, rcentroid.z};
			/// TODO: Can we replace this costly test with the simpler test? Any jump in total performance and not only this small step.
			//int lo			 = triBoxOverlap(lboxcenter, lboxhalf, triverts);
			//int ro			 = triBoxOverlap(rboxcenter, rboxhalf, triverts);
			*/
			int lo = triBoxOverlapSimple(triboxmin, triboxmax, lboxmin, lboxmax, axis);
			int ro = triBoxOverlapSimple(triboxmin, triboxmax, rboxmin, rboxmax, axis);
			int val			 = lo && ro ? 1 : (lo ? 0 : 2);
			unsigned int key = ((segment + 1) << 2)|val;
			keys[tid]		 = key;
			if(tid_within_segment == 0) {
				// only one thread for the entire segment is to write this
				split_axis[segment] = axis;
				split_pos[segment] = splitpos;
				children[2*segment] = left;
				children[2*segment+1] = right;
			}
		}
	}
}


// util function to display memory usage
extern "C"
void memoryusage(size_t memory_in_bytes) {
	printf("Memory usage : %u bytes", memory_in_bytes);
}

// Level represents one level in the implicit hierarchy
// we can pointers in the structure to either device/host memory
struct Level
{
	int		depth;
	AABB*	node_aabbs;
	uint2*	tpivots;
	uint2*	rpivots;
	uint*	tsegment_sizes;
	uint*	rsegment_sizes;
	int*	tri_idx;
	int*	ray_idx;
	uint	num_nodes;					// number of active nodes in this level
	uint    num_tris;					// number of all triangles within all the active nodes at this level
	uint    num_rays;					// number of all rays within all the active nodes at this level
	Level() {
		depth							= -1;
		node_aabbs						= NULL;
		tpivots = rpivots				= NULL;
		tsegment_sizes = rsegment_sizes = NULL;
		tri_idx = ray_idx				= NULL;
		num_nodes						= 0;
		num_tris						= 0;
		num_rays						= 0;
	}

	Level(const Level& L) {
		// for now allocate all pivots and segment sizes in host memory
		// and tri_idx and ray_idx in device memory
		depth			= L.depth;
		num_nodes		= L.num_nodes;
		num_tris		= L.num_tris;
		num_rays		= L.num_rays;
		node_aabbs		= new AABB[num_nodes];
		tpivots			= new uint2[num_nodes];
		rpivots			= new uint2[num_nodes];
		tsegment_sizes	= new uint[num_nodes];
		rsegment_sizes	= new uint[num_nodes];

		checkCuda(cudaMalloc((void**)&tri_idx, sizeof(int) * num_tris));
		checkCuda(cudaMalloc((void**)&ray_idx, sizeof(int) * num_rays));
		// copy the contents
		memcpy(node_aabbs, L.node_aabbs, sizeof(AABB) * num_nodes);
		memcpy(tpivots, L.tpivots, sizeof(uint2) * num_nodes);
		memcpy(rpivots, L.rpivots, sizeof(uint2) * num_nodes);
		memcpy(tsegment_sizes, L.tsegment_sizes, sizeof(uint) * num_nodes);
		memcpy(rsegment_sizes, L.rsegment_sizes, sizeof(uint) * num_nodes);
		checkCuda(cudaMemcpy(tri_idx, L.tri_idx, sizeof(int) * num_tris, cudaMemcpyDeviceToDevice));
		checkCuda(cudaMemcpy(ray_idx, L.ray_idx, sizeof(int) * num_rays, cudaMemcpyDeviceToDevice));
	}
	Level& operator= (const Level& L) {
		if(this != &L) {
			// clean up existing pointers and then invoke copy constructor to do the job
			depth		= 0; 
			num_nodes	= 0;
			num_tris	= 0;
			num_rays	= 0;
			SAFE_RELEASE(node_aabbs);
			SAFE_RELEASE(tpivots);
			SAFE_RELEASE(rpivots);
			SAFE_RELEASE(tsegment_sizes);
			SAFE_RELEASE(rsegment_sizes);
			CUDA_SAFE_RELEASE(tri_idx);
			CUDA_SAFE_RELEASE(ray_idx);

			// now copy everything
			depth			= L.depth;
			num_nodes		= L.num_nodes;
			num_tris		= L.num_tris;
			num_rays		= L.num_rays;
			node_aabbs		= new AABB[num_nodes];
			tpivots			= new uint2[num_nodes];
			rpivots			= new uint2[num_nodes];
			tsegment_sizes	= new uint[num_nodes];
			rsegment_sizes	= new uint[num_nodes];
			checkCuda(cudaMalloc((void**)&tri_idx, sizeof(int) * num_tris));
			checkCuda(cudaMalloc((void**)&ray_idx, sizeof(int) * num_rays));

			memcpy(node_aabbs, L.node_aabbs, sizeof(AABB) * num_nodes);
			memcpy(tpivots, L.tpivots, sizeof(uint2) * num_nodes);
			memcpy(rpivots, L.rpivots, sizeof(uint2) * num_nodes);
			memcpy(tsegment_sizes, L.tsegment_sizes, sizeof(uint) * num_nodes);
			memcpy(rsegment_sizes, L.rsegment_sizes, sizeof(uint) * num_nodes);
			checkCuda(cudaMemcpy(tri_idx, L.tri_idx, sizeof(int) * num_tris, cudaMemcpyDeviceToDevice));
			checkCuda(cudaMemcpy(ray_idx, L.ray_idx, sizeof(int) * num_rays, cudaMemcpyDeviceToDevice));
		}
		return *this;
	}
	~Level() {
		SAFE_RELEASE(node_aabbs);
		SAFE_RELEASE(tpivots);
		SAFE_RELEASE(rpivots);
		SAFE_RELEASE(tsegment_sizes);
		SAFE_RELEASE(rsegment_sizes);
		CUDA_SAFE_RELEASE(tri_idx);
		CUDA_SAFE_RELEASE(ray_idx);
	}

	// this function allocates memory for all data based on values set
	void allocateMemory(int _num_nodes, int _num_tris, int _num_rays) {
		num_nodes = _num_nodes; 
		num_tris  = _num_tris;
		num_rays  = _num_rays;
		node_aabbs = new AABB[num_nodes];
		tpivots    = new uint2[num_nodes];
		rpivots    = new uint2[num_nodes];
		tsegment_sizes = new uint[num_nodes];
		rsegment_sizes = new uint[num_nodes];
		checkCuda(cudaMalloc((void**)&tri_idx, sizeof(int) * num_tris));
		checkCuda(cudaMalloc((void**)&ray_idx, sizeof(int) * num_rays));
	}
};

bool calculateChildTriangles(uint* d_temp_tri_keys, uint* d_temp_tri_values, int* d_tsegment_filter_status, uint2* tsegment_parent_pivots, uint* tsegment_parent_sizes, uint2* tsegment_child_pivots, uint* tsegment_child_sizes, size_t num_segments) {

	//first we do a memcpy to two temporary vectors
	bool err_status				= true;
	uint* temp_keys				= new uint[num_segments * 3];
	uint* temp_values			= new uint[num_segments * 3];
	int* tsegment_filter_status = new int[num_segments * 3];
	checkCuda(cudaMemcpy(temp_keys, d_temp_tri_keys, sizeof(uint) * num_segments * 3, cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(temp_values, d_temp_tri_values, sizeof(uint) * num_segments * 3, cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(tsegment_filter_status, d_tsegment_filter_status, sizeof(int) * num_segments * 3, cudaMemcpyDeviceToHost));
	
	int index = 0;
	int status_index = 0;
	for(size_t i = 0; i < num_segments; i++) {
		bool bleft = false, bright = false, bboth = false;
		unsigned int  tleft = 0, tright = 0, tboth = 0;

		if(temp_keys[status_index] == ((i+1)<<2)) {
			bleft = true; status_index++;
		}
		if(temp_keys[status_index] == (((i+1)<<2) + 1)) {
			bboth = true; status_index++;
		}
		if(temp_keys[status_index] == (((i+1)<<2) + 2)) {
			bright = true; status_index++;
		}

		unsigned int num_types = static_cast<unsigned int>(bleft) + static_cast<unsigned int>(bright) + static_cast<unsigned int>(bboth);
		if(num_types == 3) {
			tleft  = temp_values[index++]/((i+1)<<2);
			tboth  = temp_values[index++]/(((i+1)<<2) + 1);
			tright = temp_values[index++]/(((i+1)<<2) + 2);
		} else if(num_types == 2) {
			if(bleft) {
				tleft = temp_values[index++]/((i+1)<<2);
				if(bboth) {
					tboth = temp_values[index++]/(((i+1)<<2) + 1);
				} else if(bright) {
					tright = temp_values[index++]/(((i+1)<<2) + 2);
				}
			} else if(bboth) {
				tboth = temp_values[index++]/(((i+1)<<2) + 1);
				tright = temp_values[index++]/(((i+1)<<2) + 2);
			}
		} else if(num_types == 1) {
			if(bleft)		tleft  = temp_values[index++]/((i+1)<<2);
			else if(bboth)	tboth  = temp_values[index++]/(((i+1)<<2) + 1);
			else if(bright)	tright = temp_values[index++]/(((i+1)<<2) + 2);
		} else {
			std::cerr<<"\nSEVERE ERROR : Somewhere triangle filtering produced anamalous results\nExiting!\n";
			// free up space. jump outside.!
			err_status = false;
			break;
			
		}					 

		// add a debug check for number of triangles that have been filtered.! To be sure.
		unsigned int next_left_cnt  = tleft  + tboth;
		unsigned int next_right_cnt = tright + tboth;
		assert(tleft + tboth + tright == tsegment_parent_sizes[i]);

		uint2 lpivots, rpivots;
		lpivots = make_uint2(tsegment_parent_pivots[i].x, tsegment_parent_pivots[i].x + tleft + tboth);			// the counts themselves yield correct values
		rpivots = make_uint2(tsegment_parent_pivots[i].x + tleft, tsegment_parent_pivots[i].y);
		
		tsegment_child_pivots[2*i]		= lpivots;
		tsegment_child_pivots[2*i + 1]  = rpivots;
		tsegment_child_sizes[2*i]		= next_left_cnt;
		tsegment_child_sizes[2*i+1]		= next_right_cnt;
	}

	SAFE_RELEASE(temp_keys);
	SAFE_RELEASE(temp_values);
	SAFE_RELEASE(tsegment_filter_status);
	return err_status;
}
// num_segments represents the number of segments we are considering
bool calculateChildRays(uint* d_temp_ray_keys, uint* d_temp_ray_values, int* d_rsegment_filter_status, uint2* rsegment_parent_pivots, uint* rsegment_parent_sizes, uint2* rsegment_child_pivots, uint* rsegment_child_sizes, size_t num_segments) {

	bool err_status				 = true;
	uint* temp_keys				 = new uint[num_segments * 4];
	uint* temp_values			 = new uint[num_segments * 4];
	int* rsegment_filter_status  = new int[num_segments * 4];
	checkCuda(cudaMemcpy(temp_keys, d_temp_ray_keys, sizeof(uint) * num_segments * 4, cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(temp_values, d_temp_ray_values, sizeof(uint) * num_segments * 4, cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(rsegment_filter_status, d_rsegment_filter_status, sizeof(int) * num_segments * 4, cudaMemcpyDeviceToHost));
	int index = 0;
	int status_index = 0;
	for(size_t i = 0; i < num_segments; i++) {
		bool bleft = false, bright = false, bboth = false, bnone = false;
		unsigned int rleft = 0, rright = 0, rboth = 0, rnone = 0;
		if(temp_keys[status_index] == ((i+1)<<2)) {
			bleft = true; status_index++;
		}
		if(temp_keys[status_index] == (((i+1)<<2) + 1)) {
			bboth = true; status_index++;
		}
		if(temp_keys[status_index] == (((i+1)<<2) + 2)) {
			bright = true; status_index++;
		}
		if(temp_keys[status_index] == (((i+1)<<2) + 3)) {
			bnone = true; status_index++;
		}

		unsigned int num_types = static_cast<unsigned int>(bleft) + static_cast<unsigned int>(bright) + static_cast<unsigned int>(bboth) + static_cast<unsigned int>(bnone);
		// same logic from celled data.! for now serial. then we'll go for a fully parallel implementation
		if(num_types == 4) {
			rleft  = temp_values[index++]/((i+1)<<2);
			rboth  = temp_values[index++]/(((i+1)<<2) + 1);
			rright = temp_values[index++]/(((i+1)<<2) + 2);
			rnone  = temp_values[index++]/(((i+1)<<2) + 3);
		} else if(num_types == 3) {
			if(bleft) {
				rleft = temp_values[index++]/((i+1)<<2);
				if(bboth) {
					rboth = temp_values[index++]/(((i+1)<<2) + 1);
					if(bright) {
						rright = temp_values[index++]/(((i+1)<<2) + 2);
					} else if(bnone) {
						rnone = temp_values[index++]/(((i+1)<<2) + 3);
					}
				} else if(bright) {
					rright = temp_values[index++]/(((i+1)<<2) + 2);
					rnone = temp_values[index++]/(((i+1)<<2) + 3);
				}

			} else if(bboth) {
				rboth  = temp_values[index++]/(((i+1)<<2) + 1);
				rright = temp_values[index++]/(((i+1)<<2) + 2);
				rnone  = temp_values[index++]/(((i+1)<<2) + 3);
			}
		} else if(num_types == 2) {
			if(bleft) {
				rleft = temp_values[index++]/((i+1)<<2);
				if(bboth) {
					rboth  = temp_values[index++]/(((i+1)<<2) + 1);
				} else if(bright) {
					rright = temp_values[index++]/(((i+1)<<2) + 2);
				} else if(bnone) {
					rnone  = temp_values[index++]/(((i+1)<<2) + 3);
				}
			} else if(bboth) {
				rboth = temp_values[index++]/(((i+1)<<2) + 1);
				if(bright) {
					rright = temp_values[index++]/(((i+1)<<2) + 2);
				} else if(bnone) {
					rnone  = temp_values[index++]/(((i+1)<<2) + 3);
				}
			} else if(bright) {
				rright = temp_values[index++]/(((i+1)<<2) + 2);
				rnone  = temp_values[index++]/(((i+1)<<2) + 3);
			}
		} else if(num_types == 1) {
			if(bleft)			rleft  = temp_values[index++]/((i+1)<<2);
			else if(bboth)		rboth  = temp_values[index++]/(((i+1)<<2) + 1);
			else if(bright)	rright = temp_values[index++]/(((i+1)<<2) + 2);
			else if(bnone)		rnone  = temp_values[index++]/(((i+1)<<2) + 3);
		} else {
			std::cerr<<"SEVERE ERROR. No keys are present for ray checking @ iteration : "<<i<<"\nExiting!";
			err_status = false;
			break;		// jump and clear 
		}

		// add a debug check
		assert(rleft + rright + rboth + rnone == rsegment_parent_sizes[i]);
		unsigned int next_left_cnt  = rleft + rboth;
		unsigned int next_right_cnt = rright + rboth;
		uint2 lpivots = make_uint2(rsegment_parent_pivots[i].x, rsegment_parent_pivots[i].x + rleft + rboth); 
		uint2 rpivots = make_uint2(rsegment_parent_pivots[i].x + rleft, rsegment_parent_pivots[i].y - rnone);

		rsegment_child_pivots[2*i]	   = lpivots;
		rsegment_child_pivots[2*i + 1] = rpivots;
		rsegment_child_sizes[2*i]	   = next_left_cnt;
		rsegment_child_sizes[2*i + 1]  = next_right_cnt;
	}
	SAFE_RELEASE(temp_keys);
	SAFE_RELEASE(temp_values);
	SAFE_RELEASE(rsegment_filter_status);
	return err_status;
}

uint4 markBruteForceModified(uint2* tsegment_child_pivots, uint* tsegment_child_sizes, uint2* rsegment_child_pivots, 
	uint* rsegment_child_sizes, uint* segment_flags, size_t num_parent_segments, size_t num_child_segments, size_t& num_bf_segments, DacrtRunTimeParameters& rtparams, 
	Logger& logger, uint depth, KdTree& tree, AABB* h_child_aabbs) {

		// assuming enough memory for the input params are allocated in the caller
		size_t next_level_tri_sizes		= 0;
		size_t next_level_ray_sizes		= 0;
		size_t next_level_bf_tri_sizes	= 0;
		size_t next_level_bf_ray_sizes	= 0;
		size_t bf_segment_cnt			= 0;

		// we insert the brute force elements into the 
		for(size_t i = 0; i < num_child_segments; i++) {
			// first mark the appropriate stuff for the kdtree
			tree.node_aabbs.push_back(h_child_aabbs[i]);
			logger.write<uint>("Node tri sizes : ", tsegment_child_sizes[i]);
			logger.write<uint>("Node ray sizes : ", rsegment_child_sizes[i]);
			if(tsegment_child_sizes[i] < rtparams.PARALLEL_TRI_THRESHOLD || rsegment_child_sizes[i] < rtparams.PARALLEL_RAY_THRESHOLD) {
				segment_flags[i] = 0;
				next_level_bf_tri_sizes += tsegment_child_sizes[i];
				next_level_bf_ray_sizes += rsegment_child_sizes[i];
				bf_segment_cnt++;
				tree.leaf.push_back(true);
			} else {
				next_level_tri_sizes += tsegment_child_sizes[i];
				next_level_ray_sizes += rsegment_child_sizes[i];
				segment_flags[i] = 1;
				tree.leaf.push_back(false);
			}
		}
		num_bf_segments = bf_segment_cnt;
		return make_uint4(next_level_tri_sizes, next_level_ray_sizes, next_level_bf_tri_sizes, next_level_bf_ray_sizes);
}

// scatters data appropriately
void scatterDataModified(const Level& parent, uint2* tsegment_child_pivots, uint2* rsegment_child_pivots, uint* tsegment_child_sizes, uint* rsegment_child_sizes, 
	int* next_level_tri_idx, int* next_level_ray_idx, ParallelPackModified& pack, DacrtRunTimeParameters& rtparams, uint* segment_flags, size_t num_child_segments,
	TriangleArray& d_triangles, RayArray& d_rays, Counters& ctr, KdTree& tree) {

		size_t tnum_copied = 0;
		size_t rnum_copied = 0;
		uint   bf_index	   = 0;
		for(size_t i = 0; i < num_child_segments; i++) {
			if(segment_flags[i] == 0) {
				// brute force
				// insert into leaf nodes
				if(tsegment_child_sizes[i] > 0 || rsegment_child_sizes[i] > 0) {
					
					// pack has all the data in thrust format
					// hence we use thrust copy
					thrust::copy(thrust::device_ptr<int>(parent.tri_idx) + tsegment_child_pivots[i].x, thrust::device_ptr<int>(parent.tri_idx) + tsegment_child_pivots[i].y, pack.buffered_tri_idx.begin() + pack.tri_buffer_occupied);
					thrust::copy(thrust::device_ptr<int>(parent.ray_idx) + rsegment_child_pivots[i].x, thrust::device_ptr<int>(parent.ray_idx) + rsegment_child_pivots[i].y, pack.buffered_ray_idx.begin() + pack.ray_buffer_occupied);
					// update the pack's host data for all temporary data that will help in modified approach
					// we update only host data.
					pack.htri_segment_sizes[pack.num_segments] = tsegment_child_sizes[i];
					pack.hray_segment_sizes[pack.num_segments] = rsegment_child_sizes[i];
					pack.hsegment_ids[pack.num_segments] = pack.num_segments;

					int numblocks = (rsegment_child_sizes[i] / rtparams.NUM_RAYS_PER_BLOCK) + (rsegment_child_sizes[i] % rtparams.NUM_RAYS_PER_BLOCK != 0);
					pack.blockCnt += numblocks;
					// store num_segments for blockCnt times in blockNos starting from 
					// reset this value every time I am doing this.
					int tempstart = 0;
					for(int j = pack.bstart; j < pack.blockCnt; j++) {
						pack.blockNos[j]   = pack.num_segments;
						pack.blockStart[j] = tempstart++;
					}
				
					pack.bstart += numblocks;
					pack.num_segments++;		// increment the count
					pack.ray_buffer_occupied += rsegment_child_sizes[i];
					pack.tri_buffer_occupied += tsegment_child_sizes[i];
					// update the bf status index
					bf_index++;
					
				}
			} else {
				// internal node
				thrust::copy(thrust::device_ptr<int>(parent.tri_idx) + tsegment_child_pivots[i].x, thrust::device_ptr<int>(parent.tri_idx) + tsegment_child_pivots[i].y, thrust::device_ptr<int>(next_level_tri_idx) + tnum_copied);
				thrust::copy(thrust::device_ptr<int>(parent.ray_idx) + rsegment_child_pivots[i].x, thrust::device_ptr<int>(parent.ray_idx) + rsegment_child_pivots[i].y, thrust::device_ptr<int>(next_level_ray_idx) + rnum_copied);
				tnum_copied += tsegment_child_sizes[i];
				rnum_copied += rsegment_child_sizes[i];
			}
		}
}

// removes the brute force nodes from the level list
// caller should allocate memory for the out list appropriately. 
/// NOTE: use the value num_bf_segments from mark method.! to get correct count.
void trimNodesModified(uint2* tsegment_child_pivots, uint* tsegment_child_sizes, uint2* rsegment_child_pivots, uint* rsegment_child_sizes, 
	uint2* tsegment_child_pivots_out, uint* tsegment_child_sizes_out, uint2* rsegment_child_pivots_out, uint* rsegment_child_sizes_out,
	uint* segment_flags, size_t num_child_segments) {

		int idx = 0;
		for(size_t i = 0; i < num_child_segments; i++) {
			if(segment_flags[i] == 1) {
				tsegment_child_pivots_out[idx] = tsegment_child_pivots[i];
				rsegment_child_pivots_out[idx] = rsegment_child_pivots[i];
				tsegment_child_sizes_out[idx]  = tsegment_child_sizes[i];
				rsegment_child_sizes_out[idx]  = rsegment_child_sizes[i];
				idx++;
			}
		}
}

// caller is assumed to have correctly allocated memory for all the child boxes
void computeNextLevelBoxes(AABB* parent_level_boxes, AABB* child_boxes, uint* segment_flags, size_t num_parent_segments, size_t num_child_segments) {

	uint index = 0;
	for(size_t i = 0; i < num_parent_segments; i++) {
		AABB left, right;
		float split_pos;
		splitSpatialMedian(parent_level_boxes[i], left, right, split_pos);
		if(segment_flags[2*i] == 1) child_boxes[index++] = left;
		if(segment_flags[2*i + 1] == 1) child_boxes[index++] = right;
	}
}

void reindexPivotsModified(uint2* tsegment_child_pivots, uint* tsegment_child_sizes, uint2* rsegment_child_pivots, uint* rsegment_child_sizes,
	size_t num_next_level_segments) {
		size_t tri = 0;
		size_t ray = 0;
		for(size_t i = 0; i < num_next_level_segments; i++) {
			tsegment_child_pivots[i] = make_uint2(tri, tri + tsegment_child_sizes[i]);
			rsegment_child_pivots[i] = make_uint2(ray, ray + rsegment_child_sizes[i]);
			tri += tsegment_child_sizes[i];
			ray += rsegment_child_sizes[i];
		}
}

extern "C"
void completeBruteForceModified(ParallelPackModified& pack, TriangleArray& d_triangles, RayArray& d_rays, DacrtRunTimeParameters& rtparams, Counters& ctr) {

	// first copy all the host data to device data (Note: I guess this would be not advisable. I'll copy directly into the device memory itself.
#ifdef _DEBUG
	printf("brute force called\n");
#endif
	pack.dblockNos.resize(pack.blockCnt);
	pack.dblockStart.resize(pack.blockCnt);
	Timer memcpytimer1("memcpy timer 1");
	memcpytimer1.start();
	thrust::copy(pack.htri_segment_sizes.begin(), pack.htri_segment_sizes.begin() + pack.num_segments, pack.tri_segment_sizes.begin());
	thrust::copy(pack.hray_segment_sizes.begin(), pack.hray_segment_sizes.begin() + pack.num_segments, pack.ray_segment_sizes.begin());
	thrust::copy(pack.hsegment_ids.begin(), pack.hsegment_ids.begin() + pack.num_segments, pack.segment_ids.begin());
	thrust::copy(pack.blockNos.begin(), pack.blockNos.begin() + pack.blockCnt, pack.dblockNos.begin());
	thrust::copy(pack.blockStart.begin(), pack.blockStart.begin() + pack.blockCnt, pack.dblockStart.begin());
	memcpytimer1.stop();
	ctr.mem_cpy_time += memcpytimer1.get_ms();

	// we perform the dacrt setup and completion routines here
	thrust::device_vector<int> ray_segment_start(pack.num_segments);
	thrust::device_vector<int> tri_segment_start(pack.num_segments);
	Timer misctimer1("Misc timer 1");
	misctimer1.start();
	thrust::exclusive_scan(pack.tri_segment_sizes.begin(), pack.tri_segment_sizes.begin() + pack.num_segments, tri_segment_start.begin());
	thrust::exclusive_scan(pack.ray_segment_sizes.begin(), pack.ray_segment_sizes.begin() + pack.num_segments, ray_segment_start.begin());
	misctimer1.stop();
	ctr.misc_time += misctimer1.get_ms();
	
	// calculate kernel sizes
	int num_blocks = pack.blockCnt;
	int num_threads_per_block = rtparams.NUM_RAYS_PER_BLOCK;
	// call kernel
	Timer seg_brute_timer("SegmentedBruteForce Timer");
	seg_brute_timer.start();
				
	modifiedSegmentedBruteForce<<<num_blocks, num_threads_per_block>>>(d_rays, d_triangles, thrust::raw_pointer_cast(&pack.buffered_ray_idx[0]), pack.ray_buffer_occupied,
		thrust::raw_pointer_cast(&pack.buffered_tri_idx[0]), pack.tri_buffer_occupied, thrust::raw_pointer_cast(&pack.ray_segment_sizes[0]),
		thrust::raw_pointer_cast(&pack.tri_segment_sizes[0]), thrust::raw_pointer_cast(&ray_segment_start[0]), thrust::raw_pointer_cast(&tri_segment_start[0]),
		thrust::raw_pointer_cast(&pack.dblockNos[0]), thrust::raw_pointer_cast(&pack.dblockStart[0]), thrust::raw_pointer_cast(&pack.buffered_ray_maxts[0]),
		thrust::raw_pointer_cast(&pack.buffered_ray_hitids[0]));

	seg_brute_timer.stop();
	ctr.brute_force_time += seg_brute_timer.get_ms();
	
	// complete the sort phase
	Timer seg_sort_timer("Seg Sorted Timer");
	seg_sort_timer.start();
	thrust::sort_by_key(pack.buffered_ray_idx.begin(), pack.buffered_ray_idx.begin() + pack.ray_buffer_occupied,
		thrust::make_zip_iterator(thrust::make_tuple(pack.buffered_ray_maxts.begin(), pack.buffered_ray_hitids.begin())));
	seg_sort_timer.stop();
	ctr.seg_sort_time += seg_sort_timer.get_ms();
	
	// complete the reduction phase
	thrust::device_vector<int> ray_idx(rtparams.BUFFER_SIZE);
	thrust::device_vector<float> ray_maxts(rtparams.BUFFER_SIZE);
	thrust::device_vector<int> ray_hitids(rtparams.BUFFER_SIZE);
	thrust::equal_to<int> pred;
				
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
	
	// update the global min index
	int num_valid_keys = minend.first - ray_idx.begin();
	num_threads_per_block = 512;
	num_blocks = num_valid_keys / num_threads_per_block + (num_valid_keys % num_threads_per_block != 0);
				
	Timer update_min_timer("Update Min Timer");
	update_min_timer.start();
	updateMinKernel<<<num_blocks, num_threads_per_block>>>(thrust::raw_pointer_cast(&ray_idx[0]), thrust::raw_pointer_cast(&ray_maxts[0]), thrust::raw_pointer_cast(&ray_hitids[0]),
		thrust::raw_pointer_cast(&pack.dev_ray_maxts[0]), thrust::raw_pointer_cast(&pack.dev_hitids[0]), num_valid_keys);
	update_min_timer.stop();
	ctr.update_min_time += update_min_timer.get_ms();

	// reset the block data
	pack.clearBlockData();

	// clear other temp data within function
	ray_idx.clear();
	ray_maxts.clear();
	ray_hitids.clear();
	
	// clear all device data in pack
	pack.clearDeviceData();

	// reset the host data/counters also
	// when the function returns, either we use a double buffered approach/ or single buffer memcpy continues.
	pack.clearHostData();
}

void gpuDacrtFullyParallelModified(
	AABB& root, 
	TriangleArray& d_triangles, int* tri_idx_array, int num_triangles, int tpivot,			// added a separate term for num_triangles as well as pivot. This is to facilitate some kind of LOD/back face culling mechanism
	RayArray& d_rays, int* ray_idx_array, int num_rays, int rpivot, 
	float* d_maxts, int* d_hitids, 
	DacrtRunTimeParameters& rtparams, Counters& ctr, Logger& logger) {

		// create a root node
		Level* root_level = new Level();
		root_level->depth = 0;
		root_level->allocateMemory(1, tpivot, rpivot);
		// initialize memory
		*(root_level->node_aabbs) = root;
		*(root_level->tpivots) = make_uint2(0, tpivot);
		*(root_level->rpivots) = make_uint2(0, rpivot);
		*(root_level->tsegment_sizes) = tpivot;
		*(root_level->rsegment_sizes) = rpivot;
		thrust::copy(thrust::device_ptr<int>(tri_idx_array), thrust::device_ptr<int>(tri_idx_array) + tpivot, thrust::device_ptr<int>(root_level->tri_idx));
		thrust::copy(thrust::device_ptr<int>(ray_idx_array), thrust::device_ptr<int>(ray_idx_array) + rpivot, thrust::device_ptr<int>(root_level->ray_idx));

		std::stack<Level*> workingStack;
		workingStack.push(root_level);

		int depth = 0;
		ParallelPackModified *pack = new ParallelPackModified(rtparams, num_rays);
		
		// create a kdtree
		KdTree tree(root, d_triangles);
		tree.node_aabbs.push_back(root);
		

		// for debug purposes, we push and see how much memory we actually allocate using new and delete commands
		std::vector<unsigned int> host_memory;
		std::vector<unsigned int> device_memory;
		double dstart, dend;
		/// Major steps of the algorithm will be marked with # tags.!
		while(workingStack.top()->num_nodes != 0) {
			
			
			size_t per_iteration_host_memory = 0;
			size_t per_iteration_device_memory = 0;
			
			Level* workNode = workingStack.top();
			workingStack.pop();

#ifdef _DEBUG
			printf("Depth Reached : %d\n", depth);
#endif

			// initialize the keys with respect to all the parent aabbs
			size_t num_segments = workNode->num_nodes;
			size_t trikeylen = 0;
			size_t raykeylen = 0;
#ifdef _DEBUG
			logger.write<uint>("Depth reached : ", depth);
			logger.write<uint>("Total Tris    : ", workNode->num_tris);
			logger.write<uint>("Total Rays    : ", workNode->num_rays);
			logger.write<uint>("Num Nodes  : ", workNode->num_nodes);
#endif

			unsigned int tsegment_no = 0;
			unsigned int rsegment_no = 0;

			unsigned int* tkeyblocks		= NULL;
			unsigned int* tkeyblockStart	= NULL;
			unsigned int* rkeyblocks		= NULL;
			unsigned int* rkeyblockStart	= NULL;
			unsigned int* tsegmentSizes		= new unsigned int[num_segments];
			unsigned int* rsegmentSizes		= new unsigned int[num_segments];
			unsigned int* toffsets			= new unsigned int[num_segments];								
			unsigned int* roffsets			= new unsigned int[num_segments];

			// update memory
			per_iteration_host_memory += 4 * num_segments * sizeof(unsigned int);

			// #1. Calculate required sizes for all the keys that have to be tested for data 
			//     first we have to calculate sizes required for all temporary data we need for key blocks and keyblock starts
			//     NOTE: calculation can be done entirely in device.
			size_t tblocks  = 0;
			size_t rblocks	= 0;
			dstart = omp_get_wtime();
			for(size_t i = 0; i < num_segments; i++) {
				uint tsize			= workNode->tsegment_sizes[i];
				size_t tnum_blocks	= (tsize / 256) + (tsize % 256 != 0);
				tblocks				+= tnum_blocks;
				// ray data
				uint rsize			=	workNode->rsegment_sizes[i];
				size_t rnum_blocks	= (rsize / 256) + (rsize % 256 != 0);
				rblocks				+= rnum_blocks;
			}
			
			//logger.write<uint>("tblocks : ", tblocks);
			//logger.write<uint>("rblocks : ", rblocks);

			// allocate so and so memory now
			tkeyblocks		= new unsigned int[tblocks];
			tkeyblockStart	= new unsigned int[tblocks];
			rkeyblocks		= new unsigned int[rblocks];
			rkeyblockStart	= new unsigned int[rblocks];

			per_iteration_host_memory += 2 * tblocks * sizeof(unsigned int);
			per_iteration_host_memory += 2 * rblocks * sizeof(unsigned int);

			// #2. Fill data for the blocks
			unsigned int tindex = 0;
			unsigned int rindex = 0;
			for(size_t i = 0; i < num_segments; i++) {
				toffsets[i] = trikeylen;
				roffsets[i] = raykeylen;

				size_t tsize		= workNode->tsegment_sizes[i];
				size_t num_blocks	= (tsize / 256) + (tsize % 256 != 0);
				size_t tblockStart	= 0;
				for(size_t j = 0; j < num_blocks; j++) {
					tkeyblocks[tindex]		= tsegment_no;
					tkeyblockStart[tindex]	= tblockStart++;
					tindex++;
				}

				size_t rsize		= workNode->rsegment_sizes[i];
				num_blocks			= (rsize / 256) + (rsize % 256 != 0);
				size_t rblockStart	= 0;
				for(size_t j = 0; j < num_blocks; j++) {
					rkeyblocks[rindex]		= rsegment_no;
					rkeyblockStart[rindex]	= rblockStart++;
					rindex++;
				}

				tsegmentSizes[i] = tsize;
				rsegmentSizes[i] = rsize;

				tsegment_no++;
				rsegment_no++;

				trikeylen += tsize;
				raykeylen += rsize;
			}
			
			// now allocate device memory
			unsigned int* d_tkeyblocks		= NULL;
			unsigned int* d_rkeyblocks		= NULL;
			unsigned int* d_tkeyblockStart	= NULL;
			unsigned int* d_rkeyblockStart	= NULL;
			unsigned int* d_tsegmentSizes	= NULL;
			unsigned int* d_rsegmentSizes	= NULL;
			unsigned int* d_trioffsets		= NULL;
			unsigned int* d_rayoffsets		= NULL;
			
			checkCuda(cudaMalloc((void**)&d_tkeyblocks, sizeof(unsigned int) * tblocks));
			checkCuda(cudaMalloc((void**)&d_rkeyblocks, sizeof(unsigned int) * rblocks));
			checkCuda(cudaMalloc((void**)&d_tkeyblockStart, sizeof(unsigned int) * tblocks));
			checkCuda(cudaMalloc((void**)&d_rkeyblockStart, sizeof(unsigned int) * rblocks));
			checkCuda(cudaMalloc((void**)&d_tsegmentSizes, sizeof(unsigned int) * num_segments));
			checkCuda(cudaMalloc((void**)&d_rsegmentSizes, sizeof(unsigned int) * num_segments));
			checkCuda(cudaMalloc((void**)&d_trioffsets, sizeof(unsigned int) * num_segments));
			checkCuda(cudaMalloc((void**)&d_rayoffsets, sizeof(unsigned int) * num_segments));
			dend = omp_get_wtime();
			ctr.misc_time += ((dend-dstart) * 1000.0f);
			
			// copy memory contents
			per_iteration_device_memory = per_iteration_host_memory;			// absolutely equal till now.
			Timer memcpytimer1("memcpytimer1");
			memcpytimer1.start();
			checkCuda(cudaMemcpy(d_tkeyblocks, tkeyblocks, sizeof(unsigned int) * tblocks, cudaMemcpyHostToDevice));
			checkCuda(cudaMemcpy(d_rkeyblocks, rkeyblocks, sizeof(unsigned int) * rblocks, cudaMemcpyHostToDevice));
			checkCuda(cudaMemcpy(d_tkeyblockStart, tkeyblockStart, sizeof(unsigned int) * tblocks, cudaMemcpyHostToDevice));
			checkCuda(cudaMemcpy(d_rkeyblockStart, rkeyblockStart, sizeof(unsigned int) * rblocks, cudaMemcpyHostToDevice));
			checkCuda(cudaMemcpy(d_tsegmentSizes, tsegmentSizes, sizeof(unsigned int) * num_segments, cudaMemcpyHostToDevice));
			checkCuda(cudaMemcpy(d_rsegmentSizes, rsegmentSizes, sizeof(unsigned int) * num_segments, cudaMemcpyHostToDevice));
			checkCuda(cudaMemcpy(d_trioffsets, toffsets, sizeof(unsigned int) * num_segments, cudaMemcpyHostToDevice));
			checkCuda(cudaMemcpy(d_rayoffsets, roffsets, sizeof(unsigned int) * num_segments, cudaMemcpyHostToDevice));
			memcpytimer1.stop();
			ctr.mem_cpy_time += memcpytimer1.get_ms();

			// #3. Allocate space for keys 
			dstart = omp_get_wtime();
			unsigned int* d_trikeys       = NULL;
			unsigned int* d_raykeys       = NULL;
			int* d_tsegment_filter_status = NULL;
			int* d_rsegment_filter_status = NULL;
			unsigned int* d_split_axis    = NULL;
			float* d_split_pos            = NULL;

			checkCuda(cudaMalloc((void**)&d_split_axis, sizeof(unsigned int) * num_segments));
			checkCuda(cudaMalloc((void**)&d_split_pos, sizeof(float) * num_segments));
			checkCuda(cudaMalloc((void**)&d_trikeys, sizeof(unsigned int) * trikeylen));
			checkCuda(cudaMalloc((void**)&d_raykeys, sizeof(unsigned int) * raykeylen));
			checkCuda(cudaMalloc((void**)&d_tsegment_filter_status, sizeof(int) * num_segments * 3));
			checkCuda(cudaMalloc((void**)&d_rsegment_filter_status, sizeof(int) * num_segments * 4));
			per_iteration_device_memory += sizeof(unsigned int) * (trikeylen + raykeylen) + sizeof(int) * (num_segments * 3 + num_segments * 4);
			// #3.1 Fill the status filters with 0
			cudaMemset(d_tsegment_filter_status, 0, sizeof(int) * num_segments * 3);
			cudaMemset(d_rsegment_filter_status, 0, sizeof(int) * num_segments * 4);

			// #4. Allocate space for aabb in device
			AABB* d_node_aabbs;
			AABB* d_child_node_aabbs;
			AABB* h_child_node_aabbs = new AABB[workNode->num_nodes * 2];
			checkCuda(cudaMalloc((void**)&d_node_aabbs, sizeof(AABB) * workNode->num_nodes));
			checkCuda(cudaMalloc((void**)&d_child_node_aabbs, sizeof(AABB) * workNode->num_nodes * 2));
			checkCuda(cudaMemcpy(d_node_aabbs, workNode->node_aabbs, sizeof(AABB) * workNode->num_nodes, cudaMemcpyHostToDevice));
			per_iteration_device_memory += sizeof(AABB) * workNode->num_nodes;
			dend = omp_get_wtime();
			ctr.misc_time += ((dend - dstart) * 1000.0f);

			// #5. Compute occupancy for triangles and rays
			//uint MAX_BLOCKS_X = 16384;
			//uint NUM_BLOCKS = tblocks;
			uint NUM_BLOCKS_Y = (tblocks / rtparams.GRID_DIM_X) + (tblocks % rtparams.GRID_DIM_X != 0);
			uint NUM_BLOCKS_X = rtparams.GRID_DIM_X;
			
			dim3 t_grid(NUM_BLOCKS_X, NUM_BLOCKS_Y, rtparams.GRID_DIM_Z);
			dim3 t_threads(256, 1, 1);										// this value is constant.!!

			//uint NUM_THREADS_PER_BLOCK = 256;
			Timer trifiltertimer("tri filter timer");
			trifiltertimer.start();
			/*
			cudaEvent_t start, stop;
			cudaStream_t stream1, stream2;
			checkCuda(cudaEventCreate(&start));
			checkCuda(cudaEventCreate(&stop));
			checkCuda(cudaEventRecord(start, 0));
			checkCuda(cudaStreamCreate(&stream1));
			checkCuda(cudaStreamCreate(&stream2));
			*/

			parallelTriFilter<<<t_grid, t_threads>>>(d_triangles.v0, d_triangles.v1, d_triangles.v2, 
												d_node_aabbs, d_child_node_aabbs, workNode->tri_idx, d_trikeys, d_tkeyblocks, 
												d_tkeyblockStart, d_tsegmentSizes, d_trioffsets, 
												d_tsegment_filter_status, 
												d_split_axis, d_split_pos,
												trikeylen, depth, tblocks);

			trifiltertimer.stop();
			ctr.tri_filter_time += trifiltertimer.get_ms();
			checkCuda(cudaMemcpy(h_child_node_aabbs, d_child_node_aabbs, sizeof(AABB) * workNode->num_nodes * 2, cudaMemcpyDeviceToHost));
			

			//NUM_BLOCKS = rblocks;
			NUM_BLOCKS_Y = (rblocks / rtparams.GRID_DIM_X) + (rblocks % rtparams.GRID_DIM_X != 0);
			dim3 r_grid(NUM_BLOCKS_X, NUM_BLOCKS_Y, rtparams.GRID_DIM_Z);
			dim3 r_threads(256, 1, 1);

			Timer rayfiltertimer("ray filter timer");
			rayfiltertimer.start();
			parallelRayFilter<<<r_grid, r_threads>>>(d_rays.o, d_rays.d, d_node_aabbs, workNode->ray_idx, d_raykeys, 
													 d_rkeyblocks, d_rkeyblockStart, d_rsegmentSizes,
													 d_rayoffsets, d_rsegment_filter_status, raykeylen, depth, rblocks);
			rayfiltertimer.stop();
			ctr.ray_filter_time += rayfiltertimer.get_ms();
			/*
			checkCuda(cudaStreamSynchronize(stream1));
			checkCuda(cudaStreamSynchronize(stream2));
			checkCuda(cudaEventRecord(stop, 0));
			checkCuda(cudaEventSynchronize(stop));
			float elapsed_time = 0;
			checkCuda(cudaEventElapsedTime(&elapsed_time, start, stop));
			ctr.tri_filter_time += elapsed_time;
			*/

			/** We do a compress - sort - reduce - decompress strategy 
			*/
			/*
			dstart = omp_get_wtime();
			compressSortDecompress(workNode->tri_idx, d_trikeys, trikeylen);
			dend = omp_get_wtime();
			ctr.trisortbykey_time += ((dend - dstart) * 1000.0f);

			dstart = omp_get_wtime();
			compressSortDecompress(workNode->ray_idx, d_raykeys, raykeylen);
			dend = omp_get_wtime();
			ctr.raysortbykey_time += ((dend - dstart) * 1000.0f);
			*/
			
			// #6. Sort step the keys
			
			Timer trisorttimer("trisorttimer");
			trisorttimer.start();
			thrust::sort_by_key(thrust::device_ptr<unsigned int>(d_trikeys), thrust::device_ptr<unsigned int>(d_trikeys) + trikeylen, thrust::device_ptr<int>(workNode->tri_idx));
			trisorttimer.stop();
			ctr.trisortbykey_time += trisorttimer.get_ms();

			Timer raysorttimer("raysorttimer");
			raysorttimer.start();
			thrust::sort_by_key(thrust::device_ptr<unsigned int>(d_raykeys), thrust::device_ptr<unsigned int>(d_raykeys) + raykeylen, thrust::device_ptr<int>(workNode->ray_idx));
			raysorttimer.stop();
			ctr.raysortbykey_time += raysorttimer.get_ms();
			

			// #7. Do a reduction step
			unsigned int* d_temp_tri_keys	= NULL;
			unsigned int* d_temp_ray_keys	= NULL;
			unsigned int* d_temp_tri_values = NULL;
			unsigned int* d_temp_ray_values = NULL;
			checkCuda(cudaMalloc((void**)&d_temp_tri_keys, sizeof(unsigned int) * num_segments * 3));
			checkCuda(cudaMalloc((void**)&d_temp_ray_keys, sizeof(unsigned int) * num_segments * 4));
			checkCuda(cudaMalloc((void**)&d_temp_tri_values, sizeof(unsigned int) * num_segments * 3));
			checkCuda(cudaMalloc((void**)&d_temp_ray_values, sizeof(unsigned int) * num_segments * 4));
			checkCuda(cudaMemset(d_temp_tri_keys, 0, sizeof(unsigned int) * num_segments * 3));
			checkCuda(cudaMemset(d_temp_ray_keys, 0, sizeof(unsigned int) * num_segments * 4));
			checkCuda(cudaMemset(d_temp_tri_values, 0, sizeof(unsigned int) * num_segments * 3));
			checkCuda(cudaMemset(d_temp_ray_values, 0, sizeof(unsigned int) * num_segments * 4));
			per_iteration_device_memory += 4 * sizeof(unsigned int) * num_segments;

			Timer triredtimer("tri red timer");
			triredtimer.start();
			thrust::reduce_by_key(thrust::device_ptr<unsigned int>(d_trikeys), thrust::device_ptr<unsigned int>(d_trikeys) + trikeylen, 
								  thrust::device_ptr<unsigned int>(d_trikeys), thrust::device_ptr<unsigned int>(d_temp_tri_keys), 
								  thrust::device_ptr<unsigned int>(d_temp_tri_values));
			triredtimer.stop();
			ctr.trireduction_time += triredtimer.get_ms();

			Timer rayredtimer("ray red timer");
			rayredtimer.start();
			thrust::reduce_by_key(thrust::device_ptr<unsigned int>(d_raykeys), thrust::device_ptr<unsigned int>(d_raykeys) + raykeylen,
								  thrust::device_ptr<unsigned int>(d_raykeys), thrust::device_ptr<unsigned int>(d_temp_ray_keys),
								  thrust::device_ptr<unsigned int>(d_temp_ray_values));
			rayredtimer.stop();
			ctr.rayreduction_time += rayredtimer.get_ms();

			// #8. Estimate child count values for all the nodes
			uint2* tsegment_child_pivots = new uint2[num_segments * 2];
			uint2* rsegment_child_pivots = new uint2[num_segments * 2];
			uint*  tsegment_child_sizes  = new uint[num_segments * 2];
			uint*  rsegment_child_sizes  = new uint[num_segments * 2];
			per_iteration_host_memory += 4 * (sizeof(uint2) * num_segments * 2);

			dstart = omp_get_wtime();
			calculateChildTriangles(d_temp_tri_keys, d_temp_tri_values, d_tsegment_filter_status, workNode->tpivots, workNode->tsegment_sizes, 
				tsegment_child_pivots, tsegment_child_sizes, num_segments);
			calculateChildRays(d_temp_ray_keys, d_temp_ray_values, d_rsegment_filter_status, workNode->rpivots, workNode->rsegment_sizes,
				rsegment_child_pivots, rsegment_child_sizes, num_segments);
			
			// #9. Mark Brute force
			size_t num_bf_segments		= 0;
			size_t num_child_segments	= 2 * num_segments;
			uint* segment_flags			= new uint[num_child_segments];			// each current has 2 potential child segments
			
			uint4 next_level_sizes = markBruteForceModified(tsegment_child_pivots, tsegment_child_sizes, rsegment_child_pivots,
				rsegment_child_sizes, segment_flags, num_segments, num_child_segments, num_bf_segments, rtparams, logger, depth, tree, h_child_node_aabbs);
			
			dend = omp_get_wtime();
			ctr.misc_time += ((dend-dstart) * 1000.0f);

#ifdef _DEBUG
			logger.write<uint>("Next level tris : ", next_level_sizes.x);
			logger.write<uint>("Next level rays : ", next_level_sizes.y);
			logger.write<uint>("Next bf tris    : ", next_level_sizes.z);
			logger.write<uint>("Next bf rays    : ", next_level_sizes.w);
#endif
			// #10. If we dont have space for the segmented brute force, complete the brute force now of all active segments
			
			
			if((next_level_sizes.z + pack->tri_buffer_occupied) > rtparams.BUFFER_SIZE || (next_level_sizes.w + pack->ray_buffer_occupied) > rtparams.BUFFER_SIZE || 
				(pack->num_segments + num_bf_segments) > rtparams.MAX_SEGMENTS) {
					double dstart = omp_get_wtime();
#ifdef _DEBUG
					logger.write<uint>("Brute Force Triangles : ", pack->tri_buffer_occupied);
					logger.write<uint>("Brute Force Rays      : ", pack->ray_buffer_occupied);
					logger.write<uint>("Brute Force Num Segments : ", pack->num_segments);
#endif
					completeBruteForceModified(*pack, d_triangles, d_rays, rtparams, ctr);
					double dend = omp_get_wtime();
#ifdef _DEBUG
					logger.write<double>("Brute force modified completed in : ", dend-dstart);
#endif
			}
				
			
			// #11. Allocate enough memory for next level nodes
			int* d_next_level_tri_idx = NULL;
			int* d_next_level_ray_idx = NULL;
			checkCuda(cudaMalloc((void**)&d_next_level_tri_idx, sizeof(int) * next_level_sizes.x));
			checkCuda(cudaMalloc((void**)&d_next_level_ray_idx, sizeof(int) * next_level_sizes.y));
			per_iteration_device_memory += sizeof(int) * (next_level_sizes.x + next_level_sizes.y);

			// #12. scatter data
			dstart = omp_get_wtime();
			scatterDataModified(*workNode, tsegment_child_pivots, rsegment_child_pivots, tsegment_child_sizes, rsegment_child_sizes, d_next_level_tri_idx,
				d_next_level_ray_idx, *pack, rtparams, segment_flags, num_child_segments, d_triangles, d_rays, ctr, tree);
			dend = omp_get_wtime();
			ctr.mem_cpy_time += ((dend-dstart) * 1000.0f);

			// #13. Recompute next level bounding boxes
			// NOTE: we have to take care to determine the size of the number of boxes for the next level
			//		 next_level_boxes = num_child_segments - bf_segments
			dstart = omp_get_wtime();
			size_t num_next_level = num_child_segments - num_bf_segments;
			AABB* next_level_boxes = new AABB[num_next_level];
			computeNextLevelBoxes(workNode->node_aabbs, next_level_boxes, segment_flags, num_segments, num_child_segments);
			per_iteration_host_memory += 2 * num_next_level * sizeof(AABB);

			// #14. Trim nodes
			// NOTE: we have to determine the memory for next level segments sizes and pivots
			uint2* next_level_tpivots = new uint2[num_next_level];
			uint2* next_level_rpivots = new uint2[num_next_level];
			uint*  next_level_tsizes  = new uint[num_next_level];
			uint*  next_level_rsizes  = new uint[num_next_level];
			trimNodesModified(tsegment_child_pivots, tsegment_child_sizes, rsegment_child_pivots, rsegment_child_sizes, 
				next_level_tpivots, next_level_tsizes, next_level_rpivots, next_level_rsizes, segment_flags, num_child_segments);
			per_iteration_host_memory += 2 * num_next_level * (sizeof(uint2) + sizeof(uint));

			// #15. Reindex the pivots
			reindexPivotsModified(next_level_tpivots, next_level_tsizes, next_level_rpivots, next_level_rsizes, num_next_level);
			dend = omp_get_wtime();
			ctr.misc_time += (dend-dstart) * 1000.0f;
			
#ifdef _DEBUG
			printf("Number of brute force segments : %u\n", num_bf_segments);
#endif
			logger.write<uint>("depth : ", depth);
			// #16. update depth
			depth++;
			// #17. create a new node
			Level* nextlevel		  = new Level();
			nextlevel->depth          = depth;
			nextlevel->num_nodes	  = num_next_level;
			nextlevel->num_tris		  = next_level_sizes.x;
			nextlevel->num_rays		  = next_level_sizes.y;
			nextlevel->node_aabbs	  = next_level_boxes;
			nextlevel->tpivots		  = next_level_tpivots;
			nextlevel->rpivots		  = next_level_rpivots;
			nextlevel->tsegment_sizes = next_level_tsizes;
			nextlevel->rsegment_sizes = next_level_rsizes;
			nextlevel->tri_idx	      = d_next_level_tri_idx;
			nextlevel->ray_idx		  = d_next_level_ray_idx;
			workingStack.push(nextlevel);

			// #18. Free up allocate space.
			// iteration temp variables.
			dstart = omp_get_wtime();
			CUDA_SAFE_RELEASE(d_split_axis);
			CUDA_SAFE_RELEASE(d_split_pos);
			CUDA_SAFE_RELEASE(d_tkeyblocks);
			CUDA_SAFE_RELEASE(d_rkeyblocks);
			CUDA_SAFE_RELEASE(d_tkeyblockStart);
			CUDA_SAFE_RELEASE(d_rkeyblockStart);
			CUDA_SAFE_RELEASE(d_tsegmentSizes);
			CUDA_SAFE_RELEASE(d_rsegmentSizes);
			CUDA_SAFE_RELEASE(d_trioffsets);
			CUDA_SAFE_RELEASE(d_rayoffsets);
			CUDA_SAFE_RELEASE(d_trikeys);
			CUDA_SAFE_RELEASE(d_raykeys);
			CUDA_SAFE_RELEASE(d_tsegment_filter_status);
			CUDA_SAFE_RELEASE(d_rsegment_filter_status);
			CUDA_SAFE_RELEASE(d_temp_ray_keys);
			CUDA_SAFE_RELEASE(d_temp_tri_keys);
			CUDA_SAFE_RELEASE(d_temp_ray_values);
			CUDA_SAFE_RELEASE(d_temp_tri_values);
			SAFE_RELEASE(segment_flags);
			SAFE_RELEASE(tsegment_child_pivots);
			SAFE_RELEASE(rsegment_child_pivots);
			SAFE_RELEASE(tsegment_child_sizes);
			SAFE_RELEASE(rsegment_child_sizes);
			SAFE_RELEASE(tkeyblocks);
			SAFE_RELEASE(tkeyblockStart);
			SAFE_RELEASE(rkeyblocks);
			SAFE_RELEASE(rkeyblockStart);
			SAFE_RELEASE(tsegmentSizes);
			SAFE_RELEASE(rsegmentSizes);
			SAFE_RELEASE(toffsets);								
			SAFE_RELEASE(roffsets);
			delete workNode;
			dend = omp_get_wtime();
			ctr.misc_time += ((dend-dstart) * 1000.0f);
			logger.write<double>("Free-up time : ", dend-dstart);
			logger.write("---------------------------------------------------------");
			tree.max_depth++;
		}

		// dump the contents of pack
#ifdef _DEBUG
		thrust::host_vector<int> debug_tribuffer(pack->tri_buffer_occupied);
		thrust::host_vector<int> debug_raybuffer(pack->ray_buffer_occupied);
		thrust::copy(pack->buffered_ray_idx.begin(), pack->buffered_ray_idx.begin() + pack->ray_buffer_occupied, debug_raybuffer.begin());
		thrust::copy(pack->buffered_tri_idx.begin(), pack->buffered_tri_idx.begin() + pack->tri_buffer_occupied, debug_tribuffer.begin());
		std::ofstream ofile("dump_fully_parallel.txt");
		ofile<<"Block Cnt : "<<pack->blockCnt<<"\n"<<"RBF occ : "<<pack->ray_buffer_occupied<<"\n"<<"TBF occ : "<<pack->tri_buffer_occupied<<"\n";
		ofile<<"BStart : "<<pack->bstart<<"\n";
		ofile.close();
		ofile.open("fully_parallel_ray_idx.txt");
		ofile<<"-------------------------------\n";
		ofile<<"Dumping Buffered Ray Ids : \n";
		for(size_t i = 0; i < pack->ray_buffer_occupied; i++) {
			ofile<<debug_raybuffer[i]<<"\n";
		}
		ofile.close();
		ofile.open("fully_parallel_tri_idx.txt");
		ofile<<"--------------------------------\n";
		ofile<<"Dumping Buffered Tri Ids : \n";
		for(size_t i = 0; i < pack->tri_buffer_occupied; i++) {
			ofile<<debug_tribuffer[i]<<"\n";
		}
		ofile.close();
		ofile.open("fully_parallel_tsegment.txt");
		ofile<<"---------------------------------\n";
		ofile<<"Dumping Tsegment sizes \n";
		for(size_t i = 0; i < pack->htri_segment_sizes.size(); i++) {
			ofile<<pack->htri_segment_sizes[i]<<"\n";
		}
		ofile.close();
		ofile.open("fully_parallel_rsegment.txt");
		ofile<<"----------------------------------\n";
		ofile<<"Dumping Rsegment sizes \n";
		for(size_t i = 0; i < pack->hray_segment_sizes.size(); i++) {
			ofile<<pack->hray_segment_sizes[i]<<"\n";
		}
		ofile.close();
#endif

		// complete the render
		if(pack->num_segments > 0) {
			double dstart = omp_get_wtime();
#ifdef _DEBUG
			logger.write<uint>("Brute Force Triangles : ", pack->tri_buffer_occupied);
			logger.write<uint>("Brute Force Rays      : ", pack->ray_buffer_occupied);
			logger.write<uint>("Brute Force Num Segments : ", pack->num_segments);
#endif
			completeBruteForceModified(*pack, d_triangles, d_rays, rtparams, ctr);
			double dend = omp_get_wtime();
#ifdef _DEBUG
			logger.write<double>("Brute force modified completed in : ", dend-dstart);
#endif
		}
		logger.write<uint>("Max Depth Reached : ", depth);
		// copy the results back
		// copy the data back to the device
		dstart = omp_get_wtime();
		thrust::copy(pack->dev_ray_maxts.begin(), pack->dev_ray_maxts.end(), thrust::device_ptr<float>(d_maxts));
		thrust::copy(pack->dev_hitids.begin(), pack->dev_hitids.end(), thrust::device_ptr<int>(d_hitids));
		dend = omp_get_wtime();
		ctr.mem_cpy_time += ((dend-dstart) * 1000.0f);
		// free memory
		delete pack;
		if(!workingStack.empty()) {
			Level* level = workingStack.top();
			workingStack.pop();
			if(level != NULL) delete level;
		}
}


