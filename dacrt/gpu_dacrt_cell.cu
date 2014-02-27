#include <dacrt/dacrt.h>
#include <util/cutimer.h>
#include <util/util.h>


extern "C" void dacrtCompleteRender(ParallelPack& pack, TriangleArray& dev_triangles, RayArray& dev_rays, DacrtRunTimeParameters& rtparams, Counters& ctr);
extern "C" __global__ void updateMinKernel(int* ray_id, float* min_hits, int* minhit_ids, float* global_min, int* global_hits, int num_rays);
extern "C" __global__ void segmentedBruteForce(RayArray rays, TriangleArray triangles, int* buffered_ray_ids, int ray_buffer_occupied, int* buffered_tri_ids, int tri_buffer_occupied,
	int* ray_segment_sizes, int* tri_segment_sizes, int* ray_segment_start, int* tri_segment_start, int num_segments, float* maxts,	int* hitids,
	int num_threads_launched, int num_blocks_launched);


/**
CELL BASED VERSION
******************

Idea: If I can somehow combine the two sorting operations on both the left and right child of a single node, then I will be saving twice the savings in sort operations

Expected Difficulties: How are we going to maintain state info across the nodes now? We need detailed plans
TODO: Add detailed comments below before doing any operation

 */

struct Cell
{
	AABB parent, left, right;
	int ptpivot, prpivot, ltpivot, rtpivot, lrpivot, rrpivot;	// p*pivot => parent pivot; l*pivot=>left child ; r*pivot => right child
	thrust::device_vector<int> ray_idx;
	thrust::device_vector<int> triangle_idx;
	//thrust::device_vector<int> ray_occupancy;
	//thrust::device_vector<int> tri_occupancy;
	Cell() {
		ptpivot = prpivot = ltpivot = rtpivot = lrpivot = rrpivot = 0;
	}
};

struct BruteForceCell
{
	int tricnt, raycnt;
	thrust::device_vector<int> ray_idx;
	thrust::device_vector<int> triangle_idx;
};

// cell filter kernels
/// NOTE: Key idea here is to assign values so that we can effectively split the left and right in one pass
///       So, left = 1, right = 3 and both = 2
__global__ void triCellFilterKernel(AABB left, AABB right, float3* v0, float3* v1, float3* v2, int* tri_idx, int num_tris, int* occupancy) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if(tid < num_tris) {
		/// NOTE :
	
		float3 lcentroid = (left.bmin + left.bmax) * 0.5f;
		float3 rcentroid = (right.bmin + right.bmax) * 0.5f;
		float3 lextents = left.bmax - left.bmin;
		float3 rextents = right.bmax - right.bmin;
		int triangle_id = tri_idx[tid];
	
	    float triverts[3][3] = {{v0[triangle_id].x, v0[triangle_id].y, v0[triangle_id].z}, 
								{v1[triangle_id].x, v1[triangle_id].y, v1[triangle_id].z}, 
								{v2[triangle_id].x, v2[triangle_id].y, v2[triangle_id].z}};
		float lboxhalf[3] = {lextents.x * 0.5f, lextents.y * 0.5f, lextents.z * 0.5f};
		float rboxhalf[3] = {rextents.x * 0.5f, rextents.y * 0.5f, rextents.z * 0.5f};
		float lboxcenter[3] = {lcentroid.x, lcentroid.y, lcentroid.z};
		float rboxcenter[3] = {rcentroid.x, rcentroid.y, rcentroid.z};

		/// TODO: Can we replace this costly test with the simpler test? Any jump in total performance and not only this small step.

		int lo = triBoxOverlap(lboxcenter, lboxhalf, triverts);
		int ro = triBoxOverlap(rboxcenter, rboxhalf, triverts);
		//if(lo == 1 && ro == 1) printf("Hey : threadIdx : %d\n", tid);
		int val = lo && ro ? 2 : (lo ? 1 : 3);
		occupancy[tid] = val;
	}
};

__global__ void rayCellFilterKernel(AABB left, AABB right, float3* o, float3* dir, int* ray_ids, int num_rays, int* occupancy) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if(tid < num_rays) {
		int ray_id = ray_ids[tid];
		Ray ray(o[ray_id], dir[ray_id]);
		float thit;
		bool lo = left.rayIntersect(ray, thit);
		bool ro = right.rayIntersect(ray, thit);
		// left = 1, right = 3, both = 2 and no_hit = 4
		int val = lo && ro ? 2 : (lo ? 1 : (ro ? 3 : 4));
		occupancy[tid] = val;
	}
}

// we call this function to complete the parallel brute force. We've written this as a function call so that we can use it elsewhere
extern "C"
void completeBruteForce(ParallelPack& pack, TriangleArray& d_triangles, RayArray& d_rays, DacrtRunTimeParameters& rtparams, Counters& ctr) {
	thrust::device_vector<int> ray_segment_start(pack.num_segments);
	thrust::device_vector<int> tri_segment_start(pack.num_segments);
	thrust::exclusive_scan(pack.tri_segment_sizes.begin(), pack.tri_segment_sizes.begin() + pack.num_segments, tri_segment_start.begin());
	thrust::exclusive_scan(pack.ray_segment_sizes.begin(), pack.ray_segment_sizes.begin() + pack.num_segments, ray_segment_start.begin());
				
	int num_blocks = pack.num_segments;
	int num_threads_per_block = rtparams.NUM_RAYS_PER_BLOCK;
	Timer segtimer("Segmented timer");
	segtimer.start();
	segmentedBruteForce<<<num_blocks, num_threads_per_block>>>(d_rays, d_triangles, thrust::raw_pointer_cast(&pack.buffered_ray_idx[0]), 
					pack.ray_buffer_occupied, thrust::raw_pointer_cast(&pack.buffered_tri_idx[0]), pack.tri_buffer_occupied, 
					thrust::raw_pointer_cast(&pack.ray_segment_sizes[0]), thrust::raw_pointer_cast(&pack.tri_segment_sizes[0]), 
					thrust::raw_pointer_cast(&ray_segment_start[0]), thrust::raw_pointer_cast(&tri_segment_start[0]), 
					pack.num_segments, thrust::raw_pointer_cast(&pack.buffered_ray_maxts[0]),
					thrust::raw_pointer_cast(&pack.buffered_ray_hitids[0]), num_threads_per_block * num_blocks, num_blocks);
	segtimer.stop();
	ctr.brute_force_time += segtimer.get_ms();

	Timer segsort("Segmented SOrt");
	segsort.start();

	thrust::sort_by_key(pack.buffered_ray_idx.begin(), pack.buffered_ray_idx.begin() + pack.ray_buffer_occupied,
					thrust::make_zip_iterator(thrust::make_tuple(pack.buffered_ray_maxts.begin(), pack.buffered_ray_hitids.begin())));
	segsort.stop();
	ctr.seg_sort_time += segsort.get_ms();

	static thrust::device_vector<int> ray_idx(rtparams.BUFFER_SIZE);
	static thrust::device_vector<float> ray_maxts(rtparams.BUFFER_SIZE);
	static thrust::device_vector<int> ray_hitids(rtparams.BUFFER_SIZE);
	static thrust::equal_to<int> pred;
				
	typedef thrust::device_vector<int>::iterator iter;
	typedef thrust::device_vector<float>::iterator fiter;
	typedef thrust::zip_iterator<thrust::tuple<fiter, iter> > zippy;
	thrust::pair<iter, zippy> minend;
				
	MinHitFunctor<thrust::tuple<float, int> > min_hit_functor;

	Timer segred("Segmented Reduce");
	segred.start();
	minend = thrust::reduce_by_key(pack.buffered_ray_idx.begin(), pack.buffered_ray_idx.begin() + pack.ray_buffer_occupied,
					thrust::make_zip_iterator(thrust::make_tuple(pack.buffered_ray_maxts.begin(), pack.buffered_ray_hitids.begin())),
					ray_idx.begin(), thrust::make_zip_iterator(thrust::make_tuple(ray_maxts.begin(), ray_hitids.begin())),
					pred,
					min_hit_functor);
	segred.stop();
	ctr.reduction_time += segred.get_ms();

	int num_valid_keys = minend.first - ray_idx.begin();
	num_threads_per_block = 512;
	num_blocks = num_valid_keys / num_threads_per_block + (num_valid_keys % num_threads_per_block != 0);

	Timer update("update");
	update.start();
	updateMinKernel<<<num_blocks, num_threads_per_block>>>(thrust::raw_pointer_cast(&ray_idx[0]), thrust::raw_pointer_cast(&ray_maxts[0]), thrust::raw_pointer_cast(&ray_hitids[0]),
					thrust::raw_pointer_cast(&pack.dev_ray_maxts[0]), thrust::raw_pointer_cast(&pack.dev_hitids[0]), num_valid_keys);
	update.stop();
	ctr.update_min_time += update.get_ms();


	pack.buffered_ray_idx.clear();
	pack.buffered_tri_idx.clear();
	pack.tri_segment_sizes.clear();
	pack.ray_segment_sizes.clear();
	pack.buffered_ray_maxts.clear();
	pack.buffered_ray_hitids.clear();
	pack.segment_ids.clear();
							
	pack.ray_buffer_occupied = 0;
	pack.tri_buffer_occupied = 0;
	pack.num_segments = 0;
}

void gpuDacrtCell(AABB& root, TriangleArray& d_triangles, int* tri_idx_array, int tpivot, RayArray& rays, int* ray_idx_array, int rpivot, 
	float* h_maxts, int* h_hitids, 
	DacrtRunTimeParameters& rtparams, Counters& ctr, Logger& logger) { 

	std::stack<Cell> recursion_stack;
	std::queue<BruteForceCell> bruteForceQueue;

	Cell rootcell;
	rootcell.parent = root;
	rootcell.prpivot = rpivot;
	rootcell.ptpivot = tpivot;
	rootcell.ray_idx.resize(rpivot);
	rootcell.triangle_idx.resize(tpivot);
	thrust::copy(thrust::device_ptr<int>(tri_idx_array), thrust::device_ptr<int>(tri_idx_array) + tpivot, rootcell.triangle_idx.begin());
	thrust::copy(thrust::device_ptr<int>(ray_idx_array), thrust::device_ptr<int>(ray_idx_array) + rpivot, rootcell.ray_idx.begin());

	recursion_stack.push(rootcell);

	// we will use this vector throughout the iterative procedure
	thrust::device_vector<int> ray_occupancy;
	thrust::device_vector<int> tri_occupancy;
	int tleft = 0, tright = 0, tboth = 0;
	int rleft = 0, rright = 0, rboth = 0, rnone = 0;
	int lefttricnt = 0, righttricnt = 0;
	int leftraycnt = 0, rightraycnt = 0;

	// initialize the parallel pack
	thrust::device_vector<int>		buffered_ray_idx(rtparams.BUFFER_SIZE);
	thrust::device_vector<int>		buffered_tri_idx(rtparams.BUFFER_SIZE);
	thrust::device_vector<int>		segment_ids(rtparams.MAX_SEGMENTS);
	thrust::device_vector<int>		ray_segment_sizes(rtparams.MAX_SEGMENTS);
	thrust::device_vector<int>		tri_segment_sizes(rtparams.MAX_SEGMENTS);
	thrust::device_vector<float>	buffered_ray_maxts(rtparams.BUFFER_SIZE, FLT_MAX);
	thrust::device_vector<int>		buffered_ray_hitids(rtparams.BUFFER_SIZE, -1);
	thrust::device_vector<float>    dev_ray_maxts(rpivot, FLT_MAX);
	thrust::device_vector<int>	    dev_hitids(rpivot, -1);

	int ray_buffer_occupied = 0;
	int tri_buffer_occupied = 0;
	int num_segments = 0;
	int debugctr = 0;
	ParallelPack pack(buffered_ray_idx, buffered_tri_idx, segment_ids, ray_segment_sizes, tri_segment_sizes, buffered_ray_maxts, buffered_ray_hitids,
		dev_ray_maxts, dev_hitids, ray_buffer_occupied, tri_buffer_occupied, num_segments);
	
	do
	{
		
		Cell cell = recursion_stack.top();
		recursion_stack.pop();
		debugctr++;


		int NUM_THREADS_PER_BLOCK = rtparams.NUM_RAYS_PER_BLOCK;
		int NUM_BLOCKS			  = (cell.ptpivot / NUM_THREADS_PER_BLOCK) + (cell.ptpivot % NUM_THREADS_PER_BLOCK != 0);
		float split_pos;
		splitSpatialMedian(cell.parent, cell.left, cell.right, split_pos);

		// reallocate memory for ray and tri occupancy;
		ray_occupancy.resize(cell.prpivot);
		tri_occupancy.resize(cell.ptpivot);

		Timer trifiltertimer("tri filter timer");

		trifiltertimer.start();
		triCellFilterKernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(cell.left, cell.right, d_triangles.v0, d_triangles.v1, d_triangles.v2, 
			thrust::raw_pointer_cast(&cell.triangle_idx[0]), cell.ptpivot, thrust::raw_pointer_cast(&tri_occupancy[0]));
		trifiltertimer.stop();
		ctr.tri_filter_time += trifiltertimer.get_ms();

		Timer trisorttimer("tri sort timer");
		trisorttimer.start();
		thrust::sort_by_key(tri_occupancy.begin(), tri_occupancy.end(), cell.triangle_idx.begin());
		trisorttimer.stop();
		ctr.trisortbykey_time += trisorttimer.get_ms();

		std::vector<int> temp_keys(3);
		std::vector<int> temp_values(3);
		thrust::device_vector<int> dtemp_keys(3);
		thrust::device_vector<int> dtemp_values(3);
		//std::vector<int> debug(cell.ptpivot);
		//thrust::copy(tri_occupancy.begin(), tri_occupancy.end(), debug.begin());
		thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> end;

		Timer triredtimer("tri red timer");
		triredtimer.start();
		end = thrust::reduce_by_key(tri_occupancy.begin(), tri_occupancy.end(), tri_occupancy.begin(), dtemp_keys.begin(), dtemp_values.begin());
		triredtimer.stop();
		ctr.trireduction_time += triredtimer.get_ms();

		
		thrust::copy(dtemp_keys.begin(),dtemp_keys.end(), temp_keys.begin());
		thrust::copy(dtemp_values.begin(), dtemp_values.end(), temp_values.begin());

		int num_types = end.first - dtemp_keys.begin();
		int total_tris = 0;
		bool bleft = false, bright = false, bboth = false;
		
		if(num_types == 3) {
			bleft = bright = bboth = true;
			tleft = temp_values[0];
			tboth = temp_values[1]/2;
			tright = temp_values[2]/3;			
		} else if(num_types == 2) {
			// first determine the keys
			// only two possible combos.!!
			if(temp_keys[0] == 1) {
				bleft = true;
				if(temp_keys[1] == 2) bboth = true;
				else bright = true;
			} else if(temp_keys[0] == 2) {
				bleft = false;
				bboth = true;
				bright = true;
			}
			
			if(bleft) {
				tleft = temp_values[0];
				if(bboth) {
					tboth = temp_values[1]/2;
				} else if(bright) {
					tright = temp_values[1]/3;
				}
			} else if(bboth) {
				tboth = temp_values[0]/2;
				if(bright) {
					tright = temp_values[1]/3;
				}
			}
		} else if(num_types == 1) {
			if(temp_keys[0] == 1) bleft = true;
			else if(temp_keys[0] == 2) bboth = true;
			else bright = true;
			if(bleft) {
				tleft = temp_values[0];
			}
			else if(bboth) {
				tboth = temp_values[0]/2;
			} else if(bright) {
				tright = temp_values[0]/3;
			}
		}

		total_tris = tleft + tright + tboth;
		if(total_tris != cell.ptpivot) {
			std::cerr<<"Error in triangle count ; counter value : "<<debugctr<<"\n";
		}
		lefttricnt = tleft + tboth;
		righttricnt = tright + tboth;

		if(bleft && bboth) {
			cell.ltpivot = tleft + tboth;
			cell.rtpivot = tleft + 1;		// even if we dont have separate right values, both counts as right. So this is perfectly valid.
		}
		else if(!bleft && bboth) {
			cell.ltpivot = tboth;
			cell.rtpivot = 0;
		} else if(bleft && !bboth) {
			// we have got only left triangles and no common ones. we have to check if we have right triangles only.!!
			cell.ltpivot = tleft;
			if(bright) cell.rtpivot = tleft + 1;
			else cell.rtpivot = total_tris;		// set to the end of the list.!
		} else if(!bleft && !bboth) {
			cell.ltpivot = 0;
			if(bright) cell.rtpivot = 0;				// this is the only way we can do this.
			else cell.rtpivot = total_tris;
		}
		// ray stuff
				/**
		The arrangement of ray data

		LLLLLLLLLLLLLL - BBBBBBBB - RRRRRRRRRRR - NNNNNNNNNNNNNN
						 ^      ^             ^
						 |      |			  |
						right   left          right
						start   end			  end
		*/

		// rest all the boolean values here
		bleft = bright = bboth = false;
		bool bnone = false;
		
		NUM_BLOCKS = (cell.prpivot / NUM_THREADS_PER_BLOCK) + (cell.prpivot % NUM_THREADS_PER_BLOCK != 0);

		Timer rayfiltertimer("ray filter timer");

		rayfiltertimer.start();
		rayCellFilterKernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(cell.left, cell.right, rays.o, rays.d, 
			thrust::raw_pointer_cast(&cell.ray_idx[0]), cell.prpivot, thrust::raw_pointer_cast(&ray_occupancy[0]));
		rayfiltertimer.stop();
		ctr.ray_filter_time += rayfiltertimer.get_ms();

		Timer raysortimer("ray sort timer");
		raysortimer.start();
		thrust::sort_by_key(ray_occupancy.begin(), ray_occupancy.end(), cell.ray_idx.begin());
		raysortimer.stop();
		ctr.raysortbykey_time += raysortimer.get_ms();

		dtemp_keys.resize(4);
		dtemp_values.resize(4);
		temp_keys.resize(4);
		temp_values.resize(4);

		Timer rayreduction("ray reduction timer");
		rayreduction.start();
		end = thrust::reduce_by_key(ray_occupancy.begin(), ray_occupancy.end(), ray_occupancy.begin(), dtemp_keys.begin(), dtemp_values.begin());
		rayreduction.stop();
		ctr.rayreduction_time += rayreduction.get_ms();
		
		thrust::copy(dtemp_keys.begin(), dtemp_keys.end(), temp_keys.begin());
		thrust::copy(dtemp_values.begin(), dtemp_values.end(), temp_values.begin());


		num_types = end.first - dtemp_keys.begin();
		int total_rays = 0;

		/// NOTE: We need to have a detailed ray splitting procedure similar to the triangle one because I envisage situations in which child boxes might not
		///       have any rays at all.! Possible scenario: secondary ray bounces?
		if(num_types == 4) {
			// LBRN branch
			bleft = bright = bboth = bnone = true;
			rleft = temp_values[0];
			rboth = temp_values[1]/2;
			rright = temp_values[2]/3;
			rnone = temp_values[3]/4;
		} else if(num_types == 3) {
			if(temp_keys[0] == 1) {
				bleft = true;
				rleft = temp_values[0];
				if(temp_keys[1] == 2) {
					// LB branch
					bboth = true;
					rboth = temp_values[1]/2;
					if(temp_keys[2] == 3) {
						// LBR branch
						bright = true;
						rright = temp_values[2]/3;
					} else if(temp_keys[2] == 4) {
						// LBN branch
						bnone = true;
						rnone = temp_values[2]/4;
					}
				} else if(temp_keys[1] == 3) {
					// left but not both, but right and none branch
					// LRN branch
					bright = true;
					rright = temp_values[1]/3;
					bnone = true;
					rnone = temp_values[2]/4;
				}
			} else if(temp_keys[0] == 2) {
				// Both branch
				// possible values are BRN
				bboth = bright = bnone = true;
				rboth = temp_values[0]/2;
				rright = temp_values[1]/3;
				rnone = temp_values[2]/4;
			}
		} else if(num_types == 2) {
			if(temp_keys[0] == 1) {
				// LB or LR or LN
				bleft = true;
				rleft = temp_values[0];
				if(temp_keys[1] == 2) {
					bboth = true;
					rboth = temp_values[1]/2;
				} else if(temp_keys[1] == 3) {
					bright = true;
					rright = temp_values[1]/3;
				} else if(temp_keys[2] == 4) {
					bnone = true;
					rnone = temp_values[1]/4;
				}
			} else if(temp_keys[0] == 2) {
				// BR or BN
				bboth = true;
				rboth = temp_values[0]/2;
				if(temp_keys[1] == 3) {
					bright = true;
					rright = temp_values[1]/3;
				} else if(temp_keys[1] == 4) {
					bnone = true;
					rnone = temp_values[1]/4;
				}
			} else if(temp_keys[0] == 3) {
				// RN branch
				bright = bnone = true;
				rright = temp_values[0]/3;
				rnone = temp_values[1]/4;
			}
		} else if(num_types == 1) {
			if(temp_keys[0] == 1) {
				bleft = true; rleft = temp_values[0];
			} else if(temp_keys[0] == 2) {
				bboth = true; rboth = temp_values[0]/2;
			} else if(temp_keys[0] == 3) {
				bright = true; rright = temp_values[0]/3;
			} else {
				bnone = true; rnone = temp_values[0]/4;
			}
		}

		// compute all rays
		total_rays = rleft + rboth + rright + rnone;
		if(total_rays != cell.prpivot) {
			std::cerr<<"Error in ray count; counter value : !!"<<debugctr<<"\n";
		}
		leftraycnt = rleft + rboth;
		rightraycnt = rright + rboth;
		// the following code is to calculate the right rays indices to copy correctly
		if(bleft) {
			cell.rrpivot = rleft + 1;
		} else if(!bleft) {
			cell.rrpivot = 0;
		} else if(!bleft && !bboth) {
			if(bright) cell.rrpivot = 0;
			else cell.rrpivot = total_rays;	// no rays come inside this box?
		}

		// check the conditions for both the left and right brute force children
		
		if(righttricnt < 256 || rightraycnt < 256) {
			//printf("Brute Cell : right tri cnt : %d Right ray cnt : %d\n", righttricnt, rightraycnt);
			/*BruteForceCell brutecell;
			brutecell.tricnt = righttricnt;
			brutecell.raycnt = rightraycnt;
			brutecell.triangle_idx.resize(righttricnt);
			brutecell.ray_idx.resize(rightraycnt);
			thrust::copy(cell.triangle_idx.begin() + cell.rtpivot, cell.triangle_idx.end(), brutecell.triangle_idx.begin());
			thrust::copy(cell.ray_idx.begin() + cell.rrpivot, cell.ray_idx.end() - rnone, brutecell.ray_idx.begin());
			bruteForceQueue.push(brutecell);*/

			if((pack.ray_buffer_occupied + rightraycnt) < rtparams.BUFFER_SIZE && (pack.tri_buffer_occupied + righttricnt) < rtparams.BUFFER_SIZE && pack.num_segments < rtparams.MAX_SEGMENTS) {
				// replacing with segmented code
				Timer memcpytimer("mem cpy timer");
				memcpytimer.start();
				thrust::copy(cell.triangle_idx.begin() + cell.rtpivot, cell.triangle_idx.end(), pack.buffered_tri_idx.begin() + pack.tri_buffer_occupied);
				thrust::copy(cell.ray_idx.begin() + cell.rrpivot, cell.ray_idx.end() - rnone, pack.buffered_ray_idx.begin() + pack.ray_buffer_occupied);
				memcpytimer.stop();
				ctr.mem_cpy_time += memcpytimer.get_ms();

				pack.tri_segment_sizes[pack.num_segments] = righttricnt;
				pack.ray_segment_sizes[pack.num_segments] = rightraycnt;
				pack.segment_ids[pack.num_segments] = pack.num_segments;
				pack.num_segments++;		// increment the count
				pack.ray_buffer_occupied += rightraycnt;
				pack.tri_buffer_occupied += righttricnt;
			} else {
				// complete the brute force and reset stuff
				completeBruteForce(pack, d_triangles, rays, rtparams, ctr);
				Timer memcpytimer("mem cpy timer");
				memcpytimer.start();
				thrust::copy(cell.triangle_idx.begin() + cell.rtpivot, cell.triangle_idx.end(), pack.buffered_tri_idx.begin() + pack.tri_buffer_occupied);
				thrust::copy(cell.ray_idx.begin() + cell.rrpivot, cell.ray_idx.end() - rnone, pack.buffered_ray_idx.begin() + pack.ray_buffer_occupied);
				memcpytimer.stop();
				ctr.mem_cpy_time += memcpytimer.get_ms();

				pack.tri_segment_sizes[pack.num_segments] = righttricnt;
				pack.ray_segment_sizes[pack.num_segments] = rightraycnt;
				pack.segment_ids[pack.num_segments] = pack.num_segments;
				pack.num_segments++;		// increment the count
				pack.ray_buffer_occupied += rightraycnt;
				pack.tri_buffer_occupied += righttricnt;
			}
		} else {
			Cell rightcell;
			//printf("Right Tri cnt : %d Right Ray Cnt : %d\n", righttricnt, rightraycnt);
			rightcell.parent = cell.right;
			rightcell.ptpivot = righttricnt;
			rightcell.prpivot = rightraycnt;
			rightcell.triangle_idx.resize(righttricnt);
			rightcell.ray_idx.resize(rightraycnt);

			Timer memcpytimer("mem cpy timer");
			memcpytimer.start();
			thrust::copy(cell.triangle_idx.begin() + cell.rtpivot, cell.triangle_idx.end(), rightcell.triangle_idx.begin());
			thrust::copy(cell.ray_idx.begin() + cell.rrpivot, cell.ray_idx.end() - rnone, rightcell.ray_idx.begin());
			memcpytimer.stop();
			ctr.mem_cpy_time += memcpytimer.get_ms();

			recursion_stack.push(rightcell);
		}
		
		if(lefttricnt < 256 || leftraycnt < 256) {
			//printf("Brute Cell : left tricnt : %d ; left raycnt : %d\n", lefttricnt, leftraycnt);
			/*
			BruteForceCell brutecell;
			brutecell.tricnt = lefttricnt;
			brutecell.raycnt = leftraycnt;
			brutecell.triangle_idx.resize(lefttricnt);
			brutecell.ray_idx.resize(leftraycnt);
			thrust::copy(cell.triangle_idx.begin(), cell.triangle_idx.begin() + lefttricnt, brutecell.triangle_idx.begin());
			thrust::copy(cell.ray_idx.begin(), cell.ray_idx.begin() + leftraycnt, brutecell.ray_idx.begin());
			bruteForceQueue.push(brutecell);
			*/
			if((pack.ray_buffer_occupied + leftraycnt) < rtparams.BUFFER_SIZE && (pack.tri_buffer_occupied + lefttricnt) < rtparams.BUFFER_SIZE && pack.num_segments < rtparams.MAX_SEGMENTS) {
				// replacing with segmented code
				Timer memcpytimer("mem cpy timer");
				memcpytimer.start();
				thrust::copy(cell.triangle_idx.begin(), cell.triangle_idx.begin() + lefttricnt, pack.buffered_tri_idx.begin() + pack.tri_buffer_occupied);
				thrust::copy(cell.ray_idx.begin(), cell.ray_idx.begin() + leftraycnt, pack.buffered_ray_idx.begin() + pack.ray_buffer_occupied);
				memcpytimer.stop();
				ctr.mem_cpy_time += memcpytimer.get_ms();

				pack.tri_segment_sizes[pack.num_segments] = lefttricnt;
				pack.ray_segment_sizes[pack.num_segments] = leftraycnt;
				pack.segment_ids[pack.num_segments] = pack.num_segments;
				pack.num_segments++;		// increment the count
				pack.ray_buffer_occupied += leftraycnt;
				pack.tri_buffer_occupied += lefttricnt;
			} else {
				// complete the brute force and reset stuff
				completeBruteForce(pack, d_triangles, rays, rtparams, ctr);
				Timer memcpytimer("mem cpy timer");
				memcpytimer.start();
				thrust::copy(cell.triangle_idx.begin(), cell.triangle_idx.begin() + lefttricnt, pack.buffered_tri_idx.begin() + pack.tri_buffer_occupied);
				thrust::copy(cell.ray_idx.begin(), cell.ray_idx.begin() + leftraycnt, pack.buffered_ray_idx.begin() + pack.ray_buffer_occupied);
				memcpytimer.stop();
				ctr.mem_cpy_time += memcpytimer.get_ms();

				pack.tri_segment_sizes[pack.num_segments] = lefttricnt;
				pack.ray_segment_sizes[pack.num_segments] = leftraycnt;
				pack.segment_ids[pack.num_segments] = pack.num_segments;
				pack.num_segments++;		// increment the count
				pack.ray_buffer_occupied += leftraycnt;
				pack.tri_buffer_occupied += lefttricnt;
			}
		} else {
			//printf("Left Tri cnt : %d ; Left Ray Cnt : %d\n", lefttricnt, leftraycnt);
			Cell leftcell;
			leftcell.parent = cell.left; 
			leftcell.ptpivot = lefttricnt;
			leftcell.prpivot = leftraycnt;

			leftcell.triangle_idx.resize(lefttricnt);
			leftcell.ray_idx.resize(leftraycnt);
			Timer memcpytimer("mem copy timer");
			memcpytimer.start();
			thrust::copy(cell.triangle_idx.begin(), cell.triangle_idx.begin() + lefttricnt, leftcell.triangle_idx.begin());
			thrust::copy(cell.ray_idx.begin(), cell.ray_idx.begin() + leftraycnt, leftcell.ray_idx.begin());
			memcpytimer.stop();
			ctr.mem_cpy_time += memcpytimer.get_ms();
			recursion_stack.push(leftcell);
		}

		// reset all values
		tleft = 0, tright = 0, tboth = 0;
		lefttricnt = 0, righttricnt = 0;
		rleft = 0, rright = 0, rboth = 0, rnone = 0;
		leftraycnt = 0, rightraycnt = 0;

	}while(!recursion_stack.empty());


	// complete rendering of the brute force queue here
	if(pack.num_segments > 0) 
		dacrtCompleteRender(pack, d_triangles, rays, rtparams, ctr);

	// complete the copy
	thrust::copy(dev_ray_maxts.begin(), dev_ray_maxts.end(), h_maxts);
	thrust::copy(dev_hitids.begin(), dev_hitids.end(), h_hitids);
}

