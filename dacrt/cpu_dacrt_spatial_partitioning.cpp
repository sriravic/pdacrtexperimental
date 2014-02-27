#include <dacrt/dacrt_spatial_partitioning.h>
#include <dacrt/dacrt_util.h>

#define TRI_THRESHOLD 20
#define RAY_THRESHOLD 16

// These methods are local to this file and hence not globally visible
static inline bool triBoxOverlapTest(const Triangle& T, const AABB& box, int count, int splitaxis) {
	if(count > 1000) {
		AABB tribox = T.getBounds();
		float triboxmin[3] = {tribox.bmin.x, tribox.bmin.y, tribox.bmin.z};
		float triboxmax[3] = {tribox.bmax.x, tribox.bmax.y, tribox.bmax.z};
		float boundsmin[3] = {box.bmin.x, box.bmin.y, box.bmin.z};
		float boundsmax[3] = {box.bmax.x, box.bmax.y, box.bmax.z};
		return(triBoxOverlapSimple(triboxmin, triboxmax, boundsmin, boundsmax, splitaxis));
	} else {
		float3 centroid = box.centroid();
		float3 extents = box.bmax - box.bmin;
	
	    float triverts[3][3] = {{T.v[0].x, T.v[0].y, T.v[0].z}, {T.v[1].x, T.v[1].y, T.v[1].z}, {T.v[2].x, T.v[2].y, T.v[2].z}};
		float boxhalf[3] = {extents.x * 0.5f, extents.y * 0.5f, extents.z * 0.5f};
		float boxcenter[3] = {centroid.x, centroid.y, centroid.z};

		//if(triBoxOverlap(boxcenter, boxhalf, triverts) == 1) return true;
		if(triBoxOverlapThresholded(boxcenter, boxhalf, triverts) == 1) return true;
		else return false;
	}
}

int filterRaysSpatial(RayArray& rays, int count, int* ray_ids, const AABB& bounds, int rpivot, int& testcnt) {
	int head = 0, tail = rpivot;
	float thit;
	while(head < tail) {
		while(head < tail && bounds.rayIntersect(Ray(rays.o[ray_ids[head]], rays.d[ray_ids[head]]), thit)) {
			testcnt++;
			head++;
		}
		do {
			tail--;
			testcnt++;
		} while(head < tail && !bounds.rayIntersect(Ray(rays.o[ray_ids[tail]], rays.d[ray_ids[tail]]), thit));
		if(head < tail) std::swap(ray_ids[head], ray_ids[tail]);
	}
	return head;
}

int filterShadowRaysSpatial(RayArray& rays, int count, int* sray_ids, int* pray_ids, const AABB& bounds, int rpivot, int& testcnt) {
	int head = 0, tail = rpivot;
	float thit;
	while(head < tail) {
		while(head < tail && bounds.rayIntersect(Ray(rays.o[sray_ids[head]], rays.d[sray_ids[head]]), thit)) {
			testcnt++;
			head++;
		}
		do {
			tail--;
			testcnt++;
		}while(head < tail && !bounds.rayIntersect(Ray(rays.o[sray_ids[tail]], rays.d[sray_ids[tail]]), thit));
		if(head < tail) {
			std::swap(sray_ids[head], sray_ids[tail]);
			std::swap(pray_ids[head], pray_ids[tail]);
		}
	}
	return head;
}

int filterTrianglesSpatial(TriangleArray& triangles, int count, int* tri_ids, const AABB& bounds, int tpivot, int splitaxis, int& testcnt) {
	int head = 0, tail = tpivot;
	while(head < tail) {
		while(head < tail && triBoxOverlapTest(Triangle(triangles.v0[tri_ids[head]], triangles.v1[tri_ids[head]], triangles.v2[tri_ids[head]]), bounds, tpivot, splitaxis)) { 
			head++;
			testcnt++;
		}
		do {
			tail--;
			testcnt++;
		}while(head < tail && !triBoxOverlapTest(Triangle(triangles.v0[tri_ids[tail]], triangles.v1[tri_ids[tail]], triangles.v2[tri_ids[tail]]), bounds, tpivot, splitaxis));
		if(head < tail) std::swap(tri_ids[head], tri_ids[tail]);
	}
	return head;
}

void cpuDacrtSpatialPartitioning(const AABB& space, TriangleArray& triangles, int num_triangles, int* tri_idx_array, int tpivot,
	RayArray& rays, int num_rays, int* ray_idx_array, int rpivot, float* maxts, int* hitids, Counters& ctr) {
		
		if(tpivot < TRI_THRESHOLD || rpivot < RAY_THRESHOLD) {
			double start = omp_get_wtime();
			for(int r = 0; r < rpivot; r++) {
				int ridx = ray_idx_array[r];
				const Ray ir(rays.o[ridx], rays.d[ridx]);
				for(int t = 0; t < tpivot; t++) {
					int tidx = tri_idx_array[t];
					Triangle it(triangles.v0[tidx], triangles.v1[tidx], triangles.v2[tidx]);
					double u = 0, v = 0, xt = 0;
					if(rayIntersect<double>(it, ir, u, v, xt)) {
						ctr.raytri++;
						if(xt > 0 && xt < maxts[ridx]) {
							maxts[ridx] = static_cast<float>(xt);
							hitids[ridx] = tidx;
						}
					}
				}
			}
			double end = omp_get_wtime();
			ctr.brute_force_time += static_cast<float>(end-start);
		} else {
			
			float3 extents = space.bmax - space.bmin;
			{
				// left child
				AABB left = space;
				double start = omp_get_wtime();
				int newtpivot, splitaxis;
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
				int newrpivot = filterRaysSpatial(rays, num_rays, ray_idx_array, left, rpivot, ctr.raybox);
				end = omp_get_wtime();

				ctr.ray_filter_time += static_cast<float>(end-start);

				cpuDacrtSpatialPartitioning(left, triangles, num_triangles, tri_idx_array, newtpivot, rays, num_rays, ray_idx_array, newrpivot, maxts, hitids, ctr);
				
			}
			{
				double start = omp_get_wtime();
				int newtpivot, splitaxis;
				AABB right = space;
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

				cpuDacrtSpatialPartitioning(right, triangles, num_triangles, tri_idx_array, newtpivot, rays, num_rays, ray_idx_array, newrpivot, maxts, hitids, ctr);
			}
		}
}

void cpuDacrtSpatialShadows(const AABB& space, TriangleArray& triangles, int num_triangles, int* tri_idx_array, int tpivot, 
	RayArray& rays, int num_rays, int* ray_idx_array, int* pray_idx_array, int rpivot, bool* shadows, Counters& ctr) {

		if(tpivot < TRI_THRESHOLD || rpivot < RAY_THRESHOLD) {
			double start = omp_get_wtime();
			for(int r = 0; r < rpivot; r++) {
				int ridx = ray_idx_array[r];
				const Ray ir(rays.o[ridx], rays.d[ridx]);
				if(shadows[ridx] == false) {
					for(int t = 0; t < tpivot; t++) {
						int tidx = tri_idx_array[t];
						Triangle it(triangles.v0[tidx], triangles.v1[tidx], triangles.v2[tidx]);
						double u = 0, v = 0, xt = 0;
						// already a shadow candidate?
						if(rayIntersect<double>(it, ir, u, v, xt)) {
							ctr.raytri++;
							if(xt > 0) {
								shadows[ridx] = true;			// set the correct value at the correct index
								break;
							}
						}
					}
				}
			}
			double end = omp_get_wtime();
			ctr.brute_force_time += static_cast<float>(end-start);
		} else {
			
			float3 extents = space.bmax - space.bmin;
			{
				// left child
				AABB left = space;
				double start = omp_get_wtime();
				int newtpivot, splitaxis;
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
				int newrpivot = filterShadowRaysSpatial(rays, num_rays, ray_idx_array, pray_idx_array, left, rpivot, ctr.raybox);
				end = omp_get_wtime();

				ctr.ray_filter_time += static_cast<float>(end-start);

				cpuDacrtSpatialShadows(left, triangles, num_triangles, tri_idx_array, newtpivot, rays, num_rays, ray_idx_array, pray_idx_array, newrpivot, shadows, ctr);
				
			}
			{
				double start = omp_get_wtime();
				int newtpivot, splitaxis;
				AABB right = space;
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
				int newrpivot = filterShadowRaysSpatial(rays, num_rays, ray_idx_array, pray_idx_array, right, rpivot, ctr.raybox);
				end = omp_get_wtime();

				ctr.ray_filter_time += static_cast<float>(end-start);

				cpuDacrtSpatialShadows(right, triangles, num_triangles, tri_idx_array, newtpivot, rays, num_rays, ray_idx_array, pray_idx_array, newrpivot, shadows, ctr);
			}
		}

}