#include <dacrt/dacrt.h>

#define TRI_THRESHOLD 16
#define RAY_THRESHOLD 20

int filterTrianglesObject(AabbArray& tri_aabbs, int count, int* tri_ids, int tpivot, float median, bool (*test)(const float3& , float), int& testcnt) {
	int head = 0, tail = tpivot;
	while(head < tail) {
		while(head < tail && test(tri_aabbs.centroid[tri_ids[head]], median)) {
			testcnt++;
			head++;
		}
		do {
			testcnt++;
			tail--;
		} while(head < tail && !test(tri_aabbs.centroid[tri_ids[tail]], median));
		if(head < tail) std::swap(tri_ids[head], tri_ids[tail]);
	}
	return head;
}

int filterRaysObject(RayArray& rays, int count, int* ray_ids, const AABB& bounds, int rpivot, int& testcnt) {
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

int filterShadowRaysObject(RayArray& rays, int count, int* ray_ids, int* pray_ids, const AABB& bounds, int rpivot, int& testcnt) {
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
	}
	if(head < tail) {
		std::swap(ray_ids[head], ray_ids[tail]);
		std::swap(pray_ids[head], pray_ids[tail]);
	}
	return head;
}

void cpuDacrtObjectPartitioning(const AABB& space, TriangleArray& triangles, AabbArray& tri_aabbs, int num_triangles, int* tri_idx_array, int tpivot,
	RayArray& rays, int num_rays, int* ray_idx_array, int rpivot, float* maxts, int* hitids, Counters& ctr) {
		
		if(tpivot < TRI_THRESHOLD || rpivot < RAY_THRESHOLD) {
			// we create a work queue and copy stuff?
BRUTE_FORCE:
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
				double start = omp_get_wtime();
				int newtpivot;
				if(extents.x > extents.y && extents.x > extents.z) {
					float median = (space.bmax.x + space.bmin.x) * 0.5f;
					newtpivot = filterTrianglesObject(tri_aabbs, num_triangles, tri_idx_array, tpivot, median, splitTriangleLeftX, ctr.tribox);
				} else if(extents.y > extents.x && extents.y > extents.z) {
					float median = (space.bmax.y + space.bmin.y) * 0.5f;
					newtpivot = filterTrianglesObject(tri_aabbs, num_triangles, tri_idx_array, tpivot, median, splitTriangleLeftY, ctr.tribox);
				} else {
					float median = (space.bmax.z + space.bmin.z) * 0.5f;
					newtpivot = filterTrianglesObject(tri_aabbs, num_triangles, tri_idx_array, tpivot, median, splitTriangleLeftZ, ctr.tribox);
				}

				AABB left;
				for(int t = 0; t < newtpivot; t++) 
					left.unionWith(tri_aabbs.bmin[tri_idx_array[t]], tri_aabbs.bmax[tri_idx_array[t]]);		// note that we dont create that stupid intermediate AABB object
				
				double end = omp_get_wtime();
				ctr.tri_filter_time += static_cast<float>(end-start);

				start = omp_get_wtime();
				int newrpivot = filterRaysObject(rays, num_rays, ray_idx_array, left, rpivot, ctr.raybox);
				end = omp_get_wtime();

				ctr.ray_filter_time += static_cast<float>(end-start);

				if(newrpivot == rpivot || newtpivot == tpivot) {
					goto BRUTE_FORCE;
				} else {
					cpuDacrtObjectPartitioning(left, triangles, tri_aabbs, num_triangles, tri_idx_array, newtpivot, rays, num_rays, ray_idx_array, newrpivot, maxts, hitids, ctr);
				}
			}
			{
				double start = omp_get_wtime();
				int newtpivot;
				if(extents.x > extents.y && extents.x > extents.z) {
					float median = (space.bmax.x + space.bmin.x) * 0.5f;
					newtpivot = filterTrianglesObject(tri_aabbs, num_triangles, tri_idx_array, tpivot, median, splitTriangleRightX, ctr.tribox);
				} else if(extents.y > extents.x && extents.y > extents.z) {
					float median = (space.bmax.y + space.bmin.y) * 0.5f;
					newtpivot = filterTrianglesObject(tri_aabbs, num_triangles, tri_idx_array, tpivot, median, splitTriangleRightY, ctr.tribox);
				} else {
					float median = (space.bmax.z + space.bmin.z) * 0.5f;
					newtpivot = filterTrianglesObject(tri_aabbs, num_triangles, tri_idx_array, tpivot, median, splitTriangleRightZ, ctr.tribox);
				}

				AABB right;
				for(int t = 0; t < newtpivot; t++) 
					right.unionWith(tri_aabbs.bmin[tri_idx_array[t]], tri_aabbs.bmax[tri_idx_array[t]]);		// note that we dont create that stupid intermediate AABB object
				double end = omp_get_wtime();
				ctr.tri_filter_time += static_cast<float>(end-start);

				start = omp_get_wtime();
				int newrpivot = filterRaysObject(rays, num_rays, ray_idx_array, right, rpivot, ctr.raybox);
				end = omp_get_wtime();

				ctr.ray_filter_time += static_cast<float>(end-start);

				if(newrpivot == rpivot || newtpivot == tpivot) {
					goto BRUTE_FORCE;
				} else {
					cpuDacrtObjectPartitioning(right, triangles, tri_aabbs, num_triangles, tri_idx_array, newtpivot, rays, num_rays, ray_idx_array, newrpivot, maxts, hitids, ctr);
				}
			}
		}
}

// Shadow method
// We need a custom method for shadows
void cpuDacrtObjShadow(const AABB& space, TriangleArray& triangles, AabbArray& tri_aabbs, int num_triangles, int* tri_idx_array, int tpivot,
	RayArray& shadow_rays, int num_shadow_rays, int* shadow_ray_idx_array, int* pray_associations, int rpivot, bool* shadow, Counters& ctr) {

}