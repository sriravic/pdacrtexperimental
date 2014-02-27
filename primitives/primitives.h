#ifndef __PRIMITIVES_H__
#define __PRIMITIVES_H__

#pragma once
#include <global.h>

#define EPSILON 0.00000001
#define ZERO_TOLERANCE 0.01f
#define M_PI       3.14159265358979323846f
#define INV_PI     0.31830988618379067154f
#define INV_TWOPI  0.15915494309189533577f
#define INV_FOURPI 0.07957747154594766788f

inline __host__ __device__ float Radians(float deg) {
    return ((float)M_PI/180.f) * deg;
}


inline __host__ __device__ float Degrees(float rad) {
    return (180.f/(float)M_PI) * rad;
}


// custom functors
struct Float3Minimum
{
	__device__ float3 operator() (const float3& A, const float3& B) {
		return make_float3(fminf(A.x, B.x), fminf(A.y, B.y), fminf(A.z, B.z));
	}
};
struct Float3Maximum
{
	__device__ float3 operator() (const float3& A, const float3& B) {
		return make_float3(fmaxf(A.x, B.x), fmaxf(A.y, B.y), fmaxf(A.z, B.z));
	}
};

// structures
struct Sphere
{
	float3 center;
	float radius;
	__device__ __host__ Sphere() {}
	__device__ __host__ Sphere(const float3& c, float r):center(c), radius(r) {}
};

struct Ray
{
	__device__ __host__ Ray();
	__device__ __host__ Ray(const float3& _origin, const float3& _direction):origin(_origin), direction(_direction) {original_id = -1;}
	__device__ __host__ Ray(const float3& _origin, const float3& _direction, int orig_id):origin(_origin), direction(_direction), original_id(orig_id) {}
	__device__ __host__ Ray(const Ray& R) {
		origin = R.origin;
		direction = R.direction;
		original_id = R.original_id;
	}
	__device__ __host__ float3 operator() (float t) const { return origin + t*direction; }
	float3 origin;
	float3 direction;
	int    original_id;
};

struct RayArray
{
	float3* o;
	float3* d;
	int*    original_id_array;
	__device__ __host__ RayArray() { o = d = NULL; original_id_array = NULL;}
	__device__ __host__ RayArray(float3* _o, float3* _d) { o = _o; d = _d; original_id_array = NULL;}
	__device__ __host__ RayArray(float3* _o, float3* _d, int* id) { o = _o; d = _d; original_id_array = id; }
	__device__ __host__ RayArray(const RayArray& R) {
		o = R.o;
		d = R.d;
		original_id_array = R.original_id_array;
	}
};

struct AABB
{
	__device__ __host__ AABB() {bmin = make_float3(FLT_MAX, FLT_MAX, FLT_MAX); bmax = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);}
	__device__ __host__ AABB(const AABB& box):bmin(box.bmin), bmax(box.bmax) {}
	__device__ __host__ AABB(const float3& _min, const float3& _max) { bmin = _min; bmax = _max; }
	__device__ __host__ float surfaceArea() const {
		float l = bmax.x - bmin.x;
		float w = bmax.z - bmin.z;
		float h = bmax.z - bmin.y;
		return 2*(h*w + w*l + h*l);
	}
	
	__device__ __host__ bool rayIntersect(const Ray& r, float& thit) const {
		float3 minTs = (bmin - r.origin) / r.direction;
		float3 maxTs = (bmax - r.origin) / r.direction;

		float nearT = fminf(minTs.x, maxTs.x);
		nearT = fmaxf(nearT, fminf(minTs.y, maxTs.y));
		nearT = fmaxf(nearT, fminf(minTs.z, maxTs.z));

		float farT = fmaxf(minTs.x, maxTs.x);
		farT = fminf(farT, fmaxf(minTs.y, maxTs.y));
		farT = fminf(farT, fmaxf(minTs.z, maxTs.z));
		
		thit = nearT;
		return nearT <= farT && 0 < farT;
	}
	
	__device__ __host__ float3 centroid() const { return (bmax + bmin) * 0.5f; }
	__device__ __host__ AABB unionOf(const AABB& box) const {
		AABB ret;
		ret.bmin = fminf(bmin, box.bmin);
		ret.bmax = fmaxf(bmax, box.bmax);
		return ret;
	}

	__device__ __host__ AABB& unionWith(const AABB& box) {
		bmin = fminf(bmin, box.bmin);
		bmax = fmaxf(bmax, box.bmax);
		return *this;
	}

	__device__ __host__ AABB& unionWith(const float3& _bmin, const float3& _bmax) {
		bmin = fminf(bmin, _bmin);
		bmax = fmaxf(bmax, _bmax);
		return *this;
	}

	float3 bmax, bmin;
};

struct AabbArray
{
	float3* bmin;
	float3* bmax;
	float3* centroid;
	__host__ __device__ AabbArray() { bmin = bmax = centroid = NULL; }
	__host__ __device__ AabbArray(float3* _bmin, float3* _bmax, float3* _centroid) {
		bmin = _bmin; bmax = _bmax; centroid = _centroid;
	}
	__host__ __device__ AabbArray(const AabbArray& A) {
		bmin = A.bmin; bmax = A.bmax; centroid = A.centroid;
	}
};

struct Triangle
{
	__device__ __host__ Triangle() {}
	__device__ __host__ Triangle(const float3& _v0, const float3& _v1, const float3& _v2) {
		v[0] = _v0; v[1] = _v1; v[2] = _v2;
	}
	__device__ __host__ AABB getBounds() const {
			AABB ret;
			ret.bmin = fminf(v[0], fminf(v[1], v[2]));
			ret.bmax = fmaxf(v[0], fmaxf(v[1], v[2]));
			return ret;
	}
	__device__ __host__ float3 getCentroid() const {
		return ((v[0] + v[1] + v[2]) * 1.0f/3);
	}

	float3 v[3];
};

struct Cone
{
	float3 apex;
	float3 dir;
	float  angle;
	__device__ __host__ Cone():angle(0.0f) {}
	__device__ __host__ Cone(const float3& cone_apex, const float3& _cone_dir, float ang):apex(cone_apex), dir(_cone_dir), angle(ang) {}
	__device__ __host__ Cone(const Cone& C):apex(C.apex), dir(C.dir), angle(C.angle) {}
};

struct CameraCone
{
	Cone   cone;
	float2 pixel_start;
	int    cone_id;
	__device__ __host__ CameraCone() { cone_id = 0; }
	__device__ __host__ CameraCone(const Cone& C, const float2& pixel, int id):cone(C), pixel_start(pixel), cone_id(id) {}
	__device__ __host__ CameraCone(const CameraCone& C):cone(C.cone), pixel_start(C.pixel_start), cone_id(C.cone_id) {}
};

struct TriangleArray
{
	float3* v0;
	float3* v1;
	float3* v2;
	__host__ __device__ TriangleArray() { v0 = v1 = v2 = NULL; }
	__host__ __device__ TriangleArray(float3* _v0, float3* _v1, float3* _v2) { v0 = _v0; v1 = _v1; v2 = _v2; }
	__host__ __device__ TriangleArray(const TriangleArray& T) { v0 = T.v0; v1 = T.v1; v2 = T.v2; }
};

struct UnionAabb
{
	__device__ AABB operator() (const AABB& A, const AABB& B) {
		AABB ret;
		ret.bmin = make_float3(fminf(A.bmin.x, B.bmin.x), fminf(A.bmin.y, B.bmin.y), fminf(A.bmin.z, B.bmin.z));
		ret.bmax = make_float3(fmaxf(A.bmax.x, B.bmax.x), fmaxf(A.bmax.y, B.bmax.y), fmaxf(A.bmax.z, B.bmax.z));
		return ret;
	}
};

struct ConeArray
{
	float3* apex_array;
	float3* cone_dir_array;
	float* angle_array;
};

struct CameraConeArray
{
	ConeArray cones;
	float2*   pixel_start_array;
	int*      cone_id_array;
};

struct Edge
{
	float3 edge;
	float3 v[2];
	__device__ __host__ Edge() {}
	__device__ __host__ Edge(const float3& e, const float3& v0, const float3& v1) {
		edge = e;
		v[0] = v0;
		v[1] = v1;
	}
	__device__ __host__ Edge(const Edge& E) {
		edge = E.edge;
		v[0] = E.v[0];
		v[1] = E.v[1];
	}
};


inline __device__ __host__ static float3 computeNormal(const Triangle& T) {
	return normalize(cross(T.v[0]-T.v[1], T.v[2]-T.v[0]));
}

template<typename T>
__device__ __host__ bool rayIntersect(const Triangle& t, const Ray& r, T& u, T& v, T& xt) {
	float3 edge1, edge2;
	float3 pvec, qvec, tvec;
	T det, inv_det;
	edge1 = t.v[1] - t.v[0];
	edge2 = t.v[2] - t.v[0];
	pvec = cross(r.direction, edge2);
	det = dot(edge1, pvec);
	if(det < EPSILON && det > -EPSILON) return false;
	inv_det = ((T)1.0)/det;
	tvec = r.origin - t.v[0];
	u = dot(tvec, pvec) * inv_det;
	if(u < (T)0.0 || u > (T)1.0) return false;
	qvec = cross(tvec, edge1);
	v = dot(r.direction, qvec) * inv_det;
	if(v < (T)0.0 || u+v > (T)1.0) return false;
	xt = dot(edge2, qvec) * inv_det;
	return true;
}


inline __device__ __host__ int splitSpatialMedian(const AABB& parent, AABB& left, AABB& right, float& split_pos) {
	float3 extents = parent.bmax - parent.bmin;
	left = right = parent;
	if(extents.x > extents.y && extents.x > extents.z) {
		left.bmax.x = right.bmin.x = split_pos = (parent.bmax.x + parent.bmin.x) * 0.5f;
		return 0;
	} else if(extents.y > extents.x && extents.y > extents.z) {
		left.bmax.y = right.bmin.y = split_pos = (parent.bmax.y + parent.bmin.y) * 0.5f;
		return 1;
	} else {
		left.bmax.z = right.bmin.z = split_pos = (parent.bmax.z + parent.bmin.z) * 0.5f;
		return 2;
	}
}

#endif