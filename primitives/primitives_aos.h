#ifndef __PRIMITIVES_SOA_H__
#define __PRIMITIVES_SOA_H__

/**
This represents a paradigm shift in the way the whole system works. We move on to a AoS array assuming it can give good memory access patterns.

We can use basic primitives from the primitives.h file, but all the ray array, triangle array and other arrays have to be totally different now.
*/

/**
NOTE: All the new data structures defined here are appended with either 3/4 to denote the major float structure used within them. So Ray uses float3 only
and so its appended with 3. AABB has float4 and so its appended with 4
*/

#include <global.h>

#define EPSILON 0.00000001
#define M_PI       3.14159265358979323846f
#define INV_PI     0.31830988618379067154f
#define INV_TWOPI  0.15915494309189533577f
#define INV_FOURPI 0.07957747154594766788f


struct Float4Minimum
{
	__device__ float4 operator() (const float4& A, const float4& B) {
		return make_float4(fminf(A.x, B.x), fminf(A.y, B.y), fminf(A.z, B.z), fminf(A.w, B.w));
	}
};
struct Float4Maximum
{
	__device__ float4 operator() (const float4& A, const float4& B) {
		return make_float4(fmaxf(A.x, B.x), fmaxf(A.y, B.y), fmaxf(A.z, B.z), fmaxf(A.w, B.w));
	}
};


struct Ray4
{
	__device__ __host__ Ray4() {}
	__device__ __host__ Ray4(const float4& ori, const float4& dir) {
		data.origin = ori;
		data.direction = dir;
	}
	__device__ __host__ float4 operator() (float t) { return data.origin + t * data.direction; }
	union {
		struct {
			float4 origin;
			float4 direction;
		};
		char d[32];		// 32 byte data
	}data;
};

// we are storing the inv_direction.x and inv_direction.y. Calculate only inv_direction.z when required also.
struct Ray3
{
	__device__ __host__ Ray3() {}
	__device__ __host__ Ray3(const float3& _origin, const float3& _direction) {
		data.origin = _origin; data.direction = _direction;
		data.inv_dx = 1.0f/_direction.x;
		data.inv_dy = 1.0f/_direction.y;
	}
	__device__ __host__ Ray3(const Ray3& R) {
		data.origin = R.data.origin; data.direction = R.data.direction;
		data.inv_dx = R.data.inv_dx; data.inv_dy = R.data.inv_dy;
	}
	__device__ __host__ float3 operator() (float t) const { return data.origin + t * data.direction; }
	
	union {
		struct {
			float3 origin;
			float3 direction;
			float inv_dx;
			float inv_dy;
		};
		char d[32];

	}data;
};

struct AABB4
{
	__device__ __host__ AABB4() {
		data.bmin = make_float4(FLT_MAX);
		data.bmax = make_float4(-FLT_MAX);
	}
	__device__ __host__ AABB4(const float4& _minima, const float4& _maxima) {
		data.bmin = _minima;
		data.bmax = _maxima;
	}
	__device__ __host__ AABB4(const float3& _minima, const float3& _maxima) {
		data.bmin = make_float4(_minima, 0.0f);
		data.bmax = make_float4(_maxima, 0.0f);
	}
	__device__ __host__ AABB4(const AABB4& B) {
		data.bmin = B.data.bmin;
		data.bmax = B.data.bmax;
	}
	__device__ __host__ float surfaceArea() const {
		float l = data.bmax.x - data.bmin.x;
		float w = data.bmax.z - data.bmin.z;
		float h = data.bmax.z - data.bmin.y;
		return 2*(h*w + w*l + h*l);
	}
	/*
	__device__ __host__ bool rayIntersect(const Ray3& r, float& thit) const {
		
		// use precomputed values
		float inv_dz = 1.0f/r.data.direction.z;
		float3 minTs = (make_float3(data.bmin.x, data.bmin.y, data.bmin.z) - r.origin) * make_float3(r.data.inv_dx, r.data.inv_dy, inv_dz);
		float3 maxTs = (make_float3(data.bmax.x, data.bmax.y, data.bmax.z) - r.origin) * r.inv_d;

		float nearT = fminf(minTs.x, maxTs.x);
		nearT = fmaxf(nearT, fminf(minTs.y, maxTs.y));
		nearT = fmaxf(nearT, fminf(minTs.z, maxTs.z));

		float farT = fmaxf(minTs.x, maxTs.x);
		farT = fminf(farT, fmaxf(minTs.y, maxTs.y));
		farT = fminf(farT, fmaxf(minTs.z, maxTs.z));
		
		thit = nearT;
		return nearT <= farT && 0 < farT;
	}
	*/
	
	// we can change this whenever we want if we want performance updates
	/// NOTE:!!!
	__device__ __host__ bool rayIntersect(const Ray4& r, float& thit) const {
		float3 inv_d = 1.0f/make_float3(r.data.direction.x, r.data.direction.y, r.data.direction.z);
		float3 minTs = (make_float3(data.bmin.x, data.bmin.y, data.bmin.z) - make_float3(r.data.origin.x, r.data.origin.y, r.data.origin.z)) * inv_d;
		float3 maxTs = (make_float3(data.bmax.x, data.bmax.y, data.bmax.z) - make_float3(r.data.origin.x, r.data.origin.y, r.data.origin.z)) * inv_d;

		float nearT = fminf(minTs.x, maxTs.x);
		nearT = fmaxf(nearT, fminf(minTs.y, maxTs.y));
		nearT = fmaxf(nearT, fminf(minTs.z, maxTs.z));

		float farT = fmaxf(minTs.x, maxTs.x);
		farT = fminf(farT, fmaxf(minTs.y, maxTs.y));
		farT = fminf(farT, fmaxf(minTs.z, maxTs.z));
		
		thit = nearT;
		return nearT <= farT && 0 < farT;

	}
	__device__ __host__ float3 centroid3() const { return (make_float3(data.bmax.x, data.bmax.y, data.bmax.z) + make_float3(data.bmin.x, data.bmin.y, data.bmin.z)) * 0.5f; }
	__device__ __host__ float4 centroid4() const { return (data.bmax + data.bmin) * 0.5f; }
	__device__ __host__ AABB4 unionOf(const AABB4& box) const {
		AABB4 ret;
		ret.data.bmin = fminf(data.bmin, box.data.bmin);
		ret.data.bmax = fmaxf(data.bmax, box.data.bmax);
		return ret;
	}
	__device__ __host__ AABB4& unionWith(const AABB4& box) {
		data.bmin = fminf(data.bmin, box.data.bmin);
		data.bmax = fmaxf(data.bmax, box.data.bmax);
		return *this;
	}
	__device__ __host__ AABB4& unionWith(const float4& _bmin, const float4& _bmax) {
		data.bmin = fminf(data.bmin, _bmin);
		data.bmax = fmaxf(data.bmax, _bmax);
		return *this;
	}

	union {
		struct {
			float4 bmin;
			float4 bmax;
		};
		char d[32];
	}data;
};

// Keeping the size of one triangle to 64 bytes. For coalasced accesses
struct Triangle3
{
	__device__ __host__ Triangle3() {}
	__device__ __host__ Triangle3(const float3& v0, const float3& v1, const float3& v2) {
		data.v[0] = v0; 
		data.v[1] = v1; 
		data.v[2] = v2;
		data.e[0] = v1 - v0;
		data.e[1] = v2 - v0;
	}
	__device__ __host__ Triangle3(const Triangle3& T) {
		data.v[0] = T.data.v[0]; 
		data.v[1] = T.data.v[1]; 
		data.v[2] = T.data.v[2];
		data.e[0] = T.data.e[0]; 
		data.e[1] = T.data.e[1];
	}
	
	__device__ __host__ AABB4 getBounds() const {
			AABB4 ret;
			ret.data.bmin = make_float4(fminf(data.v[0], fminf(data.v[1], data.v[2])), 0.0f);
			ret.data.bmax = make_float4(fmaxf(data.v[0], fmaxf(data.v[1], data.v[2])), 0.0f);
			return ret;
	}
	
	__device__ __host__ float3 getCentroid() const {
		return ((data.v[0] + data.v[1] + data.v[2]) * 1.0f/3);
	}
	
	union {
		struct {
			float3 v[3];		// 36 bytes
			float3 e[2];		// 24 bytes (total 60 bytes)
			float  pad;			// 4 bytes = 64 bytes
		};
		short d[32];
	} data;
};

inline __device__ __host__ static float3 computeNormal(const Triangle3& T) {
	return normalize(cross(T.data.e[0], T.data.e[1]));
}

///NOTE: RayArrayAos uses the memory friendly 32 byte ray format.!
struct RayArrayAos
{
	Ray4* rays;
	RayArrayAos() {rays = NULL; }
	RayArrayAos(Ray4* r) { rays = r; }
	RayArrayAos(const RayArrayAos& R) { rays = R.rays; }
};

struct TriangleArrayAos
{
	Triangle3* triangles;
	TriangleArrayAos() { triangles = NULL; }
	TriangleArrayAos(Triangle3* t) { triangles = t; }
	TriangleArrayAos(const TriangleArrayAos& T) { triangles = T.triangles; }
};

struct AabbArrayAos
{
	AABB4* bboxes;
	AabbArrayAos() { bboxes = NULL; }
	AabbArrayAos(AABB4* b) { bboxes = b; }
	AabbArrayAos(const AabbArrayAos& A) { bboxes = A.bboxes; }
};

// this version uses the new primitive types
template<typename T>
__device__ __host__ bool rayIntersectAos(const Triangle3& t, const Ray4& r, T& u, T& v, T& xt) {
	float3 pvec, qvec, tvec;
	T det, inv_det;
	pvec = cross(make_float3(r.data.direction.x, r.data.direction.y, r.data.direction.z), t.data.e[1]);
	det = dot(t.data.e[0], pvec);
	if(det < EPSILON && det > -EPSILON) return false;
	inv_det = ((T)1.0)/det;
	tvec = make_float3(r.data.origin.x, r.data.origin.y, r.data.origin.z) - t.data.v[0];
	u = dot(tvec, pvec) * inv_det;
	if(u < (T)0.0 || u > (T)1.0) return false;
	qvec = cross(tvec, t.data.e[0]);
	v = dot(make_float3(r.data.direction.x, r.data.direction.y, r.data.direction.z), qvec) * inv_det;
	if(v < (T)0.0 || u+v > (T)1.0) return false;
	xt = dot(t.data.e[1], qvec) * inv_det;
	return true;
}



#endif