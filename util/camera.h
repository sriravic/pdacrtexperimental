#ifndef __CAMERA_H__
#define __CAMERA_H__

#pragma once
#include <global.h>
#include <primitives/primitives.h>
#include <primitives/primitives_aos.h>

typedef float4 Pixel;

struct Film
{
	unsigned int width, height;
	Pixel* image;
	Film():width(0), height(0), image(NULL) {}
	Film(unsigned int w, unsigned int h):width(w), height(h), image(NULL) {}
};

struct Camera
{
	float3 eye;
	float3 up;
	float3 lookat;
	float fov;

	Camera(){}
	Camera(const float3& e, const float3& l, const float3& u, float f = 0.8f):eye(e), lookat(l), up(u), fov(f) {}
	
	friend std::ostream& operator<< (std::ostream& out, const Camera& c) {
		out<<"Camera : {\n\tEye: ["<<c.eye.x<<","<<c.eye.y<<","<<c.eye.z<<"]\n\tLook : ["<<c.lookat.x<<","<<c.lookat.y<<","<<c.lookat.z<<"]\n\tUp : ["
			<<c.up.x<<","<<c.up.y<<","<<c.up.z<<"]\n\tFOV : "<<c.fov<<"\n";
		return out;
	}
};

class EyeFrustum
{
public:

	__device__ __host__ EyeFrustum() {}
	__device__ __host__ EyeFrustum(const Camera& c, const Film& f):width(f.width), height(f.height), eye(c.eye) {
		float rcpH = 1.0f/height;
		float alpha2 = c.fov * 0.5f;
		float3 D = c.lookat - eye;
		float dist = sqrtf(dot(D, D));
		float a = tanf(alpha2) * dist;
		float b = a;
		a*= rcpH * width;
		D /= dist;
		across = normalize(cross(D, c.up));
		up = cross(across, D);
		//up = -1 * up;
		BL = eye + dist * D - a * across - b * up;
		across *= 2.0f * a / width;
        up *=   2.0f * b * rcpH;
	}

	// generate a non unit ray
	__host__ __device__ float3 genRayDir(unsigned int x, unsigned int y) const {
		float3 filmPt = BL + (static_cast<float>(y) * up) + (static_cast<float>(x) * across);
		return filmPt - eye;
	}

	unsigned int width, height;
	float3 eye, BL, across, up;
};

static inline float3 getEye(const AABB& box, float scale = 5.0f) {
	float minwidth;
	float3 centroid = box.centroid();
	float3 axis;
	
	float3 extents = box.bmax - box.bmin;
	if(extents.x < extents.y && extents.x < extents.z) { minwidth = extents.x; axis = make_float3(1.0, 0.0, 0.0); }
	else if(extents.y < extents.x && extents.y < extents.z) { minwidth = extents.y; axis = make_float3(0.0, 1.0, 0.0); }
	else { minwidth = extents.z; axis = make_float3(0.0, 0.0, 1.0);}
	
	//centroid moved by vector of 2*minwidth*axis
	return centroid - axis*(scale*minwidth);
}

#endif