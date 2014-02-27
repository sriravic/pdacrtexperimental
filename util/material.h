#ifndef __MATERIAL_H__
#define __MATERIAL_H__

#include <global.h>

enum Mat_Type { DIFFUSE, MIRROR, GLASS };

struct Material
{
	float3 ka;
	float3 kd;
	float3 ks;
	float Ns;
	float refractive_index;
	Mat_Type mat;
};

#endif