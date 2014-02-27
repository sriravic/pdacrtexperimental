#ifndef __IO_H__
#define __IO_H__

#pragma once
#include <global.h>
#include "jpeg.h"

#define uint unsigned int
typedef float3 color;

template <class T>
inline T Clamp01(T v) { 
	return v < T(0) ? T(0) : v > T(1) ? T(1) : v;
}

template <class T>
T Lerp(const T& from, const T& to, const float t) {
	return (1.0f - t) * from + t * to;
}

inline int ToByte(float v) {
	return int(std::pow((float)Clamp01(v), (float)1.0f/2.2f) * 255 + .5);
}

void writeToJpeg(int WIDTH, int HEIGHT, color* buffer, const char* filename) {
	// convert the color buffer to color4 buffer for jpeg use
	std::cout<<"\nConverting color format\n";
	unsigned char *output = new unsigned char[WIDTH * HEIGHT * 4];
	char r, g, b, x = 0;
	uint index = 0;
	for(uint i = 0; i < WIDTH * HEIGHT; i++) {
		r = ToByte(buffer[i].x);
		g = ToByte(buffer[i].y);
		b = ToByte(buffer[i].z);
		output[index++] = r;
		output[index++] = g;
		output[index++] = b;
		output[index++] = x;
	}
	std::cout<<"\nDone Converting. Writing to file\n";
	jo_write_jpg(filename, output, WIDTH, HEIGHT, 90);
	SAFE_RELEASE(output);
}


#endif