#include <dacrt/kdtree.h>

KdTree::KdTree(const AABB& _root_aabb, TriangleArray& _triangles) {
	root = _root_aabb;
	triangles = _triangles;
	max_depth = 0;
}


