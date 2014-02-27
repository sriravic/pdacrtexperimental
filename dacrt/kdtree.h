#ifndef __KDTREE_H__
#define __KDTREE_H__

#include <global.h>
#include <primitives/primitives.h>

#define STACK_SIZE 64
#define USE_HOST

class TreeStack {
    public:
        __device__
        TreeStack() : ptr(0) {};

        __inline__ __device__
        void push(unsigned int nodeidx, float min, float max) {
            p_min[ptr] = min;
            p_max[ptr] = max;
            node[ptr] = nodeidx;
            ptr++;
        }

        __inline__ __device__
        bool pop(unsigned int & nodeidx, float & min, float & max) {
            if(ptr > 0) {
                ptr--;
                min = p_min[ptr];
                max = p_max[ptr];
                nodeidx = node[ptr];
                return true;
            }
            return false;
        }
        __inline__ __device__
        bool empty() { return (ptr == 0); }
    private:
        int ptr;
        unsigned int node[STACK_SIZE];
        float p_min[STACK_SIZE], p_max[STACK_SIZE];
};

class KdTree
{
public:
	KdTree();
	KdTree(const AABB& _root_aabb, TriangleArray& _triangles);
	
	void create();
	void preorder();
	int maxDepth() { return max_depth; }
	void rayTraverse(RayArray& rays, int* hitids, float* maxts);

	AABB root;
	TriangleArray triangles;
	int max_depth;

#ifdef USE_HOST
	std::vector<AABB>         node_aabbs;			// what is the node aabb
	std::vector<unsigned int> num_children;		// indicates how many children this node has
	std::vector<unsigned int> num_elements;		// indicates how many elements are present in the node if its a leaf
	std::vector<unsigned int> element_start;		// indicates where the id for triangles start for this leaf node
	std::vector<int>          element_ids;		// stores all the triangle ids for the leaf node
	std::vector<unsigned int> child_idx;			// indicates what is the index of the child of this node 
	std::vector<bool>         leaf;				// indicates if its a leaf or child
	
	std::vector<unsigned int> split_dirs;			// indicates split dirs for the node
	std::vector<float>        split_pos;			// split position is also recorded
	
#elif USE_DEVICE
	thrust::host_vector<AABB>         node_aabbs;			// what is the node aabb
	thrust::device_vector<unsigned int> num_children;		// indicates how many children this node has
	thrust::device_vector<unsigned int> num_elements;		// indicates how many elements are present in the node if its a leaf
	thrust::device_vector<unsigned int> element_start;		// indicates where the id for triangles start for this leaf node
	thrust::device_vector<int>          element_ids;		// stores all the triangle ids for the leaf node
	thrust::device_vector<unsigned int> child_idx;			// indicates what is the index of the child of this node 
	thrust::device_vector<bool>         leaf;				// indicates if its a leaf or child
	
	thrust::device_vector<unsigned int> split_dirs;			// indicates split dirs for the node
	thrust::device_vector<float>        split_pos;			// split position is also recorded
#endif


private:
	// no copies and assignments
	KdTree(const KdTree& K);
	KdTree& operator= (const KdTree& K);
};





#endif