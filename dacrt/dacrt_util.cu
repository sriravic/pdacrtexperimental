#include <dacrt/dacrt_util.h>

ParallelPackModified::ParallelPackModified() {}
ParallelPackModified::ParallelPackModified(DacrtRunTimeParameters& rtparams, int num_rays) {
	// init all the things with the new rtparams values
	htri_segment_sizes.resize(rtparams.MAX_SEGMENTS);
	hray_segment_sizes.resize(rtparams.MAX_SEGMENTS);
	hsegment_ids.resize(rtparams.MAX_SEGMENTS);
	blockNos.resize(rtparams.BUFFER_SIZE);
	blockStart.resize(rtparams.BUFFER_SIZE);

	buffered_ray_idx.resize(rtparams.BUFFER_SIZE);
	buffered_tri_idx.resize(rtparams.BUFFER_SIZE);
	segment_ids.resize(rtparams.MAX_SEGMENTS);
	ray_segment_sizes.resize(rtparams.MAX_SEGMENTS);
	tri_segment_sizes.resize(rtparams.MAX_SEGMENTS);
	buffered_ray_maxts.resize(rtparams.BUFFER_SIZE, FLT_MAX);
	buffered_ray_hitids.resize(rtparams.BUFFER_SIZE, -1);
	dev_ray_maxts.resize(num_rays, FLT_MAX);
	dev_hitids.resize(num_rays, -1);

	blockCnt			= 0;
	bstart				= 0;
	ray_buffer_occupied = 0;
	tri_buffer_occupied = 0;
	num_segments		= 0;
}

void ParallelPackModified::clearHostData() {
	htri_segment_sizes.clear(); 
	hray_segment_sizes.clear(); 
	hsegment_ids.clear();
	blockNos.clear(); 
	blockStart.clear();
}
	
// reset the block variables and the corresponding device data
void ParallelPackModified::clearBlockData() {
	blockCnt = 0;			// reset this value
	bstart = 0;
	dblockStart.clear();
	dblockNos.clear();
}
	
// clears all data related to device memory and the counter values also.
void ParallelPackModified::clearDeviceData() {
	buffered_ray_idx.clear();
	buffered_tri_idx.clear();
	tri_segment_sizes.clear();
	ray_segment_sizes.clear();
	segment_ids.clear();
	ray_buffer_occupied = 0;
	tri_buffer_occupied = 0;
	num_segments = 0;
}

DoubleBuffer::DoubleBuffer(DacrtRunTimeParameters& rtparams, int num_rays) {
	buffer[0] = ParallelPackModified(rtparams, num_rays);
	buffer[1] = ParallelPackModified(rtparams, num_rays);
	to_use_buffer = 0;				// default value;
	flags[0] = true;				// both are available
	flags[1] = true;
}

// this method returns the correct buffer regarding which one is free
ParallelPackModified& DoubleBuffer::getBuffer() {
	if(flags[0]) {
		flags[0] = false;			// set as busy
		return buffer[0];
	} else {
		flags[1] = false;
		return buffer[1];
	}
}

void DoubleBuffer::resetBuffer(int buffer_num) {
	if(flags[buffer_num] == false) {
		flags[buffer_num] = true;
	} else {
		std::cerr<<"SEVERE ERROR.! Invalid reset option.!!\nExiting\n";
		exit(-1);
	}
}
	