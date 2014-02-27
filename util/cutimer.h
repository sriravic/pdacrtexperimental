#ifndef __CUTIMER_H__
#define __CUTIMER_H__

#pragma once
#include <global.h>

// cuda timer
class Timer {
    public:
        Timer(std::string name) : _name(name), time(0) {
            cudaEventCreate(&_start);
            cudaEventCreate(&_stop);
        }

        void start() {
            cudaEventRecord(_start, 0);
        }

        void stop() {
            float temp;
            cudaEventRecord(_stop, 0);
            cudaEventSynchronize(_stop);
            cudaEventElapsedTime(&temp, _start, _stop);
            time += temp;
        }

        void print() const {
            std::cout << "Timer :: Elapsed: " << _name << "\t" << time << " ms" << std::endl;
        }

        float get_ms() const { return time; };

    private:
        cudaEvent_t _start, _stop;
		std::string _name;
        float time;
};


#endif