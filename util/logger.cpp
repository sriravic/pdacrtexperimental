#include "logger.h"

Logger::Logger(const char* file) {
	this->file = file;
	logfile.open(file);
}

void Logger::write(const std::string& str) {
	if(!logfile.bad()) logfile<<str<<"\n";
	else {
		std::cerr<<"Log file Exception :  Writing to default log File";
		logfile.open("default.csv");
		logfile<<str<<"\n";
	}
}

void Logger::write(int val1, int val2) {
	if(!logfile.bad()) logfile<<val1<<","<<val2<<"\n";
	else {
		std::cerr<<"Log file Exception :  Writing to default log File";
		logfile.open("default.csv");
		logfile<<val1<<","<<val2<<"\n";
	}
}

void Logger::write(const char* str) {
	if(!logfile.bad()) logfile<<str<<"\n";
	else {
		std::cerr<<"Log file Exception :  Writing to default log File";
		logfile.open("default.csv");
		logfile<<str<<"\n";
	}
}

Logger::~Logger() { logfile.close(); }

// various other functionalities to be added as required.