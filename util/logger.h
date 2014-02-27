#ifndef __LOGGER_H__
#define __LOGGER_H__

#pragma once
#include "global.h"

/*
Logger.h:
*********

Logger is a class for 'As the name implies'. We will implement customized logging capabilities for performance evaluation, catching anomalies, etc
*/
class Logger
{
public:
	Logger() {}
	Logger(const char* filename);
	
	void write(const char* str);
	void write(const std::string& str);
	void write(int, int);					// first stats we need as of now.!
	
	// Normally the write pattern would be like "String : val"
	template<typename T>
	void write(const std::string& str, T val) {
		if(!logfile.bad()) logfile<<str<<" "<<val<<"\n";
		else {
			std::cerr<<"Log file Exception :  Writing to default log File";
			logfile.open("default.log");
			logfile<<str<<" "<<val<<"\n";
		}
	}
	
	
	~Logger();
	const char* file;
	std::ofstream logfile;
};



#endif