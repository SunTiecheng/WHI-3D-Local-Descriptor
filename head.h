#pragma once
#include <string>
using namespace std;
struct Parameter
{
	string source;
	string target;
	double leaf;
	int DimBasic;
	double SearchRadius;
	double MeshResolution;
	double threshold_ransac;
};