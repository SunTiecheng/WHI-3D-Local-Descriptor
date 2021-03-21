#pragma once
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/correspondence.h>
#include <pcl/point_representation.h>
#include <pcl/PointIndices.h>
#include <vector>
#include "opencv2/core/core.hpp"

using namespace std;

struct WHIFeature
{
	std::vector<double> feature;
};


class WHIFeatureEstimation
{
public:
	void setInputCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_in);
	void setIndices(const pcl::PointIndicesConstPtr& indices);
	void setDimBasic(int k);
	void setSearchRadius(double radius);
	void setmr(double mr);
	void compute(pcl::PointCloud<WHIFeature>::Ptr& feature);
	WHIFeatureEstimation();
	~WHIFeatureEstimation();
private:
	void LRFEstimator(const int& current_point_idx, Eigen::Matrix3f &rf, std::vector<int>& n_indices, std::vector<float>& n_sqr_distances);
	void init();
	cv::Mat imagedft(cv::Mat& input);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in;
	pcl::PointIndices indices_in;
	pcl::KdTreeFLANN<pcl::PointXYZ> tree;
	int dimBasic;
	double searchRadius;
	double meshresolution;
	bool indices_flag = false;
};


class CorrespondenceRANSAC
{
public:
	void setInputSource(pcl::PointCloud<pcl::PointXYZ>::Ptr& source);
	void setInputTarget(pcl::PointCloud<pcl::PointXYZ>::Ptr& target);
	void setInputCorrespondences(boost::shared_ptr<pcl::Correspondences>& corr);
	void setIterationTimes(int iter);
	void setThreshold(double thre);
	void runRANSAC();
	pcl::PointIndices getInliersIndices();
	pcl::Correspondences getRemainCorrespondences();
	CorrespondenceRANSAC();
	~CorrespondenceRANSAC();
private:
	pcl::PointCloud<pcl::PointXYZ>::Ptr source_in;
	pcl::PointCloud<pcl::PointXYZ>::Ptr target_in;
	boost::shared_ptr<pcl::Correspondences> corr_in;
	pcl::PointIndices inliers_indices;
	double threshold = 0.3;
	int iterationtimes = 500;
};
