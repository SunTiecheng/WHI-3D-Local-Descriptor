#include "head.h"
#include "WHI.h"
#include <iostream>
#include <fstream>
#include <pcl/io/ply_io.h>
#include <pcl/io/io.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/icp.h>
#include <pcl/kdtree/flann.h>
#include <opencv2/opencv.hpp>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>


using namespace std;


Parameter parse_parameter()
{
	Parameter out;
	ifstream parafile;
	parafile.open("data/parameters.txt");
	if (parafile.fail())
	{
		cout << "can't open the parameters.txt file" << endl;
		//exit(0);
	}
	char str[256];
	while (!parafile.eof())
	{
		parafile.getline(str, 256, '#');
		if (strcmp(str, "source") == 0)
		{
			parafile.getline(str, 100, '\n');
			out.source = str;
		}
		if (strcmp(str, "target") == 0)
		{
			parafile.getline(str, 100, '\n');
			out.target = str;
		}
		else if (strcmp(str, "leaf") == 0)
		{
			parafile.getline(str, 255, '\n');
			out.leaf = atof(str);
		}
		else if (strcmp(str, "ThresholdRANSAC") == 0)
		{
			parafile.getline(str, 255, '\n');
			out.threshold_ransac = atof(str);
		}
		else if (strcmp(str, "DimBasic") == 0)
		{
			parafile.getline(str, 255, '\n');
			out.DimBasic = atoi(str);
		}
		else if (strcmp(str, "SearchRadius") == 0)
		{
			parafile.getline(str, 255, '\n');
			out.SearchRadius = atof(str);
		}
	}
	parafile.close();
	cout << "The input parameters list:" << endl;
	cout << "1. source:" << out.source << endl;
	cout << "2. target:" << out.target << endl;
	cout << "3. leaf:" << out.leaf << endl;
	cout << "4. ThresholdGroundTruth:" << out.threshold_ransac << endl;
	cout << "5. SearchRadius:" << out.SearchRadius << endl;
	cout << "6. DimBasic:" << out.DimBasic << endl;
	cout << endl;
	return out;
}

double estresolution(pcl::PointCloud<pcl::PointXYZ>::Ptr& incloud)
{
	double resolution = 0;
	int count = 0;
	int nres;
	vector<int> indices(2);
	vector<float> distances(2);
	pcl::search::KdTree<pcl::PointXYZ> tree;
	tree.setInputCloud(incloud);
	for (int j = 0; j < incloud->size(); j++)
	{
		nres = tree.nearestKSearch(incloud->at(j), 2, indices, distances);
		if (nres == 2)
		{
			if (count > 10 && distances[1] < 5 * resolution)
			{
				resolution = resolution * count;
				resolution += sqrt(distances[1]);
				count++;
				resolution = resolution / count;
			}
			if (count <= 10)
			{
				resolution = resolution * count;
				resolution += sqrt(distances[1]);
				count++;
				resolution = resolution / count;
			}
		}
	}
	return resolution;
}

void downfilter(pcl::PointCloud<pcl::PointXYZ>::Ptr & in, pcl::PointCloud<pcl::PointXYZ>::Ptr & out, double leafsize)
{
	pcl::VoxelGrid<pcl::PointXYZ> down_filter;
	down_filter.setInputCloud(in);
	down_filter.setLeafSize(leafsize, leafsize, leafsize);
	down_filter.filter(*out);

}

void feature_points(pcl::PointIndices::Ptr & model_indices, pcl::PointCloud<pcl::PointXYZ>::Ptr & modelsdown, pcl::PointCloud<pcl::PointXYZ>::Ptr& model)
{
	pcl::KdTreeFLANN<pcl::PointXYZ> tree;
	model_indices->indices.clear();
	tree.setInputCloud(model);
	for (int j = 0; j < modelsdown->size(); j++)
	{
		vector<int> Kind;
		vector<float> Kdist;
		pcl::PointXYZ QueryPoint = modelsdown->at(j);
		tree.nearestKSearch(QueryPoint, 1, Kind, Kdist);
		model_indices->indices.push_back(Kind[0]);

	}

}

void compute_feature_WHI(pcl::PointCloud<WHIFeature>::Ptr& Feature, pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud, pcl::PointIndices::Ptr& indices,
	int dimbasic, double searchradius, double mr)
{
	WHIFeatureEstimation est_Feature;
	est_Feature.setIndices(indices);
	est_Feature.setInputCloud(input_cloud);
	est_Feature.setDimBasic(dimbasic);
	est_Feature.setSearchRadius(searchradius);
	est_Feature.setmr(mr);
	est_Feature.compute(Feature);
}
boost::shared_ptr<pcl::Correspondences> Mul_Corr(flann::Matrix<double>& queries, flann::Matrix<double>& target, pcl::PointIndices::Ptr& source_indices, pcl::PointIndices::Ptr& target_indices)
{
	boost::shared_ptr<pcl::Correspondences> corr(new pcl::Correspondences);
	int nn = 2;

	flann::Matrix<int> indices(new int[source_indices->indices.size()*nn], source_indices->indices.size(), nn);
	flann::Matrix<double> dists(new double[source_indices->indices.size()*nn], source_indices->indices.size(), nn);
	flann::KDTreeIndex<flann::L2<double>> index(target, flann::KDTreeIndexParams(8));
	index.buildIndex();
	index.knnSearch(queries, indices, dists, nn, flann::SearchParams(32));
	//index.knnSearch(scene, indices, dists, nn, flann::SearchParams(flann::FLANN_CHECKS_UNLIMITED));
	for (int i = 0; i < source_indices->indices.size(); i++)
	{
		if (indices[i][0] > 0)
		{
			pcl::Correspondence cor;
			cor.index_query = source_indices->indices.at(i);
			cor.index_match = indices[i][0];
			cor.distance = dists[i][0];
			corr->push_back(cor);
		}
	}
	return corr;
}


void registration(pcl::PointIndices::Ptr& source_indices, pcl::PointCloud<pcl::PointXYZ>::Ptr& source, 
	pcl::PointIndices::Ptr& target_indices, pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& soudown, pcl::PointCloud<pcl::PointXYZ>::Ptr& tardown,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& s, Parameter& para)
{

	//pcl::CorrespondencesPtr corr_WHI;
	pcl::PointCloud<WHIFeature>::Ptr fsource(new pcl::PointCloud<WHIFeature>);
	pcl::PointCloud<WHIFeature>::Ptr ftarget(new pcl::PointCloud<WHIFeature>);

	compute_feature_WHI(fsource, source, source_indices, para.DimBasic, para.SearchRadius, para.MeshResolution);
	compute_feature_WHI(ftarget, target, target_indices, para.DimBasic, para.SearchRadius, para.MeshResolution);
	int dimfeature = ftarget->at(0).feature.size();
	flann::Matrix<double> targetf(new double[ftarget->size()*dimfeature], ftarget->size(), dimfeature);

	for (int i = 0; i < ftarget->size(); i++)
	{
		for (int j = 0; j < dimfeature; j++)
		{
			targetf[i][j] = ftarget->at(i).feature[j];
		}
	}

	flann::Matrix<double> queries(new double[fsource->size()*dimfeature], fsource->size(), dimfeature);

	for (int i = 0; i < fsource->size(); i++)
	{
		for (int j = 0; j < dimfeature; j++)
		{
			queries[i][j] = fsource->at(i).feature[j];
		}
	}
	boost::shared_ptr<pcl::Correspondences> corr;
	corr = Mul_Corr(queries, targetf, source_indices, target_indices);

	pcl::Correspondences ransaccorr;
	CorrespondenceRANSAC corrRansac;
	corrRansac.setInputSource(source);
	corrRansac.setInputTarget(target);
	corrRansac.setInputCorrespondences(corr);
	corrRansac.setIterationTimes(10000);
	corrRansac.setThreshold(para.threshold_ransac);
	corrRansac.runRANSAC();
	ransaccorr = corrRansac.getRemainCorrespondences();
	pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> trans_est_svd;
	Eigen::Matrix4f T_SVD;
	trans_est_svd.estimateRigidTransformation(*source, *target, ransaccorr, T_SVD);
	pcl::PointCloud<pcl::PointXYZ>::Ptr source_tran(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr source_(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*s, *source_, T_SVD);
	pcl::transformPointCloud(*soudown, *source_tran, T_SVD);
	pcl::io::savePLYFile("data/coarse.ply", *source_);
	cout << "coarse finished." << endl;

	////ICP
	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree1(new pcl::search::KdTree<pcl::PointXYZ>);
	tree1->setInputCloud(source_tran);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree2(new pcl::search::KdTree<pcl::PointXYZ>);
	tree2->setInputCloud(tardown);
	icp.setSearchMethodSource(tree1);
	icp.setSearchMethodTarget(tree2);
	icp.setInputSource(source_tran);
	icp.setInputTarget(tardown);
	icp.setTransformationEpsilon(1e-12);
	//icp.setEuclideanFitnessEpsilon(0.04);
	icp.setMaxCorrespondenceDistance(para.MeshResolution*30);
	icp.setRANSACOutlierRejectionThreshold(para.MeshResolution*1000);
	icp.setMaximumIterations(5000);
	pcl::PointCloud<pcl::PointXYZ> Final;
	icp.align(Final);
	Eigen::Matrix4f T_SVD2;
	T_SVD2 = icp.getFinalTransformation();
	pcl::transformPointCloud(*source_, *source_tran, T_SVD2);
	pcl::io::savePLYFile("data/fine.ply", *source_tran);
	cout << "fined registration" << endl;
}

void regis()
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr source(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr target(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr soudown(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr tardown(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::io::loadPLYFile("data/input2.ply", *target);
	pcl::io::loadPLYFile("data/nfp.ply", *source);
	double res = estresolution(target);
	Parameter para = parse_parameter();
	para.MeshResolution = para.leaf * res;
	downfilter(source, soudown, para.leaf * res);
	downfilter(target, tardown, para.leaf * res);


	pcl::PointIndices::Ptr source_indices(new pcl::PointIndices);
	pcl::PointIndices::Ptr target_indices(new pcl::PointIndices);
	pcl::PointIndices::Ptr soudown_indices(new pcl::PointIndices);
	pcl::PointIndices::Ptr tardown_indices(new pcl::PointIndices);
	feature_points(source_indices, soudown, source);
	feature_points(target_indices, tardown, target);
	for (int i = 0; i < soudown->size(); i++)
	{
		soudown_indices->indices.push_back(i);
	}
	for (int i = 0; i < tardown->size(); i++)
	{
		tardown_indices->indices.push_back(i);
	}

	registration(soudown_indices, soudown, tardown_indices, tardown, soudown, tardown, source, para);

}

int main()
{

	regis();

	return 0;
}

