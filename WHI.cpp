#include "WHI.h"
#include <pcl/filters/extract_indices.h>
#include <pcl/common/centroid.h>
#include <pcl/common/distances.h>

#include <pcl/features/normal_3d.h>
#include <pcl/point_types.h>
//#include <pcl/kdtree/impl/kdtree_flann.hpp> 
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/transformation_estimation_svd.h>
//#include <pcl/common/impl/io.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/io.h>
//#include "opencv2/core/core.hpp"
//#include<opencv2/opencv.hpp>

//#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui/highgui.hpp"
//#include <opencv2/highgui/highgui_c.h>
//#include <pcl/registration/transforms.h>

using namespace cv;




/* *************************for class ThreePointsFeatureEstimation************************** */
void WHIFeatureEstimation::setInputCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& in)
{
	cloud_in = in;
}
//void ThreePointsFeatureEstimation::setInputNormal(pcl::PointCloud<pcl::Normal>::Ptr& in)
//{
//	normal_in = in;
//}
void WHIFeatureEstimation::setIndices(const pcl::PointIndicesConstPtr& indicesn)
{
	if (indicesn->indices.size() > 0)
	{
		indices_in = *indicesn;
		indices_flag = true;
	}
	else
	{
		indices_flag = false;
	}

}
void WHIFeatureEstimation::setDimBasic(int k)
{
	dimBasic = k;
}
void WHIFeatureEstimation::setSearchRadius(double radius)
{
	searchRadius = radius;
}
void WHIFeatureEstimation::setmr(double mr)
{
	meshresolution = mr;
}
void WHIFeatureEstimation::init()
{
	tree.setInputCloud(cloud_in);
	if (!indices_flag)
	{
		for (int i = 0; i < cloud_in->size(); i++)
		{
			indices_in.indices.push_back(i);
		}

	}
}
cv::Mat WHIFeatureEstimation::imagedft(cv::Mat& input)
{
	int m = cv::getOptimalDFTSize(input.rows);
	int n = cv::getOptimalDFTSize(input.cols);
	Mat padded;
	copyMakeBorder(input, padded, 0, m - input.rows, 0, n - input.cols, BORDER_CONSTANT, Scalar::all(0));
	Mat planes[] = { Mat_<double>(padded), Mat::zeros(padded.size(), CV_64F) };
	Mat complexI;
	merge(planes, 2, complexI);
	//DFT
	dft(complexI, complexI);
	split(complexI, planes);
	magnitude(planes[0], planes[1], planes[0]);
	Mat magnitudeImage = planes[0];
	magnitudeImage += Scalar::all(1);
	log(magnitudeImage, magnitudeImage);
	magnitudeImage = magnitudeImage(Rect(0, 0, magnitudeImage.cols & -2, magnitudeImage.rows & -2));
	int cx = magnitudeImage.cols / 2;
	int cy = magnitudeImage.rows / 2;
	Mat q0 = magnitudeImage(Rect(0, 0, cx, cy));
	Mat q1 = magnitudeImage(Rect(cx, 0, cx, cy));
	Mat q2 = magnitudeImage(Rect(0, cy, cx, cy));
	Mat q3 = magnitudeImage(Rect(cx, cy, cx, cy));
	Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	normalize(magnitudeImage, magnitudeImage, 0, 1, NORM_MINMAX);

	return magnitudeImage;
}
// LocalReferenceFrameEstimation
void WHIFeatureEstimation::LRFEstimator(const int& current_point_idx, Eigen::Matrix3f &rf, std::vector<int>& n_indices, std::vector<float>& n_sqr_distances)
{
	const Eigen::Vector4f& central_point = (*cloud_in)[current_point_idx].getVector4fMap();


	Eigen::Matrix<double, Eigen::Dynamic, 4> vij(n_indices.size(), 4);

	Eigen::Matrix3d cov_m = Eigen::Matrix3d::Zero();

	double distance = 0.0;
	double sum = 0.0;

	int valid_nn_points = 0;

	for (size_t i_idx = 0; i_idx < n_indices.size(); ++i_idx)
	{
		Eigen::Vector4f pt = cloud_in->points[n_indices[i_idx]].getVector4fMap();
		if (pt.head<3>() == central_point.head<3>())
			continue;

		// Difference between current point and origin
		vij.row(valid_nn_points).matrix() = (pt - central_point).cast<double>();
		vij(valid_nn_points, 3) = 0;

		distance = searchRadius - sqrt(n_sqr_distances[i_idx]);

		// Multiply vij * vij'
		cov_m += distance * (vij.row(valid_nn_points).head<3>().transpose() * vij.row(valid_nn_points).head<3>());

		sum += distance;
		valid_nn_points++;
	}

	if (valid_nn_points < 5)
	{
		//PCL_ERROR ("[pcl::%s::getLocalRF] Warning! Neighborhood has less than 5 vertexes. Aborting Local RF computation of feature point (%lf, %lf, %lf)\n", "SHOTLocalReferenceFrameEstimation", central_point[0], central_point[1], central_point[2]);
		rf.setConstant(std::numeric_limits<float>::quiet_NaN());
		cout << "invalid" << endl;
	}

	cov_m /= sum;

	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov_m);

	const double& e1c = solver.eigenvalues()[0];
	const double& e2c = solver.eigenvalues()[1];
	const double& e3c = solver.eigenvalues()[2];

	if (!pcl_isfinite(e1c) || !pcl_isfinite(e2c) || !pcl_isfinite(e3c))
	{
		//PCL_ERROR ("[pcl::%s::getLocalRF] Warning! Eigenvectors are NaN. Aborting Local RF computation of feature point (%lf, %lf, %lf)\n", "SHOTLocalReferenceFrameEstimation", central_point[0], central_point[1], central_point[2]);
		rf.setConstant(std::numeric_limits<float>::quiet_NaN());
		//exit(0);
	}

	// Disambiguation
	Eigen::Vector4d v1 = Eigen::Vector4d::Zero();
	Eigen::Vector4d v3 = Eigen::Vector4d::Zero();
	v1.head<3>().matrix() = solver.eigenvectors().col(2);
	v3.head<3>().matrix() = solver.eigenvectors().col(0);

	int plusNormal = 0, plusTangentDirection1 = 0;
	for (int ne = 0; ne < valid_nn_points; ne++)
	{
		double dp = vij.row(ne).dot(v1);
		if (dp >= 0)
			plusTangentDirection1++;

		dp = vij.row(ne).dot(v3);
		if (dp >= 0)
			plusNormal++;
	}

	//TANGENT
	plusTangentDirection1 = 2 * plusTangentDirection1 - valid_nn_points;
	if (plusTangentDirection1 == 0)
	{
		int points = 5; //std::min(valid_nn_points*2/2+1, 11);
		int medianIndex = valid_nn_points / 2;

		for (int i = -points / 2; i <= points / 2; i++)
			if (vij.row(medianIndex - i).dot(v1) > 0)
				plusTangentDirection1++;

		if (plusTangentDirection1 < points / 2 + 1)
			v1 *= -1;
	}
	else if (plusTangentDirection1 < 0)
		v1 *= -1;

	//Normal
	plusNormal = 2 * plusNormal - valid_nn_points;
	if (plusNormal == 0)
	{
		int points = 5; //std::min(valid_nn_points*2/2+1, 11);
		int medianIndex = valid_nn_points / 2;

		for (int i = -points / 2; i <= points / 2; i++)
			if (vij.row(medianIndex - i).dot(v3) > 0)
				plusNormal++;

		if (plusNormal < points / 2 + 1)
			v3 *= -1;
	}
	else if (plusNormal < 0)
		v3 *= -1;

	rf.row(0).matrix() = v1.head<3>().cast<float>();
	rf.row(2).matrix() = v3.head<3>().cast<float>();
	rf.row(1).matrix() = rf.row(2).cross(rf.row(0));

}

void WHIFeatureEstimation::compute(pcl::PointCloud<WHIFeature>::Ptr& feature)
{
	init();
	double lambda = 0.3;
	double sigmax = (2 * searchRadius) / sqrt(12);
	double sigmay = sigmax;
	double reso = searchRadius / dimBasic;
	feature->clear();

	for (int i = 0; i < indices_in.indices.size(); i++)
	{
		pcl::PointXYZ QueryPoint = cloud_in->at(indices_in.indices[i]);
		vector<int> Kind;
		vector<float> Kdist;
		tree.radiusSearch(QueryPoint, searchRadius, Kind, Kdist);
		if (Kind.size() > 5)
		{
			//LRF estimation
			const Eigen::Vector3f& central_point = (*cloud_in)[indices_in.indices[i]].getVector3fMap();
	
			int valid_nn_points = Kind.size() - 1;
			int valid_conv_point = floor(valid_nn_points*0.7);
			Eigen::Matrix<float, Eigen::Dynamic, 3> vij(valid_nn_points, 3);
			Eigen::Matrix3f cov_m = Eigen::Matrix3f::Zero();

			//pcl::PointCloud<pcl::PointXYZ>::Ptr ptcloud(new pcl::PointCloud<pcl::PointXYZ>);
			vector<double> distances(Kind.size());

			for (size_t i_idx = 0; i_idx < valid_nn_points; ++i_idx)
			{
				Eigen::Vector3f pt = cloud_in->points[Kind[i_idx + 1]].getVector3fMap();
				//ptcloud->push_back(cloud_in->points[Kind[i_idx + 1]]);
				vij.row(i_idx).matrix() = (pt - central_point);
				distances[i_idx] = searchRadius - sqrt(Kdist[i_idx]);
				if (i_idx < valid_conv_point)
					cov_m += distances[i_idx] * (vij.row(i_idx).transpose() * vij.row(i_idx));
			}


			Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(cov_m);

			// Disambiguation
			Eigen::Vector3f v1 = Eigen::Vector3f::Zero();
			Eigen::Vector3f v3 = Eigen::Vector3f::Zero();
			Eigen::Vector3f v2 = Eigen::Vector3f::Zero();

			v1.matrix() = solver.eigenvectors().col(2);
			v3.matrix() = solver.eigenvectors().col(0);
			v2.matrix() = v3.cross(v1);

			vector<float> ftemp(valid_nn_points, 0.0);
			vector<vector<float>> feature_space(3, ftemp);

			double sum0 = 0.0;
			double sum1 = 0.0;
			for (int di = 0; di < valid_nn_points; di++)
			{
				feature_space[0][di] = (vij.row(di).dot(v1));
				feature_space[1][di] = (vij.row(di).dot(v2));
				feature_space[2][di] = (vij.row(di).dot(v3));
				sum0 += feature_space[0][di];
				sum1 += feature_space[2][di];
			}

			int sign0, sign1, sign2;
			if (sum0 >= 0)
				sign0 = 1;
			else
				sign0 = -1;
			if (sum1 >= 0)
				sign2 = 1;
			else
				sign2 = -1;
			sign1 = sign0 * sign2;

			vector<int> tempn(2 * dimBasic, 0);
			vector<vector<int>> imagen(2 * dimBasic, tempn);
			vector<double> tempimg(2 * dimBasic, 0.0);
			vector<vector<double>> image(2 * dimBasic, tempimg);

			for (int di = 0; di < valid_nn_points; di++)
			{

				double lambda_ = lambda + (1 - lambda)*(distances[di] / searchRadius);
				int index_x = floor((sign0*feature_space[0][di] + searchRadius) / reso);

				int index_y = floor((sign1*feature_space[1][di] + searchRadius) / reso);

				image[index_x][index_y] = image[index_x][index_y] + lambda_ * sign2*feature_space[2][di];
				imagen[index_x][index_y] ++;
			}

			WHIFeature fea;

			cv::Mat His = cv::Mat::zeros(2 * dimBasic, 2 * dimBasic, CV_64F);

			for (int ix = 0; ix < 2 * dimBasic; ix++)
			{
				for (int iy = 0; iy < 2 * dimBasic; iy++)
				{
					if (imagen[ix][iy] > 0)
						His.at<double>(ix, iy) = image[ix][iy] / imagen[ix][iy];
				}
			}

			cv::GaussianBlur(His, His, Size(5, 5), sigmax, sigmay);
			fea.feature.resize(4 * dimBasic*dimBasic);
			fea.feature = (vector<double>)His.reshape(1, 1);
			feature->push_back(fea);

		}
	}
}

WHIFeatureEstimation::WHIFeatureEstimation()
{

}
WHIFeatureEstimation::~WHIFeatureEstimation()
{

}

/* ********************************class CorrespondenceRANSAC**********************************/
void CorrespondenceRANSAC::setInputSource(pcl::PointCloud<pcl::PointXYZ>::Ptr& in)
{
	source_in = in;
}
void CorrespondenceRANSAC::setInputTarget(pcl::PointCloud<pcl::PointXYZ>::Ptr& in)
{
	target_in = in;
}
void CorrespondenceRANSAC::setInputCorrespondences(boost::shared_ptr<pcl::Correspondences>& in)
{
	corr_in = in;
}

void CorrespondenceRANSAC::setThreshold(double thre)
{
	threshold = thre;
}
void CorrespondenceRANSAC::setIterationTimes(int it)
{
	iterationtimes = it;
}

pcl::PointIndices CorrespondenceRANSAC::getInliersIndices()
{
	return inliers_indices;
}
pcl::Correspondences CorrespondenceRANSAC::getRemainCorrespondences()
{
	pcl::Correspondences corr;
	for (int i = 0; i < inliers_indices.indices.size(); i++)
	{
		corr.push_back(corr_in->at(inliers_indices.indices[i]));
	}
	inliers_indices;
	return corr;
}
void CorrespondenceRANSAC::runRANSAC()
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr source(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr target(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr source_tr(new pcl::PointCloud<pcl::PointXYZ>);
	for (int i = 0; i < corr_in->size(); i++)
	{
		source->push_back(source_in->at(corr_in->at(i).index_query));
		target->push_back(target_in->at(corr_in->at(i).index_match));
	}
	std::srand((unsigned)time(NULL));

	pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> trans_est;
	Eigen::Matrix4f T_SVD;
	int inlier_num = 0;
	int max_inlier = 0;
	for (int ite = 0; ite < iterationtimes; ite++)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr source_sample(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr target_sample(new pcl::PointCloud<pcl::PointXYZ>);
		for (int i = 0; i < 4; i++)
		{
			int rd = rand() % (corr_in->size());
			source_sample->push_back(source_in->at(corr_in->at(rd).index_query));
			target_sample->push_back(target_in->at(corr_in->at(rd).index_match));
		}
		trans_est.estimateRigidTransformation(*source_sample, *target_sample, T_SVD);
		pcl::transformPointCloud(*source, *source_tr, T_SVD);
		vector<int> idx;
		for (int i = 0; i < source->size(); i++)
		{
			double dis = pcl::euclideanDistance(source_tr->at(i), target->at(i));
			if (dis < threshold)
			{
				idx.push_back(i);
			}
		}
		inlier_num = idx.size();
		if (inlier_num > max_inlier)
		{
			max_inlier = inlier_num;
			inliers_indices.indices.clear();
			inliers_indices.indices = idx;
		}
	}
}

CorrespondenceRANSAC::CorrespondenceRANSAC()
{

}

CorrespondenceRANSAC::~CorrespondenceRANSAC()
{

}

