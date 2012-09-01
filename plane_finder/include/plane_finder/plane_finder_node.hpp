#ifndef PLANE_FINDER_H_
#define PLANE_FINDER_H_

#include <string>
#include <vector>
#include <boost/format.hpp>
#include <boost/thread.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/rgbd/rgbd.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <cv_bridge/cv_bridge.h>
#include <std_msgs/ColorRGBA.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Pose.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

namespace enc = sensor_msgs::image_encodings;

namespace plane_finder
{

struct Plane
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	std::vector<Eigen::Vector3d> points;
	Eigen::Vector4d abcd;
	Eigen::Affine3d transform;
};

void camera_info_to_mat(const sensor_msgs::CameraInfo::ConstPtr info, cv::Mat &mat) {
	mat.create(3, 3, CV_32FC1);
	for(unsigned int row = 0; row < 3; row++)
		for(unsigned int col = 0; col < 3; col++)
			mat.at<float>(row, col) = info->K[row*3 + col];
}

class PlaneFinderNode
{
	struct Params {
		/* fixed TF frame that all the planes are transformed to. markers are published
		 * in this frame
		 */
		std::string fixed_frame;

		/* whether to compute normals from the depth image and use them when deciding which
		 * points belong to the same plane
		 */
		bool use_normals;

		/* whether to apply a bilateral filter to the depth image before looking for planes.
		 * this helps deal with the depth discretization effect of the kinect at ranges > 3.0 meters
		 */
		bool use_bilateral_filter;

		/* these go directly to opencv's bilateral filter which is applied
		 * to the depth image: http://opencv.willowgarage.com/documentation/cpp/imgproc_image_filtering.html#bilateralFilter
		 *
		 * reasonable params: 1.0 for color sigma, 10.0 for spatial sigma
		 */
		double bilateral_filter_color_sigma;
		double bilateral_filter_spatial_sigma;

		/* points farther away than point_max_distance (meters) are not considered when looking for planes.
		 * kinect data gets worse at longer ranges. a reasonable value is 3.0
		 */
		double point_max_distance;

		/* planes with fewer than plane_min_points points are ignored */
		int plane_min_points;
	};

public:
	PlaneFinderNode()
		: listener_(ros::Duration(1000), true) {
		/* read params */
		nh_.param<std::string>("fixed_frame", params_.fixed_frame, "/odom_combined");
		nh_.param<bool>("use_normals", params_.use_normals, false);
		nh_.param<bool>("use_bilateral_filter", params_.use_bilateral_filter, true);
		nh_.param<double>("bilateral_filter_color_sigma", params_.bilateral_filter_color_sigma, 1.0);
		nh_.param<double>("bilateral_filter_spatial_sigma", params_.bilateral_filter_spatial_sigma, 10.0);
		nh_.param<double>("point_max_distance", params_.point_max_distance, 3.0);
		nh_.param<int>("plane_min_points", params_.plane_min_points, 1000);
	}

	void setParams(const PlaneFinderNode::Params &new_params) {
		boost::lock_guard<boost::mutex> lock(params_mutex_);
		params_ = new_params;
	}

	PlaneFinderNode::Params getParams() {
		boost::lock_guard<boost::mutex> lock(params_mutex_);
		return params_;
	}

	void msgCallback(const sensor_msgs::Image::ConstPtr& depth_image, const sensor_msgs::CameraInfo::ConstPtr& depth_info) {
		ROS_INFO("Got message");
		boost::lock_guard<boost::mutex> lock(msg_mutex_);
		depth_image_msg_ = depth_image;
		depth_info_msg_ = depth_info;
	}

	void run() {

		/* subscribe to kinect */
		message_filters::Subscriber<sensor_msgs::Image> depth_image_sub(nh_, "/head_mount_kinect/depth/image", 10);
		message_filters::Subscriber<sensor_msgs::CameraInfo> depth_info_sub(nh_, "/head_mount_kinect/depth/camera_info", 10);
		message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::CameraInfo> sync_sub(depth_image_sub, depth_info_sub, 100);
		sync_sub.registerCallback(boost::bind(&PlaneFinderNode::msgCallback, this, _1, _2));

		marker_array_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("planes", 1000);

		ros::Rate loop_rate(1);

		/* let the tf buffer get started */
		loop_rate.sleep();

		while(ros::ok()) {
			sensor_msgs::CameraInfo::ConstPtr depth_info_tmp_msg;
			sensor_msgs::Image::ConstPtr depth_image_tmp_msg;
			cv_bridge::CvImagePtr depth_image_cv_ptr;
			cv::Mat depth_k;

			ROS_INFO("Waiting for messages");

			while(ros::ok()) {
				loop_rate.sleep();
				ros::spinOnce();
				{
					boost::lock_guard<boost::mutex> lock(msg_mutex_);
					depth_info_tmp_msg = depth_info_msg_;
					depth_image_tmp_msg = depth_image_msg_;
					depth_info_msg_.reset();
					depth_image_msg_.reset();
				}
				if(depth_info_tmp_msg != NULL && depth_image_tmp_msg != NULL)
					break;
			}

			ROS_INFO("Getting transform for image");

			tf::StampedTransform transform;
			{
				listener_.waitForTransform(
						params_.fixed_frame, depth_image_tmp_msg->header.frame_id, depth_image_tmp_msg->header.stamp, ros::Duration(3.0));

				listener_.lookupTransform(params_.fixed_frame, depth_image_tmp_msg->header.frame_id, depth_image_tmp_msg->header.stamp, transform);
			}
			/*
			catch(tf::TransformException e) {
				ROS_ERROR_STREAM("Tf exception: " << e.str());
				continue;
			}*/
			Eigen::Affine3d transform_eig;
			transformTFToEigen(transform, transform_eig);

			ROS_INFO("Looking for planes in depth image");

			/* convert depth image and camera info to cv::Mats */
			camera_info_to_mat(depth_info_tmp_msg, depth_k);
			depth_image_cv_ptr = cv_bridge::toCvCopy(depth_image_tmp_msg, enc::TYPE_32FC1);

			/* process the depth image */
			this->processDepthImage(depth_image_cv_ptr->image, depth_k, transform_eig);

			ROS_INFO("Done processing depth image");

			/* publish markers for the planes */
			this->displayPlanes(marker_array_pub_);

			ROS_INFO("Published markers");
		}
	}

	/*
	 * depth_image - 32 bit depth image
	 * depth_k - camera intrinsics matrix
	 * transform - transform from sensor frame to fixed frame
	 */
	void processDepthImage(const cv::Mat &depth_image, const cv::Mat &depth_k, const Eigen::Affine3d &transform) {
		boost::lock_guard<boost::mutex> lock(params_mutex_);

		/* don't store previous planes */
		planes_.clear();

		/* filter the depth image. this helps deal with the depth discretization of the kinect */
		cv::Mat depth_image_filtered;
		if(params_.use_bilateral_filter)
			cv::bilateralFilter(depth_image, depth_image_filtered, -1, params_.bilateral_filter_color_sigma, params_.bilateral_filter_spatial_sigma);
		else
			depth_image_filtered = depth_image;

		/* convert depth image to 3D points */
		cv::Mat points3d;
		cv::depthTo3d(depth_image_filtered, depth_k, points3d);

		/* find planes in the depth image */
		cv::RgbdPlane planes_estimator;
		cv::Mat planes_mask;
		std::vector<cv::Vec4f> planes_coefficients;
		if(params_.use_normals) {
			cv::RgbdNormals normal_estimator(depth_image_filtered.rows, depth_image_filtered.cols,
					depth_image_filtered.depth(), depth_k, 7);
			cv::Mat normals = normal_estimator(points3d);
			planes_estimator(points3d, normals, planes_mask, planes_coefficients);
		}
		else {
			planes_estimator(points3d, planes_mask, planes_coefficients);
		}

		for(unsigned int plane_i = 0; plane_i < planes_coefficients.size(); plane_i++) {
			Plane new_plane;

			/* fill in plane coefficients */
			cv::Vec4f abcd = planes_coefficients.at(plane_i);
			new_plane.abcd = Eigen::Vector4d(abcd[0], abcd[1], abcd[2], abcd[3]);

			/* add all the points that belong to this plane */
			for(int row = 0; row < points3d.rows; row++) {
				for(int col = 0; col < points3d.cols; col++) {
					unsigned char cur_plane_i = planes_mask.at<unsigned char>(row, col);
					if(cur_plane_i != plane_i)
						continue;

					cv::Vec<float, 3> point_cv = points3d.at<cv::Vec<float, 3> >(row, col);
					Eigen::Vector3d point(point_cv[0], point_cv[1], point_cv[2]);
					Eigen::Vector3d point_ff = transform * point;
					new_plane.points.push_back(point_ff);
				}
			}

			if(new_plane.points.size() < params_.plane_min_points)
				continue;

			planes_.push_back(new_plane);
		}

		ROS_INFO("Found %d planes", (int)planes_.size());
	}

	void displayPlanes(ros::Publisher &marker_array_pub) {
		for(unsigned int plane_i = 0; plane_i < planes_.size(); plane_i++) {
			Plane plane = planes_[plane_i];

			std_msgs::ColorRGBA c;
			if(plane_i == 255) {
				c.r = 1.0;
				c.g = 1.0;
				c.b = 1.0;
			} else {
				c.r = ((24305 * plane_i) % 256) / 256.;
				c.g = ((1773 * plane_i) % 256) / 256.;
				c.b = ((4539 * plane_i) % 256) / 256.;
			}
			c.a = 1.0;

			visualization_msgs::MarkerArray m_arr;

			/* create a marker for the segmented points */
			visualization_msgs::Marker m;
			m.header.frame_id = params_.fixed_frame;
			m.header.stamp = ros::Time::now();
			m.type = visualization_msgs::Marker::POINTS;
			m.action = visualization_msgs::Marker::ADD;
			m.ns = (boost::format("plane_%d_points") % plane_i).str();
			m.id = 0;
			m.pose.orientation.w = 1.0;
			m.scale.x = 0.003;
			m.scale.y = 0.003;
			m.color.r = 1.0;
			m.color.a = 1.0;

			for(int point_i = 0; point_i < plane.points.size(); point_i++) {
				Eigen::Vector3d p_eig = plane.points[point_i];

				geometry_msgs::Point p;
				p.x = p_eig(0);
				p.y = p_eig(1);
				p.z = p_eig(2);
				if((p.x != p.x) || (p.y != p.y) || (p.z != p.z))
					continue;

				m.points.push_back(p);
				m.colors.push_back(c);
			}

			m_arr.markers.push_back(m);
			marker_array_pub.publish(m_arr);
		}
	}

private:
	PlaneFinderNode::Params params_;
	boost::mutex params_mutex_;
	std::vector<Plane, Eigen::aligned_allocator<Plane> > planes_;

	ros::NodeHandle nh_;
	tf::TransformListener listener_;
	ros::Publisher marker_array_pub_;
	boost::mutex msg_mutex_;
	sensor_msgs::CameraInfo::ConstPtr depth_info_msg_;
	sensor_msgs::Image::ConstPtr depth_image_msg_;

};

} /* namespace plane_finder */

#endif /* PLANE_FINDER_H_ */
