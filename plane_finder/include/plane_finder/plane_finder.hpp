#ifndef PLANE_FINDER_H_
#define PLANE_FINDER_H_

#include <string>
#include <vector>
#include <boost/format.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <std_msgs/ColorRGBA.h>
#include <geometry_msgs/Point.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

struct Plane
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	std::vector<Eigen::Vector3d> points;
	Eigen::Vector4d abcd;
	Eigen::Affine3d transform;
};

class PlaneFinder
{
public:
	PlaneFinder(const std::string &fixed_frame, bool use_normals = false, int min_plane_points = 0)
		: fixed_frame_(fixed_frame), use_normals_(use_normals), min_plane_points_(min_plane_points) {}

	/*
	 * depth_image - 32 bit depth image
	 * depth_k - camera intrinsics matrix
	 * transform - transform from sensor frame to fixed frame
	 */
	void processDepthImage(const cv::Mat &depth_image, const cv::Mat &depth_k, const Eigen::Affine3d &transform) {
        cv::Mat points3d;

        /* for now, don't store previous planes */
        planes_.clear();

        /* convert depth image to 3D points */
        cv::depthTo3d(depth_image, depth_k, points3d);

        /* find planes in the depth image */
        cv::RgbdPlane planes_estimator;
        cv::Mat planes_mask;
        std::vector<cv::Vec4f> planes_coefficients;
        if(use_normals_) {
        	cv::RgbdNormals normal_estimator(depth_image.rows, depth_image.cols,
        			depth_image.depth(), depth_k, 7);
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

            if(new_plane.points.size() < min_plane_points_)
            	continue;

            planes_.push_back(new_plane);
        }
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
			m.header.frame_id = fixed_frame_;
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
	std::string fixed_frame_;
	bool use_normals_;
	int min_plane_points_;
	std::vector<Plane, Eigen::aligned_allocator<Plane> > planes_;

};

#endif /* PLANE_FINDER_H_ */
