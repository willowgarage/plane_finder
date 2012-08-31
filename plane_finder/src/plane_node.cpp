#include <vector>
#include <string>
#include <iostream>
#include <boost/foreach.hpp>
#include <boost/thread.hpp>

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/CameraInfo.h>
#include <visualization_msgs/MarkerArray.h>
#include <std_msgs/ColorRGBA.h>
#include <geometry_msgs/Pose.h>
#include <tf/transform_listener.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#include <opencv2/core/core.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/rgbd/rgbd.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <plane_finder/plane_finder.hpp>

namespace enc = sensor_msgs::image_encodings;

#define FIXED_FRAME "/odom_combined"
#define USE_NORMALS false
#define MIN_PLANE_POINTS 1000
#define MARKER_TOPIC "planes"
#define DEPTH_IMAGE_TOPIC "/head_mount_kinect/depth/image"
#define DEPTH_INFO_TOPIC "/head_mount_kinect/depth/camera_info"
#define RGB_IMAGE_TOPIC "/head_mount_kinect/rgb/image_color"
#define RGB_INFO_TOPIC "/head_mount_kinect/rgb/camera_info"

/* this function is copied from tf_conversions, since it isn't linkig properly right now */
void transformTFToEigen(const tf::Transform &t, Eigen::Affine3d &e)
{
	for(int i=0; i<3; i++)
	{
		e.matrix()(i,3) = t.getOrigin()[i];
		for(int j=0; j<3; j++)
		{
			e.matrix()(i,j) = t.getBasis()[i][j];
		}
	}
	// Fill in identity in last row
	for (int col = 0 ; col < 3; col ++)
		e.matrix()(3, col) = 0;
	e.matrix()(3,3) = 1;
};

void camera_info_to_mat(const sensor_msgs::CameraInfo::ConstPtr info, cv::Mat &mat) {
    mat.create(3, 3, CV_32FC1);
    for(unsigned int row = 0; row < 3; row++)
        for(unsigned int col = 0; col < 3; col++)
            mat.at<float>(row, col) = info->K[row*3 + col];
}

boost::mutex g_msg_mutex;
sensor_msgs::CameraInfo::ConstPtr g_depth_info_msg;
sensor_msgs::Image::ConstPtr g_depth_image_msg;

void msg_callback(const sensor_msgs::Image::ConstPtr& depth_image, const sensor_msgs::CameraInfo::ConstPtr& depth_info)
{
    ROS_INFO("Got message");
    boost::lock_guard<boost::mutex> lock(g_msg_mutex);
    g_depth_image_msg = depth_image;
    g_depth_info_msg = depth_info;
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "nomad");
    ros::NodeHandle nh;

    /* subscribe to kinect */
    message_filters::Subscriber<sensor_msgs::Image> image_sub(nh, DEPTH_IMAGE_TOPIC, 1);
    message_filters::Subscriber<sensor_msgs::CameraInfo> info_sub(nh, DEPTH_INFO_TOPIC, 1);
    message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::CameraInfo> sync(image_sub, info_sub, 10);
    sync.registerCallback(boost::bind(&msg_callback, _1, _2));

    tf::TransformListener listener;

    ros::Publisher marker_array_pub = nh.advertise<visualization_msgs::MarkerArray>(MARKER_TOPIC, 1000);

    PlaneFinder plane_finder(FIXED_FRAME, USE_NORMALS, MIN_PLANE_POINTS);

    ros::Rate loop_rate(100);
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
                boost::lock_guard<boost::mutex> lock(g_msg_mutex);
                depth_info_tmp_msg = g_depth_info_msg;
                depth_image_tmp_msg = g_depth_image_msg;
                g_depth_info_msg.reset();
                g_depth_image_msg.reset();
            }
            if(depth_info_tmp_msg != NULL && depth_image_tmp_msg != NULL)
                break;
        }

        ROS_INFO("Getting transform for image");

        tf::StampedTransform transform;
        try {
        	listener.waitForTransform(
        			FIXED_FRAME, depth_image_tmp_msg->header.frame_id, depth_image_tmp_msg->header.stamp, ros::Duration(3.0));

        	listener.lookupTransform(FIXED_FRAME, depth_image_tmp_msg->header.frame_id, depth_image_tmp_msg->header.stamp, transform);
        }
        catch(tf::TransformException) {
        	ROS_ERROR("Tf exception!");
        	continue;
        }
        Eigen::Affine3d transform_eig;
        transformTFToEigen(transform, transform_eig);

        ROS_INFO("Looking for planes in depth image");

        /* convert depth image and camera info to cv::Mats */
        camera_info_to_mat(depth_info_tmp_msg, depth_k);
        depth_image_cv_ptr = cv_bridge::toCvCopy(depth_image_tmp_msg, enc::TYPE_32FC1);

        /* process the depth image */
        plane_finder.processDepthImage(depth_image_cv_ptr->image, depth_k, transform_eig);

        ROS_INFO("Done processing depth image");

        /* publish markers for the planes */
        plane_finder.displayPlanes(marker_array_pub);

        ROS_INFO("Published markers");
    }
}
