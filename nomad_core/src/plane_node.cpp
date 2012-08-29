#include <vector>
#include <string>
#include <iostream>
#include <boost/foreach.hpp>
#include <boost/thread.hpp>

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/CameraInfo.h>
#include <visualization_msgs/Marker.h>
#include <std_msgs/ColorRGBA.h>
#include <geometry_msgs/Pose.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#include <opencv2/core/core.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/rgbd/rgbd.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace enc = sensor_msgs::image_encodings;

#define DEPTH_IMAGE_TOPIC "/head_mount_kinect/depth/image"
#define DEPTH_INFO_TOPIC "/head_mount_kinect/depth/camera_info"
#define RGB_IMAGE_TOPIC "/head_mount_kinect/rgb/image_color"
#define RGB_INFO_TOPIC "/head_mount_kinect/rgb/camera_info"

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

    ros::Publisher marker_pub = nh.advertise<visualization_msgs::Marker>("segmented_points", 1000);
    cv::RgbdPlane plane_finder();

    ros::Rate loop_rate(10);
    while(ros::ok()) {
        sensor_msgs::CameraInfo::ConstPtr depth_info_tmp_msg;
        sensor_msgs::Image::ConstPtr depth_image_tmp_msg;
        cv_bridge::CvImagePtr depth_image_cv_ptr;
        cv::Mat depth_k;
        cv::Mat points3d;

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

        ROS_INFO("Processing depth image");

        /* convert image and camera info to cv::Mats */
        camera_info_to_mat(depth_info_tmp_msg, depth_k);
        depth_image_cv_ptr = cv_bridge::toCvCopy(depth_image_tmp_msg, enc::TYPE_32FC1);

        /* convert depth image to 3D points */
        cv::depthTo3d(depth_image_cv_ptr->image, depth_k, points3d);

        /* estimate normals */
        cv::RgbdNormals normal_estimator(depth_image_cv_ptr->image.rows, depth_image_cv_ptr->image.cols,
                                         depth_image_cv_ptr->image.depth(), depth_k, 7);
        cv::Mat normals = normal_estimator(points3d);

        /* find planes */
        cv::RgbdPlane planes_estimator;
        cv::Mat planes_mask;
        std::vector<cv::Vec4f> planes_coefficients;
        planes_estimator(points3d, normals, planes_mask, planes_coefficients);

        ROS_INFO("Done processing depth image");

        /* publish a marker for the segmented points */
        visualization_msgs::Marker m;
        m.header.frame_id = depth_image_tmp_msg->header.frame_id;
        m.header.stamp = ros::Time::now(); //depth_image_tmp_msg->header.stamp;
        m.type = visualization_msgs::Marker::POINTS;
        m.action = visualization_msgs::Marker::ADD;
        m.ns = "segmented_points";
        m.id = 0;
        m.pose.orientation.w = 1.0;
        m.scale.x = 0.01;
        m.scale.y = 0.01;
        m.color.r = 1.0;
        m.color.a = 1.0;
        for(int row = 0; row < points3d.rows; row++) {
            for(int col = 0; col < points3d.cols; col++) {
                cv::Vec<float, 3>* point = points3d.ptr<cv::Vec<float, 3> >(row, col);
                geometry_msgs::Point p;
                p.x = (*point)(0);
                p.y = (*point)(1);
                p.z = (*point)(2);
                if((p.x != p.x) || (p.y != p.y) || (p.z != p.z))
                {
                	continue;
                }

                std_msgs::ColorRGBA c;
                int plane_i = planes_mask.at<uint8_t>(row, col);
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
                m.points.push_back(p);
                m.colors.push_back(c);
                //ROS_INFO("%d", planes_mask.at<int>(row, col));
            }
        }

        /* publish marker for segmented points */
        marker_pub.publish(m);

        ROS_INFO("Published %d points of segmented pointcloud", (int)m.points.size());
    }
}
