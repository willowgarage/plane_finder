#include <boost/foreach.hpp>
#include <vector>
#include <string>
#include <iostream>

#include <ros/ros.h>

#include <opencv2/core/core.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/rgbd/rgbd.hpp>
#include <pcl/ros/conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>

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

int main(int argc, char **argv) {
    ros::init(argc, argv, "nomad");
    ros::NodeHandle n;

    if(argc < 2) {
        ROS_ERROR("Usage: %s <bagfile>", argv[0]);
        return -1;
    }
    char *bag_filename = argv[1];

    rosbag::Bag bag;
    bag.open(bag_filename, rosbag::bagmode::Read);

    cv::RgbdPlane plane_finder();

    rosbag::View view(bag);
    ROS_INFO("Bag has %d messages", view.size());

    sensor_msgs::CameraInfo::ConstPtr depth_info_msg;
    sensor_msgs::CameraInfo::ConstPtr rgb_info_msg;
    sensor_msgs::Image::ConstPtr depth_image_msg;
    sensor_msgs::Image::ConstPtr rgb_image_msg;

    BOOST_FOREACH(rosbag::MessageInstance const m, view) {
        std::string topic = m.getTopic();
        ROS_INFO("Message on topic %s", topic.c_str());

        if(topic == DEPTH_IMAGE_TOPIC)
            depth_image_msg = m.instantiate<sensor_msgs::Image>();
        else if(topic == DEPTH_INFO_TOPIC)
            depth_info_msg = m.instantiate<sensor_msgs::CameraInfo>();
        else if(topic == RGB_IMAGE_TOPIC)
            rgb_image_msg = m.instantiate<sensor_msgs::Image>();
        else if(topic == RGB_INFO_TOPIC)
            rgb_info_msg = m.instantiate<sensor_msgs::CameraInfo>();
        else
            ROS_INFO("Ignoring message on topic %s", topic.c_str());
    }
    bag.close();

    if(depth_info_msg == NULL) {
        ROS_ERROR("Missing depth info message");
        return -1;
    }
    if(depth_image_msg == NULL) {
        ROS_ERROR("Missing depth image message");
        return -1;
    }
    if(rgb_info_msg == NULL) {
        ROS_ERROR("Missing rgb info message");
        return -1;
    }
    if(rgb_image_msg == NULL) {
        ROS_ERROR("Missing rgb image message");
        return -1;
    }

    cv_bridge::CvImagePtr depth_image_cv_ptr = cv_bridge::toCvCopy(depth_image_msg, enc::TYPE_32FC1);
    cv_bridge::CvImagePtr rgb_image_cv_ptr = cv_bridge::toCvCopy(rgb_image_msg, enc::BGR8);

    cv::Mat depth_k;
    camera_info_to_mat(depth_info_msg, depth_k);

    //cv::imshow("depth", depth_image_cv_ptr->image);
    //cv::waitKey(300000);

    std::cout << depth_k << "\n";

    cv::Mat rgb_k;
    camera_info_to_mat(rgb_info_msg, rgb_k);

    cv::Mat points3d;
    cv::depthTo3d(depth_image_cv_ptr->image, depth_k, points3d);
    for(int row = 0; row < points3d.rows; row++) {
        for(int col = 0; col < points3d.cols; col++) {
            cv::Vec<float, 3>* point = points3d.ptr<cv::Vec<float, 3> >(row, col);
            //ROS_INFO("%f %f %f", (*point)(0), (*point)(1), (*point)(2));
        }
    }


    /* estimate normals */
    cv::RgbdNormals normal_estimator(depth_image_cv_ptr->image.rows, depth_image_cv_ptr->image.cols,
                                     depth_image_cv_ptr->image.depth(), depth_k);
    cv::Mat normals = normal_estimator(points3d);

    /* find planes */
    cv::RgbdPlane planes_estimator;
    cv::Mat planes_mask;
    std::vector<cv::Vec4f> planes_coefficients;
    planes_estimator(points3d, normals, planes_mask, planes_coefficients);

    //cv::imshow("planes", planes_mask);
    //cv::waitKey(300000);

    /* make a pointcloud of the segmented points */
    pcl::PointCloud<pcl::PointXYZRGB> segmented_cloud;
    for(int row = 0; row < points3d.rows; row++) {
        for(int col = 0; col < points3d.cols; col++) {
            cv::Vec<float, 3>* point = points3d.ptr<cv::Vec<float, 3> >(row, col);
            pcl::PointXYZRGB p;
            p.x = (*point)(0);
            p.y = (*point)(1);
            p.z = (*point)(2);
            int plane_i = planes_mask.at<uint8_t>(row, col);
            if(plane_i == 255) {
                p.r = 255;
                p.g = 255;
                p.b = 255;
            } else {
                p.r = (24305 * plane_i) % 256;
                p.g = (1773 * plane_i) % 256;
                p.b = (4539 * plane_i) % 256;
            }
            segmented_cloud.push_back(p);
            //ROS_INFO("%d", planes_mask.at<int>(row, col));
        }
    }
    sensor_msgs::PointCloud2 segmented_cloud_msg;
    pcl::toROSMsg(segmented_cloud, segmented_cloud_msg);
    segmented_cloud_msg.header.frame_id = depth_image_msg->header.frame_id;
    segmented_cloud_msg.header.stamp = depth_image_msg->header.stamp;

    ros::Publisher points_pub = n.advertise<sensor_msgs::PointCloud2>("segmented_points", 1000);
    ros::Rate loop_rate(10);
    while(ros::ok()) {
        points_pub.publish(segmented_cloud_msg);
        loop_rate.sleep();
    }

    /* print out the planes */
    for(unsigned int ii=0; ii < planes_coefficients.size(); ii++) {
        ROS_INFO("%f %f %f %f", planes_coefficients[ii](0), planes_coefficients[ii](1), planes_coefficients[ii](2), planes_coefficients[ii](3));
    }

}
