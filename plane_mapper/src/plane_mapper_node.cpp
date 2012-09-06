#include <ros/ros.h>

#include <plane_mapper/plane_mapper_node.hpp>

int main(int argc, char **argv) {
    ros::init(argc, argv, "plane_finder");
    plane_finder::PlaneFinderNode plane_node;
    plane_node.run();
}
