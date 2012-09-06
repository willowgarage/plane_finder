cmake_minimum_required(VERSION 2.8)
project(nomad_stack)
find_package(catkin REQUIRED)
catkin_stack()

add_subdirectory(plane_mapper)
