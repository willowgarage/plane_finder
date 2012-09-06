project(plane_mapper)

include_directories(include)

# find ros dependencies
find_package(ROS REQUIRED roscpp sensor_msgs ecto_opencv cv_bridge message_filters tf tf_conversions)
include_directories(${ROS_INCLUDE_DIRS})
link_directories(${ROS_LIBRARY_DIRS})
message(STATUS "ROS_LIBRARY_DIRS: "${ROS_LIBRARY_DIRS})

# hack because ROS_LIBRARY_DIRS is empty for some reason! -jbinney
link_directories(/opt/ros/groovy/lib)

# find boost
find_package(Boost COMPONENTS thread signals REQUIRED)

# find eigen
find_package(Eigen)
include_directories(${Eigen_INCLUDE_DIRS})

# build plane finding ROS node
add_executable(plane_mapper_node src/plane_mapper_node.cpp)
target_link_libraries(plane_mapper_node ${ROS_LIBRARIES})
target_link_libraries(plane_mapper_node ${Boost_THREAD_LIBRARY} ${Boost_SIGNALS_LIBRARY})

install(TARGETS plane_mapper_node
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  )

catkin_project(${PROJECT_NAME}
  INCLUDE_DIRS include
  LIBRARIES
)
