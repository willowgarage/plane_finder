project(plane_finder)

include_directories(include)

# find ros dependencies
find_package(ROS REQUIRED roscpp sensor_msgs ecto_opencv cv_bridge message_filters tf)
include_directories(${ROS_INCLUDE_DIRS})
link_directories(${ROS_LIBRARY_DIRS})

# find boost
find_package(Boost COMPONENTS thread signals REQUIRED)

# find eigen
find_package(Eigen)
include_directories(${Eigen_INCLUDE_DIRS})

# build plane finding ROS node
add_executable(plane_node src/plane_node.cpp)
target_link_libraries(plane_node ${ROS_LIBRARIES})
target_link_libraries(plane_node ${Boost_THREAD_LIBRARY} ${Boost_SIGNALS_LIBRARY})

install(TARGETS plane_node
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  )
