cmake_minimum_required(VERSION 2.8)

project( Face-Tracking-Maya )

find_package( OpenCV REQUIRED )

#static please!
if(MSVC)
  set(CMAKE_CXX_FLAGS_RELEASE "/MT")
  set(CMAKE_CXX_FLAGS_DEBUG "/MTd")
endif(MSVC)

find_package(OpenCV REQUIRED)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories( ${PROJECT_SOURCE_DIR}/include/ )


# Generate lists
file(GLOB PROJECT_SOURCES src/*.cpp)
file(GLOB PROJECT_HEADERS include/*.h)

# Add executable
add_executable( ${PROJECT_NAME} ${PROJECT_SOURCES} ${PROJECT_HEADERS} )

# Link with libraries
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )
