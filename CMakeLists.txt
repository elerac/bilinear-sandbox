cmake_minimum_required(VERSION 3.18...3.30)

project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

if(NOT SKBUILD)
  message(WARNING "This CMake project is intended to be built through scikit-build-core.")
endif()

find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)
find_package(Threads REQUIRED)

nanobind_add_module(
  _native
  NOMINSIZE
  cpp/bindings/_native.cpp
  cpp/src/demosaicing.cpp
)

target_compile_features(_native PRIVATE cxx_std_17)
target_include_directories(_native PRIVATE cpp/include)
target_link_libraries(_native PRIVATE Threads::Threads)

install(TARGETS _native LIBRARY DESTINATION fastimg)
