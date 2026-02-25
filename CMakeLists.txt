cmake_minimum_required(VERSION 3.15)
project(tensorx_python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# pyproject.toml
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(pybind11 CONFIG REQUIRED)

add_subdirectory(Tensor-library)

# _tensorx - binaryь, will be installed to tensorx/
pybind11_add_module(_tensorx bindings/module.cpp)

target_link_libraries(_tensorx PRIVATE TensorX)

install(TARGETS _tensorx
    LIBRARY DESTINATION tensorx # --prefix <>
    RUNTIME DESTINATION tensorx
)
