cmake_minimum_required(VERSION 3.15)
project(tensorxx LANGUAGES CXX)

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

# _tensorxx - binary, will be installed to tensorxx/
pybind11_add_module(_tensorxx bindings/module.cpp)
target_link_libraries(_tensorxx PRIVATE tensorxx)
install(TARGETS _tensorxx
  LIBRARY DESTINATION tensorxx # --prefix <>
  RUNTIME DESTINATION tensorxx
  ARCHIVE DESTINATION tensorxx
)
install(FILES
  src/tensorxx/__init__.py
  # src/tensorxx/opencv_utils.py
  DESTINATION tensorxx
)
